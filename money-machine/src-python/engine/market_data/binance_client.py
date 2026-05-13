"""
Binance public klines client.

Uses the public REST endpoint at `https://api.binance.com/api/v3/klines`
which requires no API key and is rate-limited per IP (1200 requests
per minute as of 2024). For backfilling a 6-month hourly history
that comes out to ~4400 candles per symbol = ~5 paged requests,
well inside any reasonable budget.

The class only knows how to talk to Binance; cache logic lives in
`service.MarketDataService`. The HTTP transport is injectable so
tests can replay scripted responses without touching the network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .protocols import INTERVAL_MS, Candle

logger = logging.getLogger(__name__)


HttpFetch = Callable[[str, Dict[str, Any]], Awaitable[bytes]]
"""Async HTTP GET: ``await fetch(url, params) -> body_bytes``."""


@dataclass
class BinancePublicConfig:
    base_url: str = "https://api.binance.com"
    max_per_request: int = 1000
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0


class BinancePublicClient:
    """Calls `/api/v3/klines` with paged backfill."""

    def __init__(
        self,
        http_fetch: HttpFetch,
        config: Optional[BinancePublicConfig] = None,
    ) -> None:
        self.http = http_fetch
        self.config = config or BinancePublicConfig()

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Candle]:
        if interval not in INTERVAL_MS:
            raise ValueError(
                f"unknown interval {interval!r}. Known: {sorted(INTERVAL_MS)}"
            )
        if end_ms <= start_ms:
            return []

        symbol_clean = symbol.replace("/", "").upper()
        per_request = self.config.max_per_request
        interval_ms = INTERVAL_MS[interval]
        out: List[Candle] = []
        cursor = start_ms

        while cursor < end_ms:
            # Binance returns up to `limit` candles starting at
            # `startTime`. Their inclusive window means the next
            # paging cursor is the last candle's open_time + one
            # interval, which keeps us from refetching the boundary.
            params = {
                "symbol": symbol_clean,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": per_request,
            }
            body = await self._get_with_retries("/api/v3/klines", params)
            page = self._parse(body)
            if not page:
                break
            out.extend(page)
            last_open = page[-1].open_time_ms
            next_cursor = last_open + interval_ms
            if next_cursor <= cursor:
                # Defensive: never go backwards. Should never happen
                # but a misbehaving venue would otherwise infinite-loop.
                break
            cursor = next_cursor

        # Trim anything that snuck past end_ms (Binance is inclusive).
        return [c for c in out if start_ms <= c.open_time_ms < end_ms]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_with_retries(
        self, path: str, params: Dict[str, Any]
    ) -> bytes:
        url = self.config.base_url.rstrip("/") + path
        attempts = max(1, self.config.max_retries + 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    self.http(url, params),
                    timeout=self.config.request_timeout_seconds,
                )
            except Exception as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                delay = min(
                    self.config.backoff_base_seconds * (2 ** attempt),
                    self.config.backoff_max_seconds,
                )
                logger.info(
                    "Binance fetch retry %d/%d after %.2fs: %s",
                    attempt + 1,
                    attempts - 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse(body: bytes) -> List[Candle]:
        """Parse the Binance kline array response into Candle objects.

        Binance returns a list of 12-element arrays per candle:
            [openTime, open, high, low, close, volume, closeTime,
             quoteAssetVolume, trades, takerBuyBase, takerBuyQuote,
             ignore]
        We only keep the first six fields.
        """
        try:
            rows = json.loads(body)
        except ValueError as exc:
            raise ValueError(f"Binance response was not valid JSON: {exc}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Binance response was not a list: {type(rows).__name__}")
        out: List[Candle] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                out.append(
                    Candle(
                        open_time_ms=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                )
            except (TypeError, ValueError):
                # Skip a single malformed row rather than fail the
                # whole page. Binance has historically returned
                # nulls during planned maintenance windows.
                continue
        return out
