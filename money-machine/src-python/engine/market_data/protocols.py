"""
Shared types for the market-data layer.

Kept in their own module so a stub `MarketDataClient` in tests can
import them without dragging in `aiohttp` / `requests`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol


#: Canonical interval-code-to-milliseconds map. Shared by both the
#: Binance client and the cache service so gap detection knows the
#: granularity of one candle.
INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "6h": 6 * 3_600_000,
    "8h": 8 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": 24 * 3_600_000,
    "3d": 3 * 24 * 3_600_000,
    "1w": 7 * 24 * 3_600_000,
}


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle. Times are UTC epoch milliseconds.

    Mirrors the Binance kline schema (`open_time`, `open`, `high`,
    `low`, `close`, `volume`) so the public client can construct
    these directly from the API response.
    """

    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataClient(Protocol):
    """Async source of historical OHLCV candles."""

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Candle]:
        """Return a list of `Candle` between `start_ms` (inclusive)
        and `end_ms` (exclusive), in chronological order. The
        implementation is responsible for paging if the venue caps
        per-request size (Binance caps at 1000 candles).
        """
