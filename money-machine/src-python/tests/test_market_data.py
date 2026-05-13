"""
Tests for the market-data layer.

Covers:

  - BinancePublicClient: parses the raw kline array shape correctly,
    pages through multiple requests when the requested range
    exceeds the per-request cap, retries on transport errors, and
    trims candles outside the requested window.
  - MarketDataService: reads-through to the upstream client on cold
    cache, writes the cache atomically, serves subsequent calls
    entirely from cache without re-hitting the upstream, detects
    head/tail gaps and only fetches what is missing, returns a
    UTC-indexed DataFrame with the canonical OHLCV columns the
    backtest expects.
  - End-to-end glue: a backfill run produces a DataFrame that the
    real Backtest engine can consume without modification.

No real network calls; the HTTP transport is a fake that replays
pre-built JSON payloads.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.backtest import Backtest, BacktestConfig, FixedSlippage, FlatCommission  # noqa: E402
from engine.market_data import (  # noqa: E402
    BinancePublicClient,
    Candle,
    MarketDataService,
)
from engine.market_data.binance_client import INTERVAL_MS  # noqa: E402
from engine.strategies.base import Strategy, TradingSignal  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


HOUR_MS = INTERVAL_MS["1h"]


def _kline_row(open_ms: int, base: float) -> List[Any]:
    """Build one Binance-shaped kline row (12-element array)."""
    return [
        open_ms,
        f"{base:.2f}",
        f"{base + 1.0:.2f}",
        f"{base - 1.0:.2f}",
        f"{base + 0.5:.2f}",
        "100.0",
        open_ms + HOUR_MS - 1,
        "5000",
        10,
        "50",
        "2500",
        "0",
    ]


def _build_klines(start_ms: int, count: int, base: float = 100.0) -> List[List[Any]]:
    return [
        _kline_row(start_ms + i * HOUR_MS, base + i)
        for i in range(count)
    ]


class _FakeHttpFetch:
    """Async fake that returns pre-scripted JSON bytes per call."""

    def __init__(self, responses: List[bytes]) -> None:
        self._responses = list(responses)
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def __call__(self, url: str, params: Dict[str, Any]) -> bytes:
        self.calls.append((url, dict(params)))
        if not self._responses:
            raise AssertionError(
                f"Unexpected extra HTTP fetch call: url={url} params={params}"
            )
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# BinancePublicClient: parsing + paging
# ---------------------------------------------------------------------------


def test_binance_client_parses_kline_array_into_candles() -> None:
    async def scenario() -> None:
        rows = _build_klines(start_ms=1_700_000_000_000, count=3, base=100.0)
        fetch = _FakeHttpFetch([json.dumps(rows).encode("utf-8")])
        client = BinancePublicClient(http_fetch=fetch)

        candles = await client.fetch_klines(
            "BTCUSDT", "1h",
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_000_000 + 3 * HOUR_MS,
        )

        assert len(candles) == 3
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].open == pytest.approx(100.0)
        assert candles[0].close == pytest.approx(100.5)
        assert candles[1].open == pytest.approx(101.0)
        # Single page request was issued.
        assert len(fetch.calls) == 1

    _run(scenario())


def test_binance_client_pages_when_range_exceeds_max_per_request() -> None:
    async def scenario() -> None:
        start = 1_700_000_000_000
        # Configure a tiny per-request cap to force paging.
        from engine.market_data.binance_client import BinancePublicConfig
        config = BinancePublicConfig(max_per_request=3)

        # Build 7 hours of candles; with cap=3 the client should
        # issue 3 paged requests (3 + 3 + 1).
        page_a = _build_klines(start_ms=start, count=3, base=100.0)
        page_b = _build_klines(start_ms=start + 3 * HOUR_MS, count=3, base=103.0)
        page_c = _build_klines(start_ms=start + 6 * HOUR_MS, count=1, base=106.0)

        fetch = _FakeHttpFetch([
            json.dumps(page_a).encode("utf-8"),
            json.dumps(page_b).encode("utf-8"),
            json.dumps(page_c).encode("utf-8"),
        ])
        client = BinancePublicClient(http_fetch=fetch, config=config)

        candles = await client.fetch_klines(
            "BTCUSDT", "1h",
            start_ms=start,
            end_ms=start + 7 * HOUR_MS,
        )

        assert len(candles) == 7
        # Three paged requests; each subsequent startTime advanced.
        assert len(fetch.calls) == 3
        starts = [int(c[1]["startTime"]) for c in fetch.calls]
        assert starts == [start, start + 3 * HOUR_MS, start + 6 * HOUR_MS]

    _run(scenario())


def test_binance_client_trims_candles_outside_requested_window() -> None:
    async def scenario() -> None:
        start = 1_700_000_000_000
        # Venue is inclusive; engineer a response that overshoots
        # end_ms. The client must trim it.
        rows = _build_klines(start_ms=start, count=5)
        fetch = _FakeHttpFetch([json.dumps(rows).encode("utf-8")])
        client = BinancePublicClient(http_fetch=fetch)

        candles = await client.fetch_klines(
            "BTCUSDT", "1h",
            start_ms=start,
            end_ms=start + 3 * HOUR_MS,  # only 3 candles fit
        )

        assert len(candles) == 3
        assert candles[-1].open_time_ms < start + 3 * HOUR_MS

    _run(scenario())


def test_binance_client_retries_on_transport_error() -> None:
    async def scenario() -> None:
        calls = []

        async def flaky(url: str, params: Dict[str, Any]) -> bytes:
            calls.append((url, dict(params)))
            if len(calls) < 3:
                raise ConnectionError("transient")
            return json.dumps(_build_klines(start_ms=0, count=1)).encode("utf-8")

        from engine.market_data.binance_client import BinancePublicConfig
        client = BinancePublicClient(
            http_fetch=flaky,
            config=BinancePublicConfig(max_retries=3, backoff_base_seconds=0.0),
        )

        # Speed: monkey-patch asyncio.sleep.
        original_sleep = asyncio.sleep

        async def _noop(_d: float) -> None:
            return None

        asyncio.sleep = _noop  # type: ignore[assignment]
        try:
            candles = await client.fetch_klines("BTCUSDT", "1h", 0, HOUR_MS)
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert len(candles) == 1
        assert len(calls) == 3


    _run(scenario())


def test_binance_public_config_rejects_invalid_values() -> None:
    from engine.market_data.binance_client import BinancePublicConfig

    with pytest.raises(ValueError):
        BinancePublicConfig(max_per_request=0)
    with pytest.raises(ValueError):
        BinancePublicConfig(max_per_request=5000)  # Binance caps at 1000
    with pytest.raises(ValueError):
        BinancePublicConfig(request_timeout_seconds=0.0)
    with pytest.raises(ValueError):
        BinancePublicConfig(max_retries=-1)
    with pytest.raises(ValueError):
        BinancePublicConfig(backoff_base_seconds=10.0, backoff_max_seconds=1.0)
    with pytest.raises(ValueError):
        BinancePublicConfig(base_url="")


def test_service_rejects_unknown_interval_in_get_ohlcv(tmp_path: Path) -> None:
    """Even when the cache happens to be complete, an unknown
    interval must fail fast rather than slip through with a 1ms
    granularity fallback in `_missing_ranges`.
    """
    async def scenario() -> None:
        client = _StubClient(_candles(0, 5))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        with pytest.raises(ValueError):
            await svc.get_ohlcv("BTC/USDT", "37s", 0, 5 * HOUR_MS)

    _run(scenario())


def test_binance_client_rejects_unknown_interval() -> None:
    async def scenario() -> None:
        client = BinancePublicClient(http_fetch=_FakeHttpFetch([]))
        with pytest.raises(ValueError):
            await client.fetch_klines("BTCUSDT", "37s", 0, 1)

    _run(scenario())


def test_binance_client_skips_malformed_rows() -> None:
    async def scenario() -> None:
        bad = [
            _kline_row(0, 100.0),
            [None, None, None],  # malformed shape
            _kline_row(HOUR_MS, 101.0),
        ]
        fetch = _FakeHttpFetch([json.dumps(bad).encode("utf-8")])
        client = BinancePublicClient(http_fetch=fetch)
        candles = await client.fetch_klines("BTCUSDT", "1h", 0, 2 * HOUR_MS)
        # Two valid candles, malformed row dropped.
        assert len(candles) == 2

    _run(scenario())


# ---------------------------------------------------------------------------
# MarketDataService: caching and gap detection
# ---------------------------------------------------------------------------


class _StubClient:
    """In-memory client that returns slices of a fixed candle list."""

    def __init__(self, candles: List[Candle]) -> None:
        self.candles = candles
        self.calls: List[Tuple[str, str, int, int]] = []

    async def fetch_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> List[Candle]:
        self.calls.append((symbol, interval, start_ms, end_ms))
        return [c for c in self.candles if start_ms <= c.open_time_ms < end_ms]


def _candles(start_ms: int, count: int) -> List[Candle]:
    return [
        Candle(
            open_time_ms=start_ms + i * HOUR_MS,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=10.0,
        )
        for i in range(count)
    ]


def test_service_cold_cache_calls_upstream_and_writes_csv(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _StubClient(_candles(0, 5))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        df = await svc.get_ohlcv("BTC/USDT", "1h", 0, 5 * HOUR_MS)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 5
        assert str(df.index.tz) == "UTC"
        # Upstream was called exactly once with the full range.
        assert len(client.calls) == 1
        assert client.calls[0] == ("BTC/USDT", "1h", 0, 5 * HOUR_MS)
        # Cache file exists on disk.
        cache_files = list(tmp_path.glob("*.csv"))
        assert len(cache_files) == 1

    _run(scenario())


def test_service_warm_cache_does_not_call_upstream(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _StubClient(_candles(0, 5))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        # Prime the cache.
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 5 * HOUR_MS)
        # Now request a strictly inner range; upstream must not be hit.
        client.calls.clear()
        df = await svc.get_ohlcv("BTC/USDT", "1h", HOUR_MS, 4 * HOUR_MS)
        assert len(df) == 3
        assert client.calls == []

    _run(scenario())


def test_service_fetches_only_missing_tail_segment(tmp_path: Path) -> None:
    async def scenario() -> None:
        # Pre-populate cache with the first 3 candles.
        client = _StubClient(_candles(0, 8))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 3 * HOUR_MS)

        client.calls.clear()
        # Request a window that extends past the cached range.
        df = await svc.get_ohlcv("BTC/USDT", "1h", 0, 6 * HOUR_MS)
        assert len(df) == 6
        # Exactly one upstream call, and it covers only the tail
        # gap from open_time 3*HOUR_MS to 6*HOUR_MS.
        assert len(client.calls) == 1
        _, _, gap_start, gap_end = client.calls[0]
        # Strict equality: any overlap with the cached range means we
        # over-fetched and the gap-detection regressed.
        assert gap_start == 3 * HOUR_MS
        assert gap_end == 6 * HOUR_MS

    _run(scenario())


def test_service_fetches_only_missing_head_segment(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _StubClient(_candles(0, 8))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        # Cache the middle: candles 3..6.
        await svc.get_ohlcv("BTC/USDT", "1h", 3 * HOUR_MS, 6 * HOUR_MS)

        client.calls.clear()
        df = await svc.get_ohlcv("BTC/USDT", "1h", 0, 6 * HOUR_MS)
        # Resulting frame has 6 candles starting from 0.
        assert len(df) == 6
        # One head-fill call covering [0, 3*HOUR_MS).
        assert len(client.calls) == 1
        _, _, gap_start, gap_end = client.calls[0]
        assert gap_start == 0
        assert gap_end == 3 * HOUR_MS

    _run(scenario())


def test_service_dedupes_overlapping_writes(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _StubClient(_candles(0, 5))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 5 * HOUR_MS)
        # Force a second fetch over an overlapping range. The cache
        # should keep one row per open_time.
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 10 * HOUR_MS)
        df = pd.read_csv(next(tmp_path.glob("*.csv")))
        assert df["open_time_ms"].is_unique

    _run(scenario())


# ---------------------------------------------------------------------------
# End-to-end glue: a backfilled DataFrame is consumable by Backtest.
# ---------------------------------------------------------------------------


class _AlwaysHold(Strategy):
    name = "always-hold"

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        return TradingSignal(
            symbol=self.symbol, action="HOLD", confidence=0.0,
            strategy=self.name,
        )


def test_service_recovers_from_corrupted_cache(tmp_path: Path) -> None:
    """A manually-edited cache file with a non-numeric cell must not
    crash _load_cache. The service should log a warning and fall
    back to an empty cache so the next fetch repopulates it.
    """
    async def scenario() -> None:
        client = _StubClient(_candles(0, 5))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 5 * HOUR_MS)

        # Hand-edit one row to have a non-numeric `high` value.
        path = next(tmp_path.glob("*.csv"))
        lines = path.read_text().splitlines()
        # lines[0] is the header; corrupt the middle row.
        parts = lines[2].split(",")
        parts[2] = "not-a-number"
        lines[2] = ",".join(parts)
        path.write_text("\n".join(lines) + "\n")

        # Should not raise; the corrupted row is dropped silently.
        df = await svc.get_ohlcv("BTC/USDT", "1h", 0, 5 * HOUR_MS)
        # At least 4 of the 5 original candles survive (the bad
        # row is dropped).
        assert len(df) >= 4

    _run(scenario())


def test_concurrent_get_ohlcv_for_same_key_does_not_lose_writes(tmp_path: Path) -> None:
    """Two concurrent callers fetching the same (symbol, interval)
    must not clobber each other's cache writes.
    """
    async def scenario() -> None:
        client = _StubClient(_candles(0, 10))
        svc = MarketDataService(client=client, cache_dir=tmp_path)

        async def fetcher(start: int, end: int) -> pd.DataFrame:
            return await svc.get_ohlcv("BTC/USDT", "1h", start, end)

        a, b = await asyncio.gather(
            fetcher(0, 5 * HOUR_MS),
            fetcher(5 * HOUR_MS, 10 * HOUR_MS),
        )
        assert len(a) == 5
        assert len(b) == 5
        # Final cache contains both ranges deduped (10 unique candles).
        final = await svc.get_ohlcv("BTC/USDT", "1h", 0, 10 * HOUR_MS)
        assert len(final) == 10

    _run(scenario())


def test_binance_client_strips_colon_contract_suffix() -> None:
    """ccxt-style symbols like 'BTC/USDT:USDT' must be normalised to
    'BTCUSDT' before hitting Binance; otherwise Binance rejects
    them as invalid and the fetch returns empty.
    """
    async def scenario() -> None:
        rows = _build_klines(start_ms=0, count=2)
        fetch = _FakeHttpFetch([json.dumps(rows).encode("utf-8")])
        client = BinancePublicClient(http_fetch=fetch)
        await client.fetch_klines("BTC/USDT:USDT", "1h", 0, 2 * HOUR_MS)
        # The request that hit the API used the clean symbol.
        _, params = fetch.calls[0]
        assert params["symbol"] == "BTCUSDT"

    _run(scenario())


def test_cache_path_does_not_collide_for_distinct_symbols(tmp_path: Path) -> None:
    """The old `replace('/', '')` rule mapped AB/CD and A/BCD to
    the same filename, so once one warmed the cache the other
    would silently read someone else's candles. The new escape
    preserves the slash position so distinct markets stay
    distinct.
    """
    async def scenario() -> None:
        # Two stub clients with different candle universes.
        client_abcd = _StubClient(_candles(0, 5))
        client_a_bcd = _StubClient(
            [
                Candle(
                    open_time_ms=i * HOUR_MS,
                    open=500.0 + i,
                    high=501.0 + i,
                    low=499.0 + i,
                    close=500.5 + i,
                    volume=10.0,
                )
                for i in range(5)
            ]
        )
        # Both services point at the same cache dir.
        svc_abcd = MarketDataService(client=client_abcd, cache_dir=tmp_path)
        svc_a_bcd = MarketDataService(client=client_a_bcd, cache_dir=tmp_path)

        a = await svc_abcd.get_ohlcv("AB/CD", "1h", 0, 5 * HOUR_MS)
        b = await svc_a_bcd.get_ohlcv("A/BCD", "1h", 0, 5 * HOUR_MS)
        # The two services produced different bars because they
        # wrote to different cache files.
        assert a.iloc[0]["close"] != b.iloc[0]["close"]
        # Confirm two distinct files were created.
        files = sorted(p.name for p in tmp_path.glob("*.csv"))
        assert len(files) == 2

    _run(scenario())


def test_missing_ranges_detects_internal_hole(tmp_path: Path) -> None:
    """A cache with an internal hole (e.g. one row was dropped by
    _load_cache's malformed-row filter) must trigger a refetch of
    the hole on the next get_ohlcv, not silently return an
    incomplete window.
    """
    async def scenario() -> None:
        # Engineer a stub client with a fixed 10-candle universe.
        client = _StubClient(_candles(0, 10))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 10 * HOUR_MS)

        # Manually punch a hole in the cache by editing the CSV.
        path = next(tmp_path.glob("*.csv"))
        lines = path.read_text().splitlines()
        # lines[0]=header, lines[1..10]=candles. Drop candle 5.
        del lines[5]
        path.write_text("\n".join(lines) + "\n")

        client.calls.clear()
        await svc.get_ohlcv("BTC/USDT", "1h", 0, 10 * HOUR_MS)
        # One upstream call covering the internal hole.
        assert len(client.calls) == 1
        _, _, gap_start, gap_end = client.calls[0]
        # The dropped candle's open_time was 4*HOUR_MS (0-indexed
        # row 5 = original index 4); the hole spans
        # [4*HOUR_MS, 5*HOUR_MS).
        assert gap_start == 4 * HOUR_MS
        assert gap_end == 5 * HOUR_MS

    _run(scenario())


def test_service_output_feeds_backtest_engine(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = _StubClient(_candles(1_700_000_000_000, 30))
        svc = MarketDataService(client=client, cache_dir=tmp_path)
        df = await svc.get_ohlcv(
            "BTC/USDT", "1h",
            1_700_000_000_000,
            1_700_000_000_000 + 30 * HOUR_MS,
        )
        bt = Backtest(
            strategy=_AlwaysHold("BTC/USDT"),
            commission=FlatCommission(rate=0.0),
            slippage=FixedSlippage(bps=0.0),
            config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
        )
        result = bt.run(df)
        # HOLD-only strategy: no trades, flat equity, no NaN anywhere.
        assert result.metrics.num_trades == 0
        assert (result.equity_curve == 10_000.0).all()
        assert result.equity_curve.notna().all()

    _run(scenario())
