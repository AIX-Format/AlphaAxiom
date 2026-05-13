"""
On-disk OHLCV cache + DataFrame builder.

`MarketDataService` sits between any `MarketDataClient` and the rest
of the engine. It:

  - Looks for cached candles on disk for the (symbol, interval)
    pair. The cache layout is one CSV file per pair under
    `cache_dir/<symbol>_<interval>.csv` with the canonical OHLCV
    columns the backtest module expects.
  - Determines which segments of the requested time range are
    missing from the cache and fetches only those, paging through
    the upstream `MarketDataClient`.
  - Merges new candles into the cache, deduplicates on
    `open_time_ms`, sorts chronologically, and rewrites the file
    atomically (write to tmp, then rename).
  - Returns a pandas DataFrame indexed by UTC `DatetimeIndex` with
    `open`, `high`, `low`, `close`, `volume` columns. This is the
    exact shape `engine.backtest.Backtest.run` and the strategies
    consume.

CSV was chosen over Parquet to avoid a hard dependency on
pyarrow / fastparquet for backfill use cases. The cache stays
human-readable, diff-friendly, and easy to ship as a test fixture.
The downside is size for tick data, but for the OHLCV use case
backtest backfills stay under a few MB per (symbol, interval).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .protocols import INTERVAL_MS, Candle, MarketDataClient

logger = logging.getLogger(__name__)


CANONICAL_COLUMNS = ("open_time_ms", "open", "high", "low", "close", "volume")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataService:
    """Fetch-once-cache-forever OHLCV provider."""

    def __init__(
        self,
        client: MarketDataClient,
        cache_dir: Path,
    ) -> None:
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-(symbol, interval) lock so concurrent get_ohlcv calls
        # cannot read the same pre-merge snapshot and clobber each
        # other on save. atomic file replacement protects against
        # half-written files but not against two clean writers
        # racing to publish a stale-then-new version.
        self._cache_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Return a UTC-indexed OHLCV DataFrame for the requested range.

        Reads from cache, fetches any missing segments, merges, and
        returns the slice [start_ms, end_ms).

        Concurrent calls for the SAME (symbol, interval) pair are
        serialised through a per-key asyncio.Lock so two callers
        cannot load the same pre-merge snapshot and overwrite each
        other on save. Calls for DIFFERENT pairs run fully
        concurrently.
        """
        if end_ms <= start_ms:
            return self._empty_frame()

        interval_ms = INTERVAL_MS.get(interval)
        if interval_ms is None:
            # Fail loudly here, matching what BinancePublicClient
            # does on its own. Silently passing None to
            # _missing_ranges would let an unknown interval slip
            # through when the cache happened to be complete.
            raise ValueError(
                f"unknown interval {interval!r}. Known: {sorted(INTERVAL_MS)}"
            )

        # Lock key must be derived the same way as the cache path
        # so two callers for the same (symbol, interval) serialise
        # through the same lock - and two callers for DIFFERENT
        # symbols that happen to share characters do NOT collide.
        key = (str(self._cache_path(symbol, interval)), interval)
        lock = self._cache_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._load_cache(symbol, interval)
            missing = self._missing_ranges(cached, start_ms, end_ms, interval_ms)
            if missing:
                new_rows: List[Candle] = []
                for seg_start, seg_end in missing:
                    logger.info(
                        "MarketDataService fetching %s %s [%d, %d)",
                        symbol, interval, seg_start, seg_end,
                    )
                    page = await self.client.fetch_klines(
                        symbol, interval, seg_start, seg_end
                    )
                    new_rows.extend(page)
                if new_rows:
                    cached = self._merge(cached, new_rows)
                    self._save_cache(symbol, interval, cached)

            window = cached[
                (cached["open_time_ms"] >= start_ms)
                & (cached["open_time_ms"] < end_ms)
            ].copy()
            return self._to_indexed_frame(window)

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, interval: str) -> Path:
        # Preserve symbol structure so distinct markets cannot
        # collide on the cache filename. The previous version
        # mapped both `AB/CD` and `A/BCD` to `ABCD_1h.csv`,
        # silently serving candles for the wrong instrument once
        # either one warmed the cache.
        #
        # Replacement rules:
        #   '/'  -> '__'   (visual separator that cannot appear in
        #                   the original alphanumeric venue symbol)
        #   ':'  -> '_'    (kept consistent with the prior layout)
        # Anything else outside [A-Z0-9_.-] is hex-escaped so
        # exotic suffixes survive a round-trip without colliding.
        upper = symbol.upper()
        out_chars: List[str] = []
        for ch in upper:
            if ch == "/":
                out_chars.append("__")
            elif ch == ":":
                out_chars.append("_")
            elif ch.isalnum() or ch in {"_", ".", "-"}:
                out_chars.append(ch)
            else:
                out_chars.append(f"%{ord(ch):02X}")
        safe = "".join(out_chars)
        return self.cache_dir / f"{safe}_{interval}.csv"

    def _load_cache(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self._cache_path(symbol, interval)
        if not path.exists():
            return pd.DataFrame(columns=list(CANONICAL_COLUMNS)).astype(
                {"open_time_ms": "int64"}
            )
        try:
            # Type hints on read tell pandas how to allocate, avoid
            # re-scanning the data, and (importantly here) coerce
            # numeric columns at parse time so a manually-edited CSV
            # with a stray non-numeric cell raises a clean error
            # rather than crashing later on .astype(float).
            df = pd.read_csv(
                path,
                dtype={
                    "open_time_ms": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                },
            )
        except Exception as exc:
            # Two cases hit this path: a corrupted file, or a manual
            # edit with non-numeric cells the dtype= hint rejected.
            # Either way we log loudly and fall back to an empty
            # cache rather than crash; the next fetch will repopulate.
            logger.warning(
                "MarketDataService failed to read cache %s: %s. "
                "Falling back to empty cache; backfill will repopulate.",
                path, exc,
            )
            return pd.DataFrame(columns=list(CANONICAL_COLUMNS)).astype(
                {"open_time_ms": "int64"}
            )
        # Hardening: keep only the columns we control. Use
        # pd.to_numeric(errors="coerce") to turn any residual junk
        # into NaN before dropping, instead of letting .astype crash.
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[list(CANONICAL_COLUMNS)]
        df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_rows = df.isna().any(axis=1)
        if bad_rows.any():
            logger.warning(
                "MarketDataService dropping %d malformed row(s) from %s",
                int(bad_rows.sum()), path,
            )
        df = df.dropna(subset=list(CANONICAL_COLUMNS))
        df["open_time_ms"] = df["open_time_ms"].astype("int64")
        for col in OHLCV_COLUMNS:
            df[col] = df[col].astype(float)
        df = df.drop_duplicates(subset=["open_time_ms"]).sort_values(
            "open_time_ms"
        ).reset_index(drop=True)
        return df

    def _save_cache(
        self, symbol: str, interval: str, df: pd.DataFrame
    ) -> None:
        path = self._cache_path(symbol, interval)
        # Atomic write: tempfile in the same directory, then os.replace.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(self.cache_dir)
        )
        os.close(tmp_fd)
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Merge / gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(cached: pd.DataFrame, new_rows: List[Candle]) -> pd.DataFrame:
        if not new_rows:
            return cached
        new_df = pd.DataFrame(
            [
                {
                    "open_time_ms": c.open_time_ms,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in new_rows
            ]
        )
        merged = pd.concat([cached, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["open_time_ms"], keep="last")
        merged = merged.sort_values("open_time_ms").reset_index(drop=True)
        merged["open_time_ms"] = merged["open_time_ms"].astype("int64")
        return merged

    @staticmethod
    def _missing_ranges(
        cached: pd.DataFrame,
        start_ms: int,
        end_ms: int,
        interval_ms: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Return [start, end) segments inside [start_ms, end_ms) that
        are not yet present in the cache.

        Detects three kinds of gap:

        1. Head gap: [start_ms, cached_min) when the request
           starts before the cache.
        2. Internal gaps: any bar boundary inside
           [cached_min, cached_max] where two consecutive cached
           rows are more than `interval_ms` apart. Internal holes
           appear naturally when _load_cache drops malformed rows
           (a manual edit, a partial write, a venue maintenance
           gap); without detecting them we would silently serve an
           incomplete series. Skipped when `interval_ms` is None
           since we cannot tell what counts as a hole.
        3. Tail gap: [next_candle_open, end_ms) when the request
           ends past the cache.
        """
        if cached.empty:
            return [(start_ms, end_ms)]
        cached_min = int(cached["open_time_ms"].iloc[0])
        cached_max = int(cached["open_time_ms"].iloc[-1])
        gaps: List[Tuple[int, int]] = []
        if start_ms < cached_min:
            gaps.append((start_ms, min(cached_min, end_ms)))

        if interval_ms is not None and len(cached) > 1:
            # Walk consecutive open-times. A diff bigger than
            # interval_ms means a hole; intersect each hole with
            # the requested window so we never fetch outside it.
            opens = cached["open_time_ms"].to_numpy()
            for i in range(1, len(opens)):
                prev = int(opens[i - 1])
                curr = int(opens[i])
                expected_next = prev + interval_ms
                if curr > expected_next:
                    hole_start = max(expected_next, start_ms)
                    hole_end = min(curr, end_ms)
                    if hole_end > hole_start:
                        gaps.append((hole_start, hole_end))

        next_candle_open = cached_max + (interval_ms or 1)
        if end_ms > next_candle_open:
            gaps.append((max(next_candle_open, start_ms), end_ms))
        return gaps

    @staticmethod
    def _to_indexed_frame(window: pd.DataFrame) -> pd.DataFrame:
        if window.empty:
            return MarketDataService._empty_frame()
        index = pd.to_datetime(window["open_time_ms"], unit="ms", utc=True)
        out = window[list(OHLCV_COLUMNS)].copy()
        out.index = index
        out.index.name = "open_time"
        return out

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        idx = pd.DatetimeIndex([], tz="UTC", name="open_time")
        return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=idx)
