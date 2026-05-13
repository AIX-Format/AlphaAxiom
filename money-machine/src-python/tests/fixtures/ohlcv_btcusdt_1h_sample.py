"""
Deterministic OHLCV fixture used by the indicator tests.

The series is 60 hourly BTCUSDT candles starting 2024-01-01T00:00Z,
hand-curated to exercise both up-trending and down-trending regimes
so RSI, MACD, and ATR all have non-trivial values. Prices are stored
as plain floats; the test helper materialises them into a DataFrame
with a UTC DatetimeIndex.

Keep the fixture committed verbatim. If you need to refresh it from
real exchange data, run `scripts/refresh_indicator_fixture.py` (added
in a follow-up PR) and replace this file. Do not edit individual rows
in isolation: the expected-value tables in
`tests/test_indicators.py` are computed from this exact series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# 60 candles, 1h spacing, deterministic synthetic prices that walk up
# then pull back then break out again. Numbers are within plausible
# BTCUSDT range but are NOT real market data.
_HOUR_MS = 3_600_000
_START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z

_CLOSES: List[float] = [
    42000.0, 42150.0, 42080.0, 42220.0, 42310.0,
    42400.0, 42380.0, 42550.0, 42600.0, 42750.0,
    42820.0, 42700.0, 42650.0, 42580.0, 42470.0,
    42350.0, 42200.0, 42050.0, 41980.0, 41850.0,
    41700.0, 41600.0, 41750.0, 41900.0, 42050.0,
    42180.0, 42300.0, 42450.0, 42600.0, 42780.0,
    42900.0, 43050.0, 43200.0, 43180.0, 43100.0,
    43000.0, 42950.0, 42880.0, 42820.0, 42900.0,
    43000.0, 43150.0, 43300.0, 43500.0, 43700.0,
    43900.0, 44050.0, 44200.0, 44150.0, 44080.0,
    44000.0, 43950.0, 44100.0, 44250.0, 44400.0,
    44600.0, 44800.0, 44950.0, 45100.0, 45300.0,
]


def _build_candles() -> List[Candle]:
    candles: List[Candle] = []
    prev_close = _CLOSES[0]
    for i, close in enumerate(_CLOSES):
        # Deterministic intraday range: 0.4% above and below the
        # midpoint of the open->close move. This is enough variance
        # for the ATR test to be meaningful.
        op = prev_close
        mid = (op + close) / 2.0
        spread = mid * 0.004
        hi = max(op, close) + spread
        lo = min(op, close) - spread
        volume = 100.0 + (i % 7) * 25.0
        candles.append(
            Candle(
                timestamp_ms=_START_MS + i * _HOUR_MS,
                open=op,
                high=hi,
                low=lo,
                close=close,
                volume=volume,
            )
        )
        prev_close = close
    return candles


CANDLES: List[Candle] = _build_candles()
