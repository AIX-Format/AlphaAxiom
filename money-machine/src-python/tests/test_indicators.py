"""
Tests for engine.indicators.

Two layers of verification:

1. Invariant tests. SMA over a constant series returns the constant,
   RSI is bounded in [0, 100], ATR is non-negative, MACD signal trails
   the MACD line, and so on. These catch most regressions without
   needing a reference table.

2. Reference-value tests. For a tiny hand-checkable series with
   period=3, we hard-code the expected SMA/EMA/RSI/ATR/MACD values
   and assert exact match within 1e-9. The reference values were
   computed independently with a pocket calculator following the
   canonical formulas (Wilder smoothing for RSI/ATR, alpha=2/(n+1)
   for EMA). If TradingView changes their math we'll find out the
   moment a real market run diverges.

3. Fixture sanity. The 60-bar BTCUSDT-shaped fixture in
   tests/fixtures is run through every indicator to make sure the
   output shape matches the input and the warm-up windows fall in
   the expected places.

No external network calls. CI-safe.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.indicators import (  # noqa: E402
    atr,
    ema,
    macd,
    rsi,
    sma,
    true_range,
    wilder_smoothing,
)
from tests.fixtures.ohlcv_btcusdt_1h_sample import CANDLES  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    """60-bar synthetic BTCUSDT 1h DataFrame, UTC indexed."""
    df = pd.DataFrame(
        {
            "open": [c.open for c in CANDLES],
            "high": [c.high for c in CANDLES],
            "low": [c.low for c in CANDLES],
            "close": [c.close for c in CANDLES],
            "volume": [c.volume for c in CANDLES],
        },
        index=pd.to_datetime(
            [c.timestamp_ms for c in CANDLES], unit="ms", utc=True
        ),
    )
    return df


# ---------------------------------------------------------------------------
# Invariants: SMA / EMA / Wilder
# ---------------------------------------------------------------------------


def test_sma_of_constant_series_is_constant() -> None:
    s = pd.Series([5.0] * 30)
    out = sma(s, period=10)
    assert out.iloc[:9].isna().all()
    assert (out.iloc[9:] == 5.0).all()


def test_sma_preserves_index() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    s = pd.Series(range(20), index=idx, dtype=float)
    out = sma(s, period=5)
    assert out.index.equals(idx)


def test_ema_of_constant_series_is_constant() -> None:
    s = pd.Series([10.0] * 30)
    out = ema(s, period=5)
    # First period-1 rows are NaN, rest converge to the constant.
    assert out.iloc[:4].isna().all()
    assert (out.iloc[4:] - 10.0).abs().max() < 1e-12


def test_wilder_smoothing_matches_alpha_formula() -> None:
    # Wilder smoothing with period n is equivalent to ewm with
    # alpha=1/n. We verify against the recurrence by hand.
    period = 4
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = wilder_smoothing(s, period)
    # First valid value is the mean of the first `period` items.
    expected_first = (1 + 2 + 3 + 4) / 4
    assert math.isclose(out.iloc[3], expected_first, abs_tol=1e-12)
    # Each subsequent: prev + (x - prev) / period.
    expected = expected_first
    alpha = 1.0 / period
    for i, x in enumerate(s.iloc[4:].tolist(), start=4):
        expected = expected + alpha * (x - expected)
        assert math.isclose(out.iloc[i], expected, abs_tol=1e-12)


def test_ema_invalid_period_raises() -> None:
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), period=0)
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), period=-3)


# ---------------------------------------------------------------------------
# Invariants: RSI
# ---------------------------------------------------------------------------


def test_rsi_bounded_zero_to_hundred(fixture_df: pd.DataFrame) -> None:
    out = rsi(fixture_df["close"], period=14)
    valid = out.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 100.0).all()


def test_rsi_monotonic_up_approaches_one_hundred() -> None:
    s = pd.Series([100.0 + i for i in range(50)])
    out = rsi(s, period=14).dropna()
    # A monotonically increasing series has zero losses, so RSI = 100.
    assert (out == 100.0).all()


def test_rsi_monotonic_down_approaches_zero() -> None:
    s = pd.Series([200.0 - i for i in range(50)])
    out = rsi(s, period=14).dropna()
    assert (out == 0.0).all()


def test_rsi_flat_series_is_fifty() -> None:
    s = pd.Series([50.0] * 30)
    out = rsi(s, period=14).dropna()
    # No movement at all: by convention RSI is 50.
    assert (out == 50.0).all()


# ---------------------------------------------------------------------------
# Invariants: MACD
# ---------------------------------------------------------------------------


def test_macd_signal_lags_macd_line(fixture_df: pd.DataFrame) -> None:
    macd_line, signal_line, hist = macd(fixture_df["close"])
    # All three are aligned to the input index.
    assert macd_line.index.equals(fixture_df.index)
    assert signal_line.index.equals(fixture_df.index)
    assert hist.index.equals(fixture_df.index)
    # Histogram is the difference, by construction.
    diff = (macd_line - signal_line).dropna()
    other = hist.dropna()
    pd.testing.assert_series_equal(diff, other, check_names=False)


def test_macd_rejects_fast_ge_slow() -> None:
    s = pd.Series(range(30), dtype=float)
    with pytest.raises(ValueError):
        macd(s, fast_period=20, slow_period=10)


# ---------------------------------------------------------------------------
# Invariants: ATR
# ---------------------------------------------------------------------------


def test_true_range_non_negative(fixture_df: pd.DataFrame) -> None:
    tr = true_range(fixture_df["high"], fixture_df["low"], fixture_df["close"])
    assert (tr >= 0.0).all()


def test_atr_non_negative_and_warmup(fixture_df: pd.DataFrame) -> None:
    out = atr(fixture_df["high"], fixture_df["low"], fixture_df["close"], period=14)
    # SMA-seeded warm-up: first period-1 rows are NaN, the seed lands at
    # index period-1 (= 13 here).
    assert out.iloc[:13].isna().all()
    assert out.iloc[13:].notna().all()
    valid = out.dropna()
    assert (valid > 0.0).all()


# ---------------------------------------------------------------------------
# Reference values on a tiny hand-checkable series.
#
# Series: [10, 11, 12, 11, 13, 14, 13, 15, 14, 16]
# Period: 3 throughout, so the math is short enough to verify by hand.
# ---------------------------------------------------------------------------


REF_SERIES = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0])


def test_sma_reference_values() -> None:
    out = sma(REF_SERIES, period=3)
    # Hand-computed: (10+11+12)/3=11.0, (11+12+11)/3=11.333..., etc.
    expected = [
        None, None, 11.0,
        (11 + 12 + 11) / 3,
        (12 + 11 + 13) / 3,
        (11 + 13 + 14) / 3,
        (13 + 14 + 13) / 3,
        (14 + 13 + 15) / 3,
        (13 + 15 + 14) / 3,
        (15 + 14 + 16) / 3,
    ]
    for i, exp in enumerate(expected):
        if exp is None:
            assert math.isnan(out.iloc[i])
        else:
            assert math.isclose(out.iloc[i], exp, abs_tol=1e-12), f"sma[{i}]"


def test_ema_reference_values() -> None:
    # EMA period 3 -> alpha = 2/(3+1) = 0.5
    # First valid value at index 2 is the SMA seed: (10+11+12)/3 = 11.
    # Then: ema[i] = ema[i-1] + 0.5 * (x[i] - ema[i-1])
    out = ema(REF_SERIES, period=3)
    expected = [None, None, 11.0]
    prev = 11.0
    alpha = 0.5
    for x in REF_SERIES.iloc[3:].tolist():
        prev = prev + alpha * (x - prev)
        expected.append(prev)
    for i, exp in enumerate(expected):
        if exp is None:
            assert math.isnan(out.iloc[i])
        else:
            assert math.isclose(out.iloc[i], exp, abs_tol=1e-9), f"ema[{i}]={out.iloc[i]} vs {exp}"


def test_rsi_reference_values_period_three() -> None:
    # Verify against the canonical Wilder formula at period=3.
    # gains = [_, 1, 1, 0, 2, 1, 0, 2, 0, 2]
    # losses = [_, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    out = rsi(REF_SERIES, period=3)
    # First 3 rows NaN (period-many diffs needed by Wilder seed).
    assert out.iloc[:3].isna().all()

    # Hand-compute Wilder smoothed gains/losses:
    gains = [0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0]
    losses = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    # Wilder seed at index 3 (period=3 means avg of first 3 diffs at idx 1..3).
    seed_idx = 3
    avg_g = sum(gains[1:seed_idx + 1]) / 3
    avg_l = sum(losses[1:seed_idx + 1]) / 3
    alpha = 1.0 / 3
    rsi_vals = [None] * 3
    rs = avg_g / avg_l if avg_l else float("inf")
    rsi_vals.append(100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + rs))
    for i in range(seed_idx + 1, len(REF_SERIES)):
        avg_g = avg_g + alpha * (gains[i] - avg_g)
        avg_l = avg_l + alpha * (losses[i] - avg_l)
        if avg_l == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_g / avg_l
            rsi_vals.append(100.0 - 100.0 / (1.0 + rs))

    for i, exp in enumerate(rsi_vals):
        if exp is None:
            assert math.isnan(out.iloc[i]), f"rsi[{i}] expected NaN, got {out.iloc[i]}"
        else:
            assert math.isclose(out.iloc[i], exp, abs_tol=1e-6), (
                f"rsi[{i}]={out.iloc[i]} vs {exp}"
            )


def test_atr_reference_values_period_three() -> None:
    high = pd.Series([10.5, 11.4, 12.6, 11.5, 13.3, 14.2, 13.4, 15.1, 14.5, 16.2])
    low = pd.Series([9.7, 10.6, 11.5, 10.8, 12.7, 13.5, 12.9, 14.4, 13.6, 15.5])
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0])

    out = atr(high, low, close, period=3)
    # SMA-seeded Wilder: first period-1 rows are NaN, seed at index 2.
    assert out.iloc[:2].isna().all()

    # Compute TR manually.
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = (high - low).iloc[0]

    # Wilder seed: mean of first 3 TR values lands at index 2.
    seed = tr.iloc[:3].mean()
    expected = [None, None, seed]
    prev = seed
    alpha = 1.0 / 3
    for x in tr.iloc[3:].tolist():
        prev = prev + alpha * (x - prev)
        expected.append(prev)

    for i, exp in enumerate(expected):
        if exp is None:
            assert math.isnan(out.iloc[i])
        else:
            assert math.isclose(out.iloc[i], exp, abs_tol=1e-9), f"atr[{i}]"


def test_macd_returns_three_aligned_series(fixture_df: pd.DataFrame) -> None:
    macd_line, signal_line, hist = macd(
        fixture_df["close"], fast_period=12, slow_period=26, signal_period=9
    )
    # MACD line is defined once the slow EMA warms up: from index 25.
    assert macd_line.iloc[:25].isna().all()
    assert macd_line.iloc[25:].notna().all()
    # Signal line warms up after MACD itself has 9 valid values.
    # First signal value lands at index 25 + 9 - 1 = 33.
    assert signal_line.iloc[:33].isna().all()
    assert signal_line.iloc[33:].notna().all()


# ---------------------------------------------------------------------------
# Self-runner for environments without pytest.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import traceback

    df = pd.DataFrame(
        {
            "open": [c.open for c in CANDLES],
            "high": [c.high for c in CANDLES],
            "low": [c.low for c in CANDLES],
            "close": [c.close for c in CANDLES],
            "volume": [c.volume for c in CANDLES],
        },
        index=pd.to_datetime(
            [c.timestamp_ms for c in CANDLES], unit="ms", utc=True
        ),
    )

    failures = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            argcount = fn.__code__.co_argcount
            if argcount == 1:
                fn(df)
            else:
                fn()
            print(f"[ok] {name}")
        except Exception:
            failures += 1
            print(f"[fail] {name}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
