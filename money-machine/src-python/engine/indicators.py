"""
Vectorized technical indicators.

Every public function takes a pandas Series (or DataFrame columns for
OHLC) and returns a Series aligned on the input index. The first N
values are NaN where N matches the warm-up window TradingView, TA-Lib,
and pandas-ta all use, so reference comparisons line up.

Smoothing conventions:

- `sma(period)`: standard rolling mean, first valid value at index
  `period - 1`.
- `ema(period)`: alpha = `2 / (period + 1)`. Seeded with the SMA of
  the first `period` observations at index `period - 1`, then
  recurses. This is what TradingView's `ta.ema()` produces and what
  most strategy literature assumes.
- `wilder_smoothing(period)`: alpha = `1 / period`. Same SMA-seeded
  start, used as the building block for RSI and ATR. Matches
  TradingView's `ta.rma()`.

The functions are pure and stateless: they do not mutate the input,
they do not log, and they do not raise on short inputs (a series
shorter than the warm-up window simply returns all-NaN, which the
strategy layer is expected to handle).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    Returns NaN for the first `period - 1` rows so the output index
    matches the input.
    """
    _validate_period(period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, SMA-seeded.

    Matches TradingView's `ta.ema()`. The first `period - 1` rows are
    NaN. The value at index `period - 1` is the simple mean of the
    first `period` observations; subsequent values follow the
    recurrence `ema[i] = ema[i-1] + alpha * (x[i] - ema[i-1])` with
    `alpha = 2 / (period + 1)`.
    """
    _validate_period(period)
    alpha = 2.0 / (period + 1.0)
    return _sma_seeded_recursive(series, period, alpha)


def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA), SMA-seeded.

    Same shape as `ema` but with `alpha = 1 / period`. This is the
    smoothing TradingView's `ta.rma()` uses and the canonical building
    block for RSI and ATR per Welles Wilder's original definitions.
    """
    _validate_period(period)
    alpha = 1.0 / period
    return _sma_seeded_recursive(series, period, alpha)


def _sma_seeded_recursive(
    series: pd.Series, period: int, alpha: float
) -> pd.Series:
    """Shared engine for SMA-seeded EMA-style smoothing.

    The recursive step is computed in a Python loop. A typical chart
    pulls a few thousand candles at most, and the loop runs in low
    milliseconds at that size. We avoid the bias of pandas'
    `ewm(adjust=False)` which starts the recursion at the first sample
    rather than the SMA, which is why our outputs match TradingView
    while raw `ewm(min_periods=N)` does not.
    """
    values = series.to_numpy(dtype=float, copy=True)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)

    if n < period:
        return pd.Series(out, index=series.index, name=series.name)

    # SMA seed at index period - 1, ignoring leading NaN inputs.
    window = values[:period]
    if np.isnan(window).any():
        # If the warm-up window contains NaN there is no honest seed.
        # Mark everything NaN; the strategy layer should drop leading
        # NaN before passing the series in.
        return pd.Series(out, index=series.index, name=series.name)

    seed = float(window.mean())
    out[period - 1] = seed

    prev = seed
    for i in range(period, n):
        x = values[i]
        if np.isnan(x):
            out[i] = prev  # carry forward across rare gaps
            continue
        prev = prev + alpha * (x - prev)
        out[i] = prev

    return pd.Series(out, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Output is bounded to [0, 100]. The first `period` rows are NaN
    (the first diff is NaN, then `period - 1` more rows are consumed
    by the Wilder warm-up on the diff series).

    Conventions:
    - All gains and no losses over the window: RSI = 100.
    - All losses and no gains: RSI = 0.
    - No movement at all (both averages zero): RSI = 50.
    """
    _validate_period(period)
    delta = close.diff()
    gains = delta.clip(lower=0.0).dropna()
    losses = (-delta).clip(lower=0.0).dropna()

    avg_gain = wilder_smoothing(gains, period).reindex(close.index)
    avg_loss = wilder_smoothing(losses, period).reindex(close.index)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 and avg_gain > 0 -> 100. Both zero -> 50.
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    rsi_values = rsi_values.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    # Preserve the warm-up NaN where the underlying averages are NaN.
    rsi_values = rsi_values.where(avg_gain.notna() & avg_loss.notna(), np.nan)
    return rsi_values.astype(float)


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Returns `(macd_line, signal_line, histogram)` all aligned to the
    input index.

    - `macd_line = ema(close, fast) - ema(close, slow)`
    - `signal_line = ema(macd_line, signal)` computed on the
      NaN-stripped MACD series, then reindexed back so the warm-up
      lands at the right bar.
    - `histogram = macd_line - signal_line`
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        )
    fast = ema(close, fast_period)
    slow = ema(close, slow_period)
    macd_line = fast - slow

    macd_clean = macd_line.dropna()
    signal_clean = ema(macd_clean, signal_period)
    signal_line = signal_clean.reindex(close.index)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range.

    `TR = max(high - low, |high - prev_close|, |low - prev_close|)`.
    The first row falls back to `high - low` because there is no
    previous close.
    """
    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    tr.iloc[0] = hl.iloc[0]
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    Returns NaN for the first `period - 1` rows; the value at index
    `period - 1` is the SMA seed over the first `period` True Range
    observations.
    """
    _validate_period(period)
    tr = true_range(high, low, close)
    return wilder_smoothing(tr, period)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_period(period: int) -> None:
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"period must be a positive int, got {period!r}")
