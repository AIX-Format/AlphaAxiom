"""
Position sizing helpers.

Three deterministic, stateless functions:

- `fixed_fractional(equity, risk_pct, stop_loss_pct)` returns the
  notional dollar size that risks exactly `risk_pct` of equity given
  a stop-loss `stop_loss_pct` away from entry.
- `atr_based(equity, atr, entry_price, risk_pct, atr_multiplier)`
  derives the same risk-budgeted size but using ATR as the stop
  distance, which adapts position size to current volatility.
- `kelly_criterion(win_rate, avg_win, avg_loss, fraction)` returns a
  capped fraction of equity to deploy on a strategy with the given
  historical edge. A fractional Kelly multiplier (default 0.5) keeps
  the suggestion well below full Kelly, which is known to be highly
  variance-prone in practice.

All three never return negative or NaN sizes. Bad inputs (zero stop
distance, negative equity, losing edge) return 0.0 so the caller can
filter on truthiness without special-casing.
"""

from __future__ import annotations

import math


def fixed_fractional(
    equity: float,
    risk_pct: float,
    stop_loss_pct: float,
) -> float:
    """Risk-of-ruin-aware fixed fractional sizing.

    Returns the notional position size such that hitting `stop_loss_pct`
    away from entry loses exactly `risk_pct` of equity.

    notional = equity * risk_pct / stop_loss_pct
    """
    if not _is_positive(equity) or not _is_positive(risk_pct):
        return 0.0
    if not _is_positive(stop_loss_pct):
        return 0.0
    return equity * risk_pct / stop_loss_pct


def atr_based(
    equity: float,
    atr: float,
    entry_price: float,
    risk_pct: float,
    atr_multiplier: float = 2.0,
) -> float:
    """ATR-based position size.

    The stop is `atr_multiplier * atr` away from the entry price.
    Returns the notional size that risks `risk_pct` of equity over
    that stop distance.

    notional = equity * risk_pct * entry_price / (atr_multiplier * atr)
    """
    if not _is_positive(equity) or not _is_positive(risk_pct):
        return 0.0
    if not _is_positive(atr) or not _is_positive(entry_price):
        return 0.0
    if not _is_positive(atr_multiplier):
        return 0.0
    stop_distance = atr_multiplier * atr
    if stop_distance <= 0:
        return 0.0
    return equity * risk_pct * entry_price / stop_distance


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """Fractional Kelly bet sizing.

    Returns the fraction of equity to allocate to the next bet given
    a historical edge. Inputs:

    - `win_rate`: empirical P(win), in [0, 1].
    - `avg_win`: average gain on winners, in the same units as
      `avg_loss`. Must be positive.
    - `avg_loss`: average loss on losers, expressed as a positive
      number (the magnitude). Must be positive.
    - `fraction`: scales the raw Kelly fraction. The default 0.5 is
      "half Kelly", which keeps variance manageable in practice.

    Full Kelly: f* = win_rate - (1 - win_rate) / (avg_win / avg_loss).
    Capped at [0.0, 1.0]. A negative or zero edge returns 0.0.
    """
    if not 0.0 <= win_rate <= 1.0:
        return 0.0
    if not _is_positive(avg_win) or not _is_positive(avg_loss):
        return 0.0
    if not _is_positive(fraction):
        return 0.0

    payoff_ratio = avg_win / avg_loss
    full_kelly = win_rate - (1.0 - win_rate) / payoff_ratio
    if full_kelly <= 0.0:
        return 0.0
    capped = min(max(fraction * full_kelly, 0.0), 1.0)
    return capped


def _is_positive(x: float) -> bool:
    """True if x is a finite positive number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0
