"""
Extended performance metrics beyond what `engine.backtest.BacktestMetrics`
already reports.

The engine's built-in `BacktestMetrics` covers total return, Sharpe,
max drawdown, win rate, and trade counts. Researchers comparing
strategies typically want a few more measures: Sortino (downside-only
risk), Calmar (return-to-drawdown), profit factor (gross wins over
gross losses), expectancy, and exposure share.

This module is pure: it takes the same objects `Backtest.run()`
already produces and computes additional numbers. It does NOT modify
the existing backtest contract, and the engine code is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.backtest import TradeRecord, _infer_periods_per_year


@dataclass
class ExtendedMetrics:
    """Researcher-facing metrics computed from a `BacktestResult`."""

    sortino: float
    calmar: float
    profit_factor: float
    expectancy: float
    exposure_pct: float
    avg_trade_duration_bars: float
    best_trade_pnl: float
    worst_trade_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sortino": self.sortino,
            "calmar": self.calmar,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "exposure_pct": self.exposure_pct,
            "avg_trade_duration_bars": self.avg_trade_duration_bars,
            "best_trade_pnl": self.best_trade_pnl,
            "worst_trade_pnl": self.worst_trade_pnl,
        }


def compute_extended_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    *,
    initial_equity: float,
    periods_per_year: Optional[float] = None,
) -> ExtendedMetrics:
    """Compute the metrics in `ExtendedMetrics` from a finished backtest."""
    if equity_curve.empty:
        return _empty()

    returns = equity_curve.pct_change().dropna()
    ann = periods_per_year or _infer_periods_per_year(equity_curve.index)

    # Sortino: like Sharpe but the denominator is the std of NEGATIVE
    # returns only. Zero downside means we cannot annualise; report 0
    # rather than divide by zero or invent a positive number.
    downside = returns[returns < 0]
    if len(returns) > 1 and not downside.empty and downside.std(ddof=0) > 0:
        sortino = float(returns.mean() / downside.std(ddof=0) * math.sqrt(ann))
    else:
        sortino = 0.0

    # Calmar: total return divided by max drawdown over the same window.
    # Returns 0 if max drawdown is zero (e.g. monotonic equity).
    running_peak = equity_curve.cummax()
    drawdowns = (equity_curve - running_peak) / running_peak.where(
        running_peak > 0, np.nan
    )
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
    if math.isnan(max_dd):
        max_dd = 0.0
    max_dd_abs = abs(min(max_dd, 0.0))
    final = float(equity_curve.iloc[-1])
    total_return = (final / initial_equity) - 1.0 if initial_equity > 0 else 0.0
    calmar = (total_return / max_dd_abs) if max_dd_abs > 0 else 0.0

    # Profit factor: sum of winning pnl over abs(sum of losing pnl).
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = sum(-t.pnl for t in trades if t.pnl < 0)
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        # All winners is technically infinite PF. Report a large
        # sentinel rather than inf so JSON serialisation works and
        # downstream comparisons stay numeric.
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Expectancy: average $ won/lost per trade.
    if trades:
        expectancy = float(sum(t.pnl for t in trades) / len(trades))
    else:
        expectancy = 0.0

    # Exposure: share of bars in which a position was open. Computed
    # from trade timestamps rather than tracked tick-by-tick, which is
    # cheap and accurate at bar resolution.
    exposure_pct = _exposure_pct(equity_curve.index, trades)

    # Trade duration in bars.
    durations = _trade_durations_bars(equity_curve.index, trades)
    avg_duration = float(sum(durations) / len(durations)) if durations else 0.0

    pnls = [t.pnl for t in trades]
    best = float(max(pnls)) if pnls else 0.0
    worst = float(min(pnls)) if pnls else 0.0

    return ExtendedMetrics(
        sortino=sortino,
        calmar=calmar,
        profit_factor=profit_factor,
        expectancy=expectancy,
        exposure_pct=exposure_pct,
        avg_trade_duration_bars=avg_duration,
        best_trade_pnl=best,
        worst_trade_pnl=worst,
    )


def _empty() -> ExtendedMetrics:
    return ExtendedMetrics(
        sortino=0.0,
        calmar=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        exposure_pct=0.0,
        avg_trade_duration_bars=0.0,
        best_trade_pnl=0.0,
        worst_trade_pnl=0.0,
    )


def _exposure_pct(index: pd.Index, trades: List[TradeRecord]) -> float:
    """Share of bars where a position was open, on the [0, 1] scale.

    Counts a bar as "exposed" if its timestamp falls in the closed
    interval [entry_time, exit_time] of any trade. Trades whose
    timestamps fall outside the equity-curve index (which should not
    happen but the engine sets the final force-close index to the
    last bar) are clamped to the index range.
    """
    if len(index) == 0 or not trades:
        return 0.0
    total_bars = len(index)
    if total_bars <= 0:
        return 0.0
    exposed = np.zeros(total_bars, dtype=bool)
    for t in trades:
        # `searchsorted` gives us the first index >= the timestamp,
        # which matches "the first bar at or after the entry".
        start = int(index.searchsorted(t.entry_time, side="left"))
        end = int(index.searchsorted(t.exit_time, side="right"))
        # Defensive clamp: a trade closed exactly on the last bar
        # can produce end == total_bars which would silently no-op
        # in numpy slicing but we want it counted.
        start = max(0, min(start, total_bars))
        end = max(start, min(end, total_bars))
        if end > start:
            exposed[start:end] = True
    return float(exposed.sum()) / total_bars


def _trade_durations_bars(
    index: pd.Index, trades: List[TradeRecord]
) -> List[int]:
    if len(index) == 0 or not trades:
        return []
    out: List[int] = []
    for t in trades:
        start = int(index.searchsorted(t.entry_time, side="left"))
        end = int(index.searchsorted(t.exit_time, side="right"))
        out.append(max(0, end - start))
    return out
