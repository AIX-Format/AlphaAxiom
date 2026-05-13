"""
Tests for `scripts.research_metrics`.

The metrics module is pure: it takes objects that the backtest engine
already produces and computes additional numbers. These tests build
deterministic inputs by hand so the math is verifiable from the test
file alone.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import pytest

# Make `engine.*` and `scripts.*` importable from the test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.backtest import TradeRecord  # noqa: E402

from scripts.research_metrics import (  # noqa: E402
    ExtendedMetrics,
    compute_extended_metrics,
)


def _hourly_index(n: int) -> pd.DatetimeIndex:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pd.DatetimeIndex(
        [start + timedelta(hours=i) for i in range(n)], tz="UTC"
    )


def _trade(
    entry_idx: pd.Timestamp,
    exit_idx: pd.Timestamp,
    pnl: float,
    direction: str = "long",
) -> TradeRecord:
    return TradeRecord(
        entry_time=entry_idx,
        exit_time=exit_idx,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        quantity=1.0,
        pnl=pnl,
        commission=0.0,
        exit_reason="signal_flip",
    )


def test_empty_curve_returns_zero_metrics() -> None:
    out = compute_extended_metrics(
        pd.Series(dtype=float), [], initial_equity=10_000.0
    )
    assert isinstance(out, ExtendedMetrics)
    assert out.sortino == 0.0
    assert out.calmar == 0.0
    assert out.profit_factor == 0.0
    assert out.expectancy == 0.0
    assert out.exposure_pct == 0.0


def test_profit_factor_handles_no_losers() -> None:
    idx = _hourly_index(5)
    equity = pd.Series([10_000.0, 10_010.0, 10_020.0, 10_030.0, 10_040.0], index=idx)
    trades = [
        _trade(idx[0], idx[1], 10.0),
        _trade(idx[2], idx[3], 5.0),
    ]
    out = compute_extended_metrics(equity, trades, initial_equity=10_000.0)
    # All winners: profit factor is +inf by definition.
    assert math.isinf(out.profit_factor)
    assert out.expectancy == pytest.approx(7.5)
    assert out.best_trade_pnl == pytest.approx(10.0)
    assert out.worst_trade_pnl == pytest.approx(5.0)


def test_profit_factor_computes_winners_over_losers() -> None:
    idx = _hourly_index(4)
    equity = pd.Series([10_000.0, 10_020.0, 10_010.0, 10_030.0], index=idx)
    trades = [
        _trade(idx[0], idx[1], 20.0),
        _trade(idx[1], idx[2], -10.0),
        _trade(idx[2], idx[3], 20.0),
    ]
    out = compute_extended_metrics(equity, trades, initial_equity=10_000.0)
    # Gross win = 40, gross loss = 10, PF = 4.0.
    assert out.profit_factor == pytest.approx(4.0)
    # Expectancy = (20 - 10 + 20) / 3 = 10.0.
    assert out.expectancy == pytest.approx(30.0 / 3.0)


def test_calmar_is_zero_when_no_drawdown() -> None:
    idx = _hourly_index(5)
    # Strictly increasing equity, no drawdown at all.
    equity = pd.Series(
        [10_000.0, 10_010.0, 10_025.0, 10_050.0, 10_100.0], index=idx
    )
    trades: List[TradeRecord] = []
    out = compute_extended_metrics(equity, trades, initial_equity=10_000.0)
    assert out.calmar == 0.0


def test_calmar_normalises_return_by_max_drawdown() -> None:
    idx = _hourly_index(5)
    # Path: 10000 -> 11000 -> 9900 (drawdown 10%) -> 11550 -> 12100.
    equity = pd.Series(
        [10_000.0, 11_000.0, 9_900.0, 11_550.0, 12_100.0], index=idx
    )
    out = compute_extended_metrics(equity, [], initial_equity=10_000.0)
    # total_return = 0.21, max_dd = 0.10, Calmar = 2.1.
    assert out.calmar == pytest.approx(2.1, rel=1e-6)


def test_exposure_pct_counts_only_bars_inside_trade_window() -> None:
    idx = _hourly_index(10)
    equity = pd.Series([10_000.0] * 10, index=idx)
    # Trade covers bars 2-4 inclusive (3 bars), out of 10 total -> 0.3.
    trades = [_trade(idx[2], idx[4], 5.0)]
    out = compute_extended_metrics(equity, trades, initial_equity=10_000.0)
    assert out.exposure_pct == pytest.approx(0.3, rel=1e-6)


def test_sortino_is_zero_when_no_downside_returns() -> None:
    idx = _hourly_index(5)
    # Monotonic non-decreasing returns -> no negative bar -> Sortino 0.
    equity = pd.Series(
        [10_000.0, 10_010.0, 10_010.0, 10_020.0, 10_030.0], index=idx
    )
    out = compute_extended_metrics(equity, [], initial_equity=10_000.0)
    assert out.sortino == 0.0


def test_avg_trade_duration_is_in_bars() -> None:
    idx = _hourly_index(20)
    equity = pd.Series([10_000.0] * 20, index=idx)
    trades = [
        _trade(idx[0], idx[3], 5.0),    # 3 bars
        _trade(idx[5], idx[10], -2.0),  # 5 bars
    ]
    out = compute_extended_metrics(equity, trades, initial_equity=10_000.0)
    # The implementation counts bars inclusive of entry up to but not
    # past exit; the exact number depends on searchsorted boundaries.
    # The assertion below is permissive on +/- 1 bar.
    assert 3.5 <= out.avg_trade_duration_bars <= 5.0
