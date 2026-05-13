"""
Tests for engine.position_sizing.

Three sizing strategies with clean mathematical contracts. The tests
verify:

  - the closed-form math holds exactly,
  - every bad input collapses to 0.0 (caller can use truthiness),
  - Kelly stays in [0, 1] and respects the fractional cap,
  - the size produced by `fixed_fractional` is exactly what the risk
    shield's `max_position_size_pct` would allow when configured
    with the same risk fraction (this is the cross-module acceptance
    criterion from the sprint plan).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.position_sizing import (  # noqa: E402
    atr_based,
    fixed_fractional,
    kelly_criterion,
)
from engine.risk_shield import RiskConfig  # noqa: E402


# ---------------------------------------------------------------------------
# fixed_fractional
# ---------------------------------------------------------------------------


def test_fixed_fractional_basic_math() -> None:
    # 10k equity, 1% risk, 2% stop -> 10000 * 0.01 / 0.02 = 5000.
    assert fixed_fractional(10_000.0, 0.01, 0.02) == pytest.approx(5_000.0)


def test_fixed_fractional_scales_linearly_with_equity() -> None:
    a = fixed_fractional(10_000.0, 0.01, 0.02)
    b = fixed_fractional(20_000.0, 0.01, 0.02)
    assert b == pytest.approx(2 * a)


def test_fixed_fractional_zero_stop_returns_zero() -> None:
    # Stop of 0 would imply infinite size; we collapse to 0 instead.
    assert fixed_fractional(10_000.0, 0.01, 0.0) == 0.0


def test_fixed_fractional_bad_inputs_return_zero() -> None:
    assert fixed_fractional(-10.0, 0.01, 0.02) == 0.0
    assert fixed_fractional(10_000.0, -0.01, 0.02) == 0.0
    assert fixed_fractional(float("nan"), 0.01, 0.02) == 0.0
    assert fixed_fractional(float("inf"), 0.01, 0.02) == 0.0


# ---------------------------------------------------------------------------
# atr_based
# ---------------------------------------------------------------------------


def test_atr_based_matches_closed_form() -> None:
    # equity=10k, ATR=100, entry=50000, risk=1%, mult=2
    # stop_distance = 200, size = 10000 * 0.01 * 50000 / 200 = 25000
    assert atr_based(10_000.0, 100.0, 50_000.0, 0.01, atr_multiplier=2.0) == pytest.approx(
        25_000.0
    )


def test_atr_based_size_falls_when_volatility_rises() -> None:
    low_vol = atr_based(10_000.0, 50.0, 50_000.0, 0.01, atr_multiplier=2.0)
    high_vol = atr_based(10_000.0, 500.0, 50_000.0, 0.01, atr_multiplier=2.0)
    assert high_vol < low_vol


def test_atr_based_bad_inputs_return_zero() -> None:
    assert atr_based(10_000.0, 0.0, 50_000.0, 0.01) == 0.0
    assert atr_based(10_000.0, 100.0, 0.0, 0.01) == 0.0
    assert atr_based(10_000.0, 100.0, 50_000.0, 0.0) == 0.0
    assert atr_based(10_000.0, 100.0, 50_000.0, 0.01, atr_multiplier=0.0) == 0.0


# ---------------------------------------------------------------------------
# kelly_criterion
# ---------------------------------------------------------------------------


def test_kelly_full_for_positive_edge() -> None:
    # win_rate=0.6, avg_win=2, avg_loss=1 -> full Kelly = 0.6 - 0.4/2 = 0.4
    # Half Kelly (default fraction=0.5) -> 0.2
    assert kelly_criterion(0.6, 2.0, 1.0) == pytest.approx(0.2)


def test_kelly_zero_for_no_edge() -> None:
    # win_rate=0.5, payoff=1:1 -> Kelly = 0
    assert kelly_criterion(0.5, 1.0, 1.0) == 0.0


def test_kelly_zero_for_losing_edge() -> None:
    # Losing strategy: negative full Kelly clamps to 0.
    assert kelly_criterion(0.4, 1.0, 1.0) == 0.0


def test_kelly_caps_at_one() -> None:
    # Extreme inputs that would give f* > 1.
    out = kelly_criterion(0.95, 10.0, 1.0, fraction=1.0)
    assert 0.0 < out <= 1.0


def test_kelly_fractional_multiplier_scales_output() -> None:
    full = kelly_criterion(0.6, 2.0, 1.0, fraction=1.0)
    half = kelly_criterion(0.6, 2.0, 1.0, fraction=0.5)
    quarter = kelly_criterion(0.6, 2.0, 1.0, fraction=0.25)
    assert math.isclose(half, full * 0.5)
    assert math.isclose(quarter, full * 0.25)


def test_kelly_rejects_out_of_range_win_rate() -> None:
    assert kelly_criterion(-0.1, 1.0, 1.0) == 0.0
    assert kelly_criterion(1.1, 1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Cross-module acceptance: sizing respects risk-shield position cap.
# ---------------------------------------------------------------------------


def test_fixed_fractional_can_be_made_to_respect_position_cap() -> None:
    """If risk_pct <= stop_loss_pct * max_position_size_pct, the
    fixed-fractional size is <= max_position_size_pct of equity.

    This is the algebraic identity behind the sprint acceptance
    criterion 'sizing يحترم max_position_size من Risk Shield'.
    The integration is enforced at the RiskShield.check() call site;
    here we lock the math down so an honest caller will pass.
    """
    cfg = RiskConfig(max_position_size_pct=0.20)
    equity = 10_000.0
    stop_loss_pct = 0.02
    # Pick risk_pct so the resulting size hits exactly the cap.
    risk_pct = stop_loss_pct * cfg.max_position_size_pct  # 0.004
    size = fixed_fractional(equity, risk_pct, stop_loss_pct)
    assert size == pytest.approx(equity * cfg.max_position_size_pct)
    assert size / equity == pytest.approx(cfg.max_position_size_pct)
