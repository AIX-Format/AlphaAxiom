"""
Tests for engine.risk_shield.

Coverage matrix:

  - Per-rule rejection: position-size, concurrent-positions,
    daily-loss, max-drawdown, non-actionable, invalid state.
  - Approval path: a clean signal with healthy state goes through.
  - Audit log: every rejection is recorded with the right fields;
    approvals are not.
  - Emergency stop: triggered by daily loss; triggered by drawdown;
    blocks all subsequent signals; auto-clears after cooldown_hours;
    can be cleared manually.
  - Deterministic time: every test injects a fake clock so the
    cooldown logic is testable without sleeping.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

import math  # noqa: E402

from engine.risk_shield import (  # noqa: E402
    PortfolioState,
    RejectionReason,
    RiskConfig,
    RiskShield,
)
from engine.strategies.base import TradingSignal  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(action: str = "BUY", confidence: float = 0.7) -> TradingSignal:
    return TradingSignal(
        symbol="BTC/USDT",
        action=action,
        confidence=confidence,
        strategy="test",
        entry_price=50_000.0,
        stop_loss=49_000.0,
        take_profit=52_000.0,
    )


def _frozen_clock(*, t: datetime):
    """Return a callable that always returns the given timestamp."""

    def _now() -> datetime:
        return t

    return _now


T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Approval path + non-actionable signals
# ---------------------------------------------------------------------------


def test_approves_healthy_signal() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    state = PortfolioState(
        equity=10_000.0,
        open_positions=0,
        proposed_notional=500.0,
        daily_pnl=0.0,
        high_water_mark=10_000.0,
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is True
    assert decision.reason is None
    assert shield.audit_log() == []


def test_rejects_hold_signal_without_logging() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    state = PortfolioState(
        equity=10_000.0, open_positions=0, proposed_notional=0.0
    )
    decision = shield.check(_signal(action="HOLD", confidence=0.0), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.NON_ACTIONABLE_SIGNAL
    # HOLD is not logged: it is not a rejection in the audit sense.
    assert shield.audit_log() == []


def test_rejects_zero_confidence_buy() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    state = PortfolioState(
        equity=10_000.0, open_positions=0, proposed_notional=500.0
    )
    # Confidence 0 on BUY is still non-actionable.
    sig = TradingSignal(
        symbol="BTC/USDT", action="BUY", confidence=0.0, strategy="t"
    )
    decision = shield.check(sig, state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.NON_ACTIONABLE_SIGNAL


# ---------------------------------------------------------------------------
# Per-rule rejections
# ---------------------------------------------------------------------------


def test_rejects_oversized_position() -> None:
    shield = RiskShield(
        config=RiskConfig(max_position_size_pct=0.10),
        clock=_frozen_clock(t=T0),
    )
    state = PortfolioState(
        equity=10_000.0,
        open_positions=0,
        proposed_notional=2_000.0,  # 20% > 10% cap
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.POSITION_SIZE_EXCEEDED
    # Audit log should now have exactly one entry.
    log = shield.audit_log()
    assert len(log) == 1
    assert log[0].reason == RejectionReason.POSITION_SIZE_EXCEEDED
    assert log[0].symbol == "BTC/USDT"
    assert log[0].action == "BUY"


def test_rejects_at_concurrent_position_cap() -> None:
    shield = RiskShield(
        config=RiskConfig(max_concurrent_positions=3),
        clock=_frozen_clock(t=T0),
    )
    state = PortfolioState(
        equity=10_000.0,
        open_positions=3,
        proposed_notional=100.0,
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.MAX_CONCURRENT_POSITIONS


def test_rejects_invalid_state() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    bad = PortfolioState(
        equity=0.0, open_positions=0, proposed_notional=100.0
    )
    decision = shield.check(_signal(), bad)
    assert decision.approved is False
    assert decision.reason == RejectionReason.INVALID_PORTFOLIO_STATE


# ---------------------------------------------------------------------------
# Emergency stop: daily loss trip + cooldown
# ---------------------------------------------------------------------------


def test_daily_loss_trips_emergency_stop_and_rejects_signal() -> None:
    shield = RiskShield(
        config=RiskConfig(daily_loss_limit_pct=0.02),
        clock=_frozen_clock(t=T0),
    )
    # Daily PnL of -3% on 10k equity (-300) exceeds the 2% limit.
    state = PortfolioState(
        equity=10_000.0,
        open_positions=0,
        proposed_notional=100.0,
        daily_pnl=-300.0,
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.DAILY_LOSS_LIMIT
    assert shield.is_emergency_stop_active() is True


def test_emergency_stop_rejects_everything_during_cooldown() -> None:
    clock_t = [T0]
    shield = RiskShield(
        config=RiskConfig(cooldown_hours=24.0),
        clock=lambda: clock_t[0],
    )
    shield.emergency_stop(reason="manual test")

    state = PortfolioState(
        equity=10_000.0, open_positions=0, proposed_notional=100.0
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.EMERGENCY_STOP_ACTIVE

    # One hour later: still active.
    clock_t[0] = T0 + timedelta(hours=1)
    decision = shield.check(_signal(), state)
    assert decision.reason == RejectionReason.EMERGENCY_STOP_ACTIVE


def test_emergency_stop_auto_clears_after_cooldown() -> None:
    clock_t = [T0]
    shield = RiskShield(
        config=RiskConfig(cooldown_hours=24.0),
        clock=lambda: clock_t[0],
    )
    shield.emergency_stop(reason="manual test")

    # Just before 24h: still active.
    clock_t[0] = T0 + timedelta(hours=23, minutes=59)
    assert shield.is_emergency_stop_active() is True

    # 24h+ later: auto-clears.
    clock_t[0] = T0 + timedelta(hours=24, minutes=1)
    assert shield.is_emergency_stop_active() is False

    # And new signals are accepted again.
    state = PortfolioState(
        equity=10_000.0, open_positions=0, proposed_notional=100.0
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is True


def test_emergency_stop_can_be_cleared_manually() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    shield.emergency_stop(reason="operator override test")
    assert shield.is_emergency_stop_active() is True
    shield.clear_emergency_stop()
    assert shield.is_emergency_stop_active() is False


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------


def test_drawdown_circuit_breaker_trips() -> None:
    shield = RiskShield(
        config=RiskConfig(max_drawdown_pct=0.05),
        clock=_frozen_clock(t=T0),
    )
    state = PortfolioState(
        equity=9_400.0,
        open_positions=0,
        proposed_notional=100.0,
        high_water_mark=10_000.0,  # 6% drawdown > 5% cap
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.MAX_DRAWDOWN
    assert shield.is_emergency_stop_active() is True


def test_drawdown_below_threshold_passes() -> None:
    shield = RiskShield(
        config=RiskConfig(max_drawdown_pct=0.05),
        clock=_frozen_clock(t=T0),
    )
    state = PortfolioState(
        equity=9_700.0,
        open_positions=0,
        proposed_notional=100.0,
        high_water_mark=10_000.0,  # 3% drawdown, under the cap
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Audit log shape
# ---------------------------------------------------------------------------


def test_audit_log_records_each_rejection_in_order() -> None:
    shield = RiskShield(
        config=RiskConfig(
            max_position_size_pct=0.10,
            max_concurrent_positions=1,
        ),
        clock=_frozen_clock(t=T0),
    )
    # Reject 1: oversized.
    shield.check(
        _signal(),
        PortfolioState(
            equity=10_000.0, open_positions=0, proposed_notional=5_000.0
        ),
    )
    # Reject 2: too many positions.
    shield.check(
        _signal(),
        PortfolioState(
            equity=10_000.0, open_positions=2, proposed_notional=100.0
        ),
    )
    log = shield.audit_log()
    assert len(log) == 2
    reasons = [e.reason for e in log]
    assert reasons == [
        RejectionReason.POSITION_SIZE_EXCEEDED,
        RejectionReason.MAX_CONCURRENT_POSITIONS,
    ]
    # All entries share the same fake clock timestamp.
    for entry in log:
        assert entry.timestamp == T0


def test_audit_log_returns_a_snapshot_copy() -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    shield.check(
        _signal(),
        PortfolioState(
            equity=10_000.0, open_positions=10, proposed_notional=100.0
        ),
    )
    snapshot = shield.audit_log()
    snapshot.clear()
    # Internal log untouched.
    assert len(shield.audit_log()) == 1


# ---------------------------------------------------------------------------
# RiskConfig bounds validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_position_size_pct": -0.01},
        {"max_position_size_pct": 1.5},
        {"max_position_size_pct": float("nan")},
        {"max_position_size_pct": float("inf")},
        {"daily_loss_limit_pct": -0.01},
        {"daily_loss_limit_pct": 2.0},
        {"daily_loss_limit_pct": float("nan")},
        {"max_drawdown_pct": -0.01},
        {"max_drawdown_pct": 1.5},
        {"max_drawdown_pct": float("inf")},
        {"max_concurrent_positions": 0},
        {"max_concurrent_positions": -3},
        {"max_concurrent_positions": True},  # bool slips past int check w/o explicit guard
        {"cooldown_hours": -1.0},
        {"cooldown_hours": float("nan")},
        {"cooldown_hours": float("inf")},
    ],
)
def test_risk_config_rejects_out_of_range_values(kwargs) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**kwargs)


def test_risk_config_accepts_edge_values() -> None:
    # 0.0 and 1.0 are valid for fractions; cooldown=0 is valid (no
    # cooldown). These should not raise.
    RiskConfig(
        max_position_size_pct=0.0,
        max_concurrent_positions=1,
        daily_loss_limit_pct=1.0,
        max_drawdown_pct=0.0,
        cooldown_hours=0.0,
    )


# ---------------------------------------------------------------------------
# Portfolio state hardening: NaN/inf and structurally impossible fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state_kwargs",
    [
        {"equity": float("nan"), "open_positions": 0, "proposed_notional": 100.0},
        {"equity": float("inf"), "open_positions": 0, "proposed_notional": 100.0},
        {
            "equity": 10_000.0,
            "open_positions": 0,
            "proposed_notional": float("nan"),
        },
        {
            "equity": 10_000.0,
            "open_positions": 0,
            "proposed_notional": float("inf"),
        },
        {
            "equity": 10_000.0,
            "open_positions": 0,
            "proposed_notional": 100.0,
            "daily_pnl": float("nan"),
        },
        {
            "equity": 10_000.0,
            "open_positions": -1,
            "proposed_notional": 100.0,
        },
        {
            "equity": 10_000.0,
            "open_positions": 0,
            "proposed_notional": 100.0,
            "high_water_mark": -500.0,
        },
        {
            "equity": 10_000.0,
            "open_positions": 0,
            "proposed_notional": 100.0,
            "high_water_mark": float("nan"),
        },
    ],
)
def test_check_rejects_invalid_portfolio_state(state_kwargs) -> None:
    shield = RiskShield(clock=_frozen_clock(t=T0))
    state = PortfolioState(**state_kwargs)
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.INVALID_PORTFOLIO_STATE
    # And the invalid state is logged with its detail string.
    assert len(shield.audit_log()) == 1


def test_check_does_not_get_fooled_by_nan_comparisons() -> None:
    """NaN comparisons return False, which would silently bypass the
    `equity > 0` guard if we forgot the explicit isfinite check.
    """
    shield = RiskShield(clock=_frozen_clock(t=T0))
    state = PortfolioState(
        equity=math.nan,
        open_positions=0,
        proposed_notional=100.0,
    )
    decision = shield.check(_signal(), state)
    assert decision.approved is False
    assert decision.reason == RejectionReason.INVALID_PORTFOLIO_STATE
