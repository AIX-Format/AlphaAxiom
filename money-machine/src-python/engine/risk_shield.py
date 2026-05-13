"""
Aladdin Risk Shield.

A single class, `RiskShield`, sits between strategy output and the
execution adapter. Every order proposed by the engine flows through
`check(signal, portfolio_state)` and is either approved or rejected
with a reason. Rejections are recorded in an audit log so we can
diagnose later why a strategy did not trade.

Enforced rules (configurable via `RiskConfig`):

1. **max_position_size_pct**: the notional value of any single new
   position cannot exceed this fraction of current equity.
2. **max_concurrent_positions**: at most N positions open at once.
3. **daily_loss_limit_pct**: if realised + unrealised PnL for the
   trading day drops below `-equity * daily_loss_limit_pct`, the
   shield triggers an emergency stop.
4. **max_drawdown_pct**: if `equity / high_water_mark <= 1 -
   max_drawdown_pct`, the shield triggers an emergency stop.

`emergency_stop()` flips a flag, records the timestamp, and stays
active for `cooldown_hours` (24 by default). While active the shield
rejects every signal with `EMERGENCY_STOP_ACTIVE`. The caller is
responsible for actually closing open positions; the shield only
authorises future trades.

The shield is intentionally stateful but small. State lives on the
instance and can be persisted by callers if needed (e.g. between
desktop app launches). All time inputs accept an injectable `now`
clock so tests stay deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, List, Optional

from .strategies.base import TradingSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """Tunable thresholds. All percentages are fractions in [0, 1]."""

    max_position_size_pct: float = 0.20
    max_concurrent_positions: int = 3
    daily_loss_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.05
    cooldown_hours: float = 24.0


@dataclass
class PortfolioState:
    """Minimal snapshot the shield needs to evaluate a signal."""

    equity: float
    open_positions: int
    proposed_notional: float
    daily_pnl: float = 0.0
    high_water_mark: Optional[float] = None

    def __post_init__(self) -> None:
        if self.high_water_mark is None:
            self.high_water_mark = self.equity


class RejectionReason(str, Enum):
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    MAX_CONCURRENT_POSITIONS = "MAX_CONCURRENT_POSITIONS"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    EMERGENCY_STOP_ACTIVE = "EMERGENCY_STOP_ACTIVE"
    NON_ACTIONABLE_SIGNAL = "NON_ACTIONABLE_SIGNAL"
    INVALID_PORTFOLIO_STATE = "INVALID_PORTFOLIO_STATE"


@dataclass(frozen=True)
class RiskDecision:
    """Result of `RiskShield.check`."""

    approved: bool
    reason: Optional[RejectionReason] = None
    detail: str = ""


@dataclass
class AuditEntry:
    """Single entry in the rejection audit log."""

    timestamp: datetime
    symbol: str
    action: str
    reason: RejectionReason
    detail: str
    confidence: float


# ---------------------------------------------------------------------------
# Risk shield
# ---------------------------------------------------------------------------


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class RiskShield:
    """Pre-trade risk gate with audit trail and emergency stop."""

    config: RiskConfig = field(default_factory=RiskConfig)
    clock: Clock = field(default=_utc_now)

    # Mutable state.
    _emergency_stop_until: Optional[datetime] = None
    _audit_log: List[AuditEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        signal: TradingSignal,
        state: PortfolioState,
    ) -> RiskDecision:
        """Run every rule in order. First failure wins.

        Approved signals are returned with `approved=True` and no
        reason. Rejected signals are recorded in the audit log.
        """
        # Always re-evaluate emergency stop first: state from an old
        # check might have expired.
        if self.is_emergency_stop_active():
            return self._reject(
                signal,
                RejectionReason.EMERGENCY_STOP_ACTIVE,
                f"emergency stop active until {self._emergency_stop_until!s}",
            )

        if not signal.is_actionable():
            # HOLD or zero-confidence: nothing to approve, nothing to
            # log noisily.
            return RiskDecision(
                approved=False,
                reason=RejectionReason.NON_ACTIONABLE_SIGNAL,
                detail="signal is HOLD or has zero confidence",
            )

        if state.equity <= 0 or state.proposed_notional < 0:
            return self._reject(
                signal,
                RejectionReason.INVALID_PORTFOLIO_STATE,
                f"equity={state.equity}, proposed_notional={state.proposed_notional}",
            )

        # Rule 1: per-position size cap.
        size_ratio = state.proposed_notional / state.equity
        if size_ratio > self.config.max_position_size_pct:
            return self._reject(
                signal,
                RejectionReason.POSITION_SIZE_EXCEEDED,
                f"size {size_ratio:.2%} > cap {self.config.max_position_size_pct:.2%}",
            )

        # Rule 2: concurrent position cap.
        if state.open_positions >= self.config.max_concurrent_positions:
            return self._reject(
                signal,
                RejectionReason.MAX_CONCURRENT_POSITIONS,
                f"{state.open_positions} open >= cap {self.config.max_concurrent_positions}",
            )

        # Rule 3: daily loss.
        daily_loss_ratio = -state.daily_pnl / state.equity if state.daily_pnl < 0 else 0.0
        if daily_loss_ratio >= self.config.daily_loss_limit_pct:
            # Trip the emergency stop and reject this signal.
            self.emergency_stop(
                reason=f"daily loss {daily_loss_ratio:.2%} >= "
                f"limit {self.config.daily_loss_limit_pct:.2%}"
            )
            return self._reject(
                signal,
                RejectionReason.DAILY_LOSS_LIMIT,
                f"daily loss {daily_loss_ratio:.2%} >= "
                f"limit {self.config.daily_loss_limit_pct:.2%}",
            )

        # Rule 4: drawdown from high-water mark.
        hwm = state.high_water_mark or state.equity
        if hwm > 0:
            drawdown = (hwm - state.equity) / hwm
            if drawdown >= self.config.max_drawdown_pct:
                self.emergency_stop(
                    reason=f"drawdown {drawdown:.2%} >= "
                    f"limit {self.config.max_drawdown_pct:.2%}"
                )
                return self._reject(
                    signal,
                    RejectionReason.MAX_DRAWDOWN,
                    f"drawdown {drawdown:.2%} >= "
                    f"limit {self.config.max_drawdown_pct:.2%}",
                )

        return RiskDecision(approved=True)

    def emergency_stop(self, *, reason: str = "") -> datetime:
        """Trip the emergency stop. Returns the time it expires."""
        now = self.clock()
        self._emergency_stop_until = now + timedelta(
            hours=self.config.cooldown_hours
        )
        logger.warning(
            "RiskShield emergency stop tripped until %s: %s",
            self._emergency_stop_until.isoformat(),
            reason or "no reason given",
        )
        return self._emergency_stop_until

    def clear_emergency_stop(self) -> None:
        """Manually clear the emergency stop (operator override)."""
        self._emergency_stop_until = None
        logger.info("RiskShield emergency stop manually cleared")

    def is_emergency_stop_active(self) -> bool:
        if self._emergency_stop_until is None:
            return False
        if self.clock() >= self._emergency_stop_until:
            # Cooldown expired, auto-clear.
            self._emergency_stop_until = None
            return False
        return True

    def audit_log(self) -> List[AuditEntry]:
        """Return a snapshot of every rejection so far (ordered)."""
        return list(self._audit_log)

    def audit_rejection(
        self,
        signal: TradingSignal,
        reason: RejectionReason,
        detail: str = "",
    ) -> AuditEntry:
        """Record a rejection in the audit log and return the entry."""
        entry = AuditEntry(
            timestamp=self.clock(),
            symbol=signal.symbol,
            action=signal.action,
            reason=reason,
            detail=detail,
            confidence=signal.confidence,
        )
        self._audit_log.append(entry)
        logger.info(
            "RiskShield rejection: %s %s (%s) %s",
            entry.symbol,
            entry.action,
            entry.reason.value,
            entry.detail,
        )
        return entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        signal: TradingSignal,
        reason: RejectionReason,
        detail: str,
    ) -> RiskDecision:
        self.audit_rejection(signal, reason, detail)
        return RiskDecision(approved=False, reason=reason, detail=detail)
