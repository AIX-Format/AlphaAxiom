"""
End-to-end signal flow.

`SignalPipeline.run_once(df)` chains the four pieces of Sprint 1:

    strategy.generate_signal(df)
        -> position_sizing.fixed_fractional(...)
        -> risk_shield.check(signal, state)
        -> engine.execute_trade(...)        (only when approved)

It is intentionally a thin orchestrator. Strategies, the risk shield,
and the engine each enforce their own contracts; the pipeline just
wires them together with the right state snapshot, decides what
notional to propose, and records the outcome in a `PipelineResult`.

The pipeline never raises. Every failure path (non-actionable
signal, risk rejection, zero size, execution error) returns a
`PipelineResult` with the relevant fields populated so the caller
can audit the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import pandas as pd

from .position_sizing import fixed_fractional
from .risk_shield import (
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskShield,
)
from .strategies.base import Strategy, TradingSignal

logger = logging.getLogger(__name__)


class SupportsExecute(Protocol):
    """Minimal contract the pipeline needs from a trading engine.

    `TradingEngine` in `engine.trading_core` already satisfies this.
    A separate Protocol keeps the pipeline testable with simple stubs
    that do not need to spin up an exchange connection.
    """

    portfolio: Any  # Portfolio-like: get_balance, daily_pnl, etc.

    async def execute_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class PipelineResult:
    """Outcome of a single `run_once` call.

    Always present:
      - `signal`: what the strategy produced.
      - `risk_decision`: the shield's verdict (approved or rejected).
      - `position_size`: notional dollars proposed (0.0 if not
        actionable or rejected before sizing was relevant).

    Present only when the trade was actually sent to the engine:
      - `execution`: raw dict returned by `engine.execute_trade`.
    """

    signal: TradingSignal
    risk_decision: RiskDecision
    position_size: float
    execution: Optional[Dict[str, Any]] = None

    @property
    def executed(self) -> bool:
        return self.execution is not None


@dataclass
class PipelineConfig:
    """Knobs for the sizing math the pipeline applies.

    Defaults are picked so the resulting notional fits comfortably
    inside the risk shield's default 20% per-position cap. The
    closed-form constraint is:

        notional = equity * risk_per_trade_pct / stop_loss_pct
        notional / equity <= max_position_size_pct

    With a 2% stop, `risk_per_trade_pct = 0.003` gives 15% notional,
    safely under the 20% cap. Callers that want tighter sizing or a
    different stop convention should override this config.
    """

    risk_per_trade_pct: float = 0.003  # 0.3% of equity per trade
    fallback_stop_pct: float = 0.02    # used when signal has no stop


class SignalPipeline:
    """Strategy + risk + sizing + execution, wired end-to-end."""

    def __init__(
        self,
        strategy: Strategy,
        risk_shield: RiskShield,
        engine: SupportsExecute,
        config: Optional[PipelineConfig] = None,
        *,
        execute_live: bool = True,
    ) -> None:
        self.strategy = strategy
        self.risk_shield = risk_shield
        self.engine = engine
        self.config = config or PipelineConfig()
        # `execute_live=False` short-circuits the execute_trade call
        # so the pipeline can be used in paper / shadow mode without
        # touching the exchange adapter.
        self.execute_live = execute_live

    # ------------------------------------------------------------------

    async def run_once(self, df: pd.DataFrame) -> PipelineResult:
        """Run one pass: generate, size, risk-check, optionally execute."""
        signal = self.strategy.generate_signal(df)

        # Skip every downstream step for HOLD / zero-confidence: the
        # risk shield would also reject them, but doing it here keeps
        # the audit log free of routine HOLDs.
        if not signal.is_actionable():
            return PipelineResult(
                signal=signal,
                risk_decision=RiskDecision(
                    approved=False,
                    reason=RejectionReason.NON_ACTIONABLE_SIGNAL,
                    detail="signal is HOLD or zero confidence",
                ),
                position_size=0.0,
            )

        proposed_size = self._compute_size(signal)
        state = self._snapshot_portfolio(proposed_size)

        decision = self.risk_shield.check(signal, state)
        if not decision.approved or proposed_size <= 0.0:
            if not decision.approved:
                logger.info(
                    "Pipeline rejected %s %s: %s",
                    signal.symbol,
                    signal.action,
                    decision.reason.value if decision.reason else "unknown",
                )
            return PipelineResult(
                signal=signal,
                risk_decision=decision,
                position_size=proposed_size if decision.approved else 0.0,
            )

        execution = await self._execute(signal, proposed_size)
        return PipelineResult(
            signal=signal,
            risk_decision=decision,
            position_size=proposed_size,
            execution=execution,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_size(self, signal: TradingSignal) -> float:
        """Convert a signal + portfolio equity into a notional size.

        Uses the signal's own stop distance when it has one, falling
        back to `config.fallback_stop_pct` otherwise. Returns 0.0 on
        any bad input so the risk shield can reject cleanly without
        a math error.
        """
        equity = self._equity()
        if equity <= 0.0:
            return 0.0

        entry = signal.entry_price
        stop = signal.stop_loss
        stop_pct: float
        if entry and stop and entry > 0.0:
            stop_pct = abs(float(entry) - float(stop)) / float(entry)
            if stop_pct <= 0.0:
                stop_pct = self.config.fallback_stop_pct
        else:
            stop_pct = self.config.fallback_stop_pct

        return fixed_fractional(
            equity=equity,
            risk_pct=self.config.risk_per_trade_pct,
            stop_loss_pct=stop_pct,
        )

    def _snapshot_portfolio(self, proposed_notional: float) -> PortfolioState:
        """Capture the state the risk shield needs for one decision."""
        portfolio = self.engine.portfolio
        equity = self._equity()
        open_positions = self._open_position_count()
        daily_pnl = (
            portfolio.daily_pnl()
            if hasattr(portfolio, "daily_pnl")
            else 0.0
        )
        hwm = (
            portfolio.high_water_mark()
            if hasattr(portfolio, "high_water_mark")
            else equity
        )
        return PortfolioState(
            equity=equity,
            open_positions=open_positions,
            proposed_notional=proposed_notional,
            daily_pnl=daily_pnl,
            high_water_mark=hwm,
        )

    def _equity(self) -> float:
        portfolio = self.engine.portfolio
        getter = getattr(portfolio, "get_balance", None)
        if callable(getter):
            return float(getter())
        return float(getattr(portfolio, "balance", 0.0))

    def _open_position_count(self) -> int:
        portfolio = self.engine.portfolio
        getter = getattr(portfolio, "get_positions", None)
        positions = getter() if callable(getter) else getattr(portfolio, "positions", {})
        try:
            return len(positions)
        except TypeError:
            return 0

    async def _execute(
        self,
        signal: TradingSignal,
        notional: float,
    ) -> Optional[Dict[str, Any]]:
        """Send the order to the engine adapter.

        When `execute_live` is False the pipeline returns a shadow
        execution record so callers can still see what would have
        happened. This is the paper-trading hook.
        """
        order_type = "buy" if signal.action == "BUY" else "sell"
        entry_price = signal.entry_price or 0.0
        amount = notional / entry_price if entry_price > 0.0 else 0.0

        params: Dict[str, Any] = {
            "symbol": signal.symbol,
            "order_type": order_type,
            "amount": amount,
            "price": signal.entry_price,
            "notional": notional,
            "strategy": signal.strategy,
        }

        if not self.execute_live:
            return {
                "success": True,
                "shadow": True,
                "order_id": f"shadow_{signal.timestamp}",
                "params": params,
            }

        try:
            return await self.engine.execute_trade(params)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("execute_trade raised: %s", exc)
            return {"success": False, "error": str(exc)}
