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

from .adapters import (
    ExecutionAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
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
    """Strategy + risk + sizing + execution, wired end-to-end.

    Two execution backends are supported:

    1. **Adapter** (`ExecutionAdapter` from `engine.adapters`). The
       pipeline builds an `OrderRequest` from the signal and calls
       `adapter.place_order`. The returned `OrderResult` is mapped
       into the execution dict on `PipelineResult.execution`. This
       is the path every new venue (paper, MT5, CCXT, EVM, Solana)
       plugs into.

    2. **Legacy engine** (`SupportsExecute`). Pre-adapter callers
       pass an object exposing `engine.execute_trade(params)` and
       `engine.portfolio`. Kept for backward compatibility so the
       existing `TradingEngine` keeps working while strategies
       migrate.

    Exactly one of `adapter` or `engine` must be supplied. Risk
    shield state (equity, daily PnL, high-water mark) still comes
    from the portfolio object on the engine when present, or from
    the adapter's `get_balance` / `get_positions` when only an
    adapter is given.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_shield: RiskShield,
        engine: Optional[SupportsExecute] = None,
        config: Optional[PipelineConfig] = None,
        *,
        adapter: Optional[ExecutionAdapter] = None,
        execute_live: bool = True,
    ) -> None:
        if adapter is None and engine is None:
            raise ValueError(
                "SignalPipeline requires either an adapter or a legacy engine"
            )
        self.strategy = strategy
        self.risk_shield = risk_shield
        self.engine = engine
        self.adapter = adapter
        self.config = config or PipelineConfig()
        # `execute_live=False` short-circuits the execution call so
        # the pipeline can be used in paper / shadow mode without
        # touching the venue.
        self.execute_live = execute_live
        # Instance-scoped caches for adapter portfolio snapshots.
        # Defined here (not as class attributes) so two pipelines
        # cannot accidentally share state on a mutable dict.
        self._adapter_balance_cache: float = 0.0
        self._adapter_positions_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------

    async def run_once(self, df: pd.DataFrame) -> PipelineResult:
        """Run one pass: generate, size, risk-check, optionally execute.

        The pipeline never raises. Any escape from the strategy's
        generate_signal (a transient bug, malformed input, a buggy
        third-party indicator) is converted into a synthetic HOLD
        PipelineResult so the caller still gets a record of the tick
        and downstream telemetry (audit log, dashboards) sees the
        failure instead of an opaque crash.
        """
        # Refresh adapter portfolio cache up front so `_equity` and
        # `_open_position_count` can stay sync helpers downstream.
        # We do this whenever an adapter is configured (even if a
        # legacy engine is also present) because the adapter is the
        # source of truth for whichever backend will actually
        # execute the trade. Fail CLOSED on read errors: if we
        # cannot trust the portfolio snapshot we must not approve
        # new orders against stale state.
        if self.adapter is not None:
            try:
                self._adapter_balance_cache = float(await self.adapter.get_balance())
                self._adapter_positions_cache = dict(await self.adapter.get_positions())
            except Exception as exc:
                logger.error(
                    "Adapter portfolio refresh failed; failing closed: %s", exc
                )
                fallback = TradingSignal(
                    symbol=getattr(self.strategy, "symbol", ""),
                    action="HOLD",
                    confidence=0.0,
                    strategy=getattr(self.strategy, "name", ""),
                    reasoning=f"adapter portfolio refresh failed: {exc}",
                )
                return PipelineResult(
                    signal=fallback,
                    risk_decision=RiskDecision(
                        approved=False,
                        reason=RejectionReason.INVALID_PORTFOLIO_STATE,
                        detail=f"adapter portfolio refresh failed: {exc}",
                    ),
                    position_size=0.0,
                )

        try:
            signal = self.strategy.generate_signal(df)
        except Exception as exc:
            logger.error(
                "Strategy %s raised on generate_signal: %s",
                getattr(self.strategy, "name", type(self.strategy).__name__),
                exc,
            )
            symbol = getattr(self.strategy, "symbol", "")
            fallback = TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                strategy=getattr(self.strategy, "name", ""),
                reasoning=f"strategy raised: {exc}",
            )
            return PipelineResult(
                signal=fallback,
                risk_decision=RiskDecision(
                    approved=False,
                    reason=RejectionReason.NON_ACTIONABLE_SIGNAL,
                    detail=f"strategy exception: {exc}",
                ),
                position_size=0.0,
            )

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
            # Preserve the proposed size on the result even when the
            # shield rejects the trade. Audit/telemetry consumers need
            # to be able to distinguish "no size was computed" (e.g.
            # equity <= 0) from "a sized order was explicitly blocked
            # by the risk shield" (e.g. POSITION_SIZE_EXCEEDED). Both
            # cases reach this branch; only the latter carries useful
            # sizing context.
            return PipelineResult(
                signal=signal,
                risk_decision=decision,
                position_size=proposed_size,
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
        """Capture the state the risk shield needs for one decision.

        Daily PnL fed into the shield combines realised PnL since
        UTC midnight with the current unrealised mark on open
        positions. Without the unrealised piece a strategy could
        sit on a 10% open drawdown all day and only trip the
        emergency stop after the position was closed, which defeats
        the point of `daily_loss_limit`.

        Adapter-only pipelines (no legacy engine) get daily_pnl=0
        and high_water_mark=equity because a stateless venue cannot
        tell us either. The first PR that ships a stateful adapter
        bookkeeping layer can override this method or extend the
        snapshot helper.
        """
        equity = self._equity()
        open_positions = self._open_position_count()
        # daily_pnl and high_water_mark come from the engine
        # Portfolio if we have one AND the adapter is NOT executing
        # (otherwise the engine Portfolio is stale relative to
        # actual fills). When the adapter is the execution backend
        # these default to 0 / equity; a stateful portfolio decorator
        # for adapters is the natural follow-up.
        portfolio = self._portfolio() if self.adapter is None else None
        if portfolio is not None:
            daily_pnl = self._read_metric(portfolio, "daily_pnl", default=0.0)
            unrealised = self._read_metric(portfolio, "calculate_pnl", default=0.0)
            daily_pnl_total = float(daily_pnl) + float(unrealised)
            hwm = self._read_metric(portfolio, "high_water_mark", default=equity)
        else:
            daily_pnl_total = 0.0
            hwm = equity
        return PortfolioState(
            equity=equity,
            open_positions=open_positions,
            proposed_notional=proposed_notional,
            daily_pnl=daily_pnl_total,
            high_water_mark=float(hwm),
        )

    @staticmethod
    def _read_metric(portfolio: Any, name: str, *, default: float) -> float:
        """Read a portfolio metric exposed either as a method or attribute.

        Some adapters expose `high_water_mark` as a numeric field or
        a `@property`; others as a method. We accept both shapes so
        the pipeline does not raise `TypeError` on a bare attribute.
        """
        attr = getattr(portfolio, name, None)
        if attr is None:
            return float(default)
        if callable(attr):
            try:
                return float(attr())
            except Exception:
                return float(default)
        try:
            return float(attr)
        except (TypeError, ValueError):
            return float(default)

    def _equity(self) -> float:
        # When an adapter is the execution backend, the adapter's
        # balance is the source of truth (it reflects fills that
        # bypass the legacy engine's Portfolio). Fall back to the
        # engine's portfolio only when no adapter is configured.
        if self.adapter is not None:
            return float(self._adapter_balance_cache)
        portfolio = self._portfolio()
        if portfolio is None:
            return 0.0
        getter = getattr(portfolio, "get_balance", None)
        if callable(getter):
            return float(getter())
        return float(getattr(portfolio, "balance", 0.0))

    def _open_position_count(self) -> int:
        if self.adapter is not None:
            return len(self._adapter_positions_cache)
        portfolio = self._portfolio()
        if portfolio is None:
            return 0
        getter = getattr(portfolio, "get_positions", None)
        positions = getter() if callable(getter) else getattr(portfolio, "positions", {})
        try:
            return len(positions)
        except TypeError:
            return 0

    def _portfolio(self) -> Any:
        """Return the portfolio-state source for daily PnL and HWM.

        Only consulted when no adapter is configured. The engine's
        Portfolio owns daily_pnl and high_water_mark history that
        a stateless adapter cannot provide. When an adapter IS the
        execution backend its balance/positions are read from the
        cached snapshot in `_equity` / `_open_position_count` and
        daily_pnl/HWM default to 0/equity.
        """
        if self.engine is not None:
            return getattr(self.engine, "portfolio", None)
        return None



    async def _execute(
        self,
        signal: TradingSignal,
        notional: float,
    ) -> Optional[Dict[str, Any]]:
        """Send the order to whichever execution backend is configured.

        When `execute_live` is False the pipeline returns a shadow
        execution record so callers can still see what would have
        happened. This is the paper-trading hook.

        When an `ExecutionAdapter` is configured the order goes
        through `adapter.place_order` and the `OrderResult` is
        translated into the execution dict. Otherwise the legacy
        `engine.execute_trade(params)` path is used.
        """
        if not self.execute_live:
            order_type = "buy" if signal.action == "BUY" else "sell"
            entry_price = signal.entry_price or 0.0
            amount = notional / entry_price if entry_price > 0.0 else 0.0
            return {
                "success": True,
                "shadow": True,
                "order_id": f"shadow_{signal.timestamp}",
                "params": {
                    "symbol": signal.symbol,
                    "order_type": order_type,
                    "amount": amount,
                    "price": signal.entry_price,
                    "notional": notional,
                    "strategy": signal.strategy,
                },
            }

        if self.adapter is not None:
            return await self._execute_via_adapter(signal, notional)
        return await self._execute_via_engine(signal, notional)

    async def _execute_via_adapter(
        self, signal: TradingSignal, notional: float
    ) -> Dict[str, Any]:
        side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL
        request = OrderRequest(
            client_order_id=self._client_order_id(signal),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            notional=float(notional),
            limit_price=None,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            metadata={"signal_timestamp": signal.timestamp},
        )
        try:
            result: OrderResult = await self.adapter.place_order(request)
        except Exception as exc:
            logger.error("adapter.place_order raised: %s", exc)
            return {"success": False, "error": str(exc)}
        return {
            "success": result.status not in (OrderStatus.REJECTED,),
            "order_id": result.venue_order_id,
            "client_order_id": result.client_order_id,
            "status": result.status.value,
            "filled_quantity": result.filled_quantity,
            "average_fill_price": result.average_fill_price,
            "error": result.error,
        }

    async def _execute_via_engine(
        self, signal: TradingSignal, notional: float
    ) -> Optional[Dict[str, Any]]:
        if self.engine is None:
            return {"success": False, "error": "no execution backend configured"}
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
        try:
            return await self.engine.execute_trade(params)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("execute_trade raised: %s", exc)
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _client_order_id(signal: TradingSignal) -> str:
        """Stable per-signal client order id.

        Combines the strategy name, symbol, action and timestamp so
        the same signal flowing through the same pipeline always
        gets the same id. Adapters use this as the idempotency key,
        so a retry on a transient error never produces a duplicate
        order at the venue.
        """
        ts = int(signal.timestamp * 1000)
        symbol = signal.symbol.replace("/", "").replace(":", "_")
        strategy = (signal.strategy or "unknown").replace(" ", "_")
        return f"{strategy}-{symbol}-{signal.action}-{ts}"
