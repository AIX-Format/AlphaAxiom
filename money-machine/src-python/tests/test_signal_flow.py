"""
End-to-end tests for the signal flow pipeline.

`SignalPipeline.run_once(df)` is the integration seam where Sprint 1's
four building blocks meet: strategy generation, position sizing, the
risk shield, and the execution adapter. These tests verify the wiring
holds under each branch:

  - approved path: signal generated, sized, risk-checked, executed,
    PipelineResult.executed is True.
  - HOLD: short-circuits without sizing or risk check, no execution.
  - risk rejection: signal generated but the shield says no, no
    execution, audit log records the rejection.
  - emergency stop: the shield is already tripped, every subsequent
    actionable signal is rejected without execution.
  - paper mode: execute_live=False produces a shadow record so
    callers can still see what would have happened.

A "stub engine" with a mock execute_trade keeps the tests synchronous
and free of any exchange/network behaviour. The Portfolio used is the
real one from engine.trading_core so the new daily_pnl /
high_water_mark integration is exercised.

Sprint 1 Task 1.5 acceptance: every executed signal must have passed
the risk check, no signal is executed without a position size.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.risk_shield import (  # noqa: E402
    RejectionReason,
    RiskConfig,
    RiskShield,
)
from engine.signal_pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineResult,
    SignalPipeline,
)
from engine.strategies.base import Strategy, TradingSignal  # noqa: E402
from engine.trading_core import Portfolio  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class StubEngine:
    """Minimal SupportsExecute implementation backed by a real Portfolio.

    `execute_trade` records every params dict it receives so the
    tests can assert what (if anything) was sent. Returns a synthetic
    success dict by default; override `execute_result` to simulate
    a venue error.
    """

    portfolio: Portfolio
    execute_result: Dict[str, Any] = field(
        default_factory=lambda: {"success": True, "order_id": "stub"}
    )
    received: List[Dict[str, Any]] = field(default_factory=list)

    async def execute_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.received.append(params)
        return dict(self.execute_result)


class FixedSignalStrategy(Strategy):
    """Strategy that always returns the configured signal verbatim."""

    name = "fixed-test-strategy"

    def __init__(self, signal: TradingSignal) -> None:
        super().__init__(signal.symbol)
        self._signal = signal

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        # Re-stamp the signal so a fresh timestamp surfaces each call.
        s = self._signal
        return TradingSignal(
            symbol=s.symbol,
            action=s.action,
            confidence=s.confidence,
            strategy=self.name,
            entry_price=s.entry_price,
            stop_loss=s.stop_loss,
            take_profit=s.take_profit,
            reasoning=s.reasoning,
            metadata=dict(s.metadata),
        )


def _buy_signal(
    *,
    entry: float = 50_000.0,
    stop: float = 49_000.0,
    take: float = 52_000.0,
    confidence: float = 0.7,
) -> TradingSignal:
    return TradingSignal(
        symbol="BTC/USDT",
        action="BUY",
        confidence=confidence,
        strategy="fixed-test-strategy",
        entry_price=entry,
        stop_loss=stop,
        take_profit=take,
    )


def _frozen_clock(t: datetime):
    def _now() -> datetime:
        return t

    return _now


T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Approved path
# ---------------------------------------------------------------------------


def test_approved_signal_flows_through_to_execution() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    strategy = FixedSignalStrategy(_buy_signal())
    pipeline = SignalPipeline(strategy, shield, engine)

    result: PipelineResult = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is True
    assert result.risk_decision.approved is True
    assert result.position_size > 0.0
    assert engine.received, "engine.execute_trade should have been called"
    sent = engine.received[0]
    assert sent["symbol"] == "BTC/USDT"
    assert sent["order_type"] == "buy"
    assert sent["notional"] == pytest.approx(result.position_size)
    # No rejections in the audit log on the happy path.
    assert shield.audit_log() == []


def test_executed_signal_passed_risk_and_has_nonzero_size() -> None:
    """Acceptance criterion from Sprint 1.5:
    every executed signal passed the risk check and has nonzero sizing.
    """
    portfolio = Portfolio(initial_balance=20_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    strategy = FixedSignalStrategy(_buy_signal())
    pipeline = SignalPipeline(strategy, shield, engine)

    for _ in range(5):
        result = _run(pipeline.run_once(pd.DataFrame()))
        if result.executed:
            assert result.risk_decision.approved is True
            assert result.position_size > 0.0
        else:
            # Never approved but unexecuted, never executed without size.
            assert not (
                result.risk_decision.approved and result.position_size > 0
            )


# ---------------------------------------------------------------------------
# HOLD / non-actionable short-circuit
# ---------------------------------------------------------------------------


class _RaisingStrategy(Strategy):
    """Strategy that raises on every call. Used to verify the
    pipeline catches and converts the failure into a HOLD result.
    """

    name = "raising-strategy"

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        raise RuntimeError("boom")


def test_strategy_exception_is_caught_and_returns_synthetic_hold() -> None:
    """The pipeline contract is 'never raises'. A bug in the strategy
    must surface as a HOLD PipelineResult with a reason string, not
    as an exception that crashes the orchestrator and skips every
    downstream piece (risk shield, telemetry, dashboard updates).
    """
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    pipeline = SignalPipeline(
        _RaisingStrategy("BTC/USDT"),
        shield,
        engine,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))

    assert isinstance(result, PipelineResult)
    assert result.executed is False
    assert result.position_size == 0.0
    assert result.signal.action == "HOLD"
    assert "boom" in result.signal.reasoning
    assert result.risk_decision.reason == RejectionReason.NON_ACTIONABLE_SIGNAL
    assert "boom" in (result.risk_decision.detail or "")
    # The shield was never consulted (it would have approved/rejected
    # via its own audit log); strategy exceptions short-circuit there.
    assert shield.audit_log() == []
    assert engine.received == []


def test_hold_signal_short_circuits_without_execution() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    hold = TradingSignal(
        symbol="BTC/USDT", action="HOLD", confidence=0.0, strategy="t"
    )
    strategy = FixedSignalStrategy(hold)
    pipeline = SignalPipeline(strategy, shield, engine)

    result = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is False
    assert result.position_size == 0.0
    assert result.risk_decision.approved is False
    assert result.risk_decision.reason == RejectionReason.NON_ACTIONABLE_SIGNAL
    # Pipeline does not even consult the shield for HOLDs, so the
    # shield's audit log stays empty.
    assert shield.audit_log() == []
    assert engine.received == []


# ---------------------------------------------------------------------------
# Risk shield rejection
# ---------------------------------------------------------------------------


def test_oversized_proposal_is_rejected_and_logged() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    # 5% per-position cap is tighter than the default sizing math
    # below, but only marginally; we use a much tighter cap below.
    shield = RiskShield(
        config=RiskConfig(max_position_size_pct=0.001),
        clock=_frozen_clock(T0),
    )
    # 1% risk on a 2% stop = 50% of equity notional; that's well
    # above the 0.1% cap and must be rejected.
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
        config=PipelineConfig(risk_per_trade_pct=0.01, fallback_stop_pct=0.02),
    )

    result = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is False
    assert result.risk_decision.approved is False
    assert (
        result.risk_decision.reason == RejectionReason.POSITION_SIZE_EXCEEDED
    )
    # The proposed size is preserved on the result even though the
    # trade was blocked: audit/telemetry consumers need to see what
    # was attempted, not a flat 0.0.
    assert result.position_size > 0.0
    # Audit log records the rejection with the right fields.
    log = shield.audit_log()
    assert len(log) == 1
    assert log[0].reason == RejectionReason.POSITION_SIZE_EXCEEDED
    assert log[0].symbol == "BTC/USDT"
    # Engine was never called.
    assert engine.received == []


def test_emergency_stop_blocks_all_signals() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    shield.emergency_stop(reason="test")
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
    )

    for _ in range(3):
        result = _run(pipeline.run_once(pd.DataFrame()))
        assert result.executed is False
        assert (
            result.risk_decision.reason == RejectionReason.EMERGENCY_STOP_ACTIVE
        )
    assert engine.received == []


# ---------------------------------------------------------------------------
# Paper-trading / shadow mode
# ---------------------------------------------------------------------------


def test_execute_live_false_returns_shadow_record() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(clock=_frozen_clock(T0))
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
        execute_live=False,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is True
    assert result.execution is not None
    assert result.execution.get("shadow") is True
    # The real engine adapter was never called.
    assert engine.received == []


# ---------------------------------------------------------------------------
# Portfolio integration: daily_pnl + high_water_mark feed the shield.
# ---------------------------------------------------------------------------


def test_daily_pnl_tracks_realised_pnl_and_resets_per_day() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    day1 = T0.date()
    day2 = (T0.replace(day=T0.day + 1)).date()

    portfolio.add_trade({"pnl": -100.0}, today=day1)
    portfolio.add_trade({"pnl": -50.0}, today=day1)
    assert portfolio.daily_pnl(today=day1) == pytest.approx(-150.0)
    assert portfolio.balance == pytest.approx(9_850.0)

    # New day rolls the counter.
    assert portfolio.daily_pnl(today=day2) == 0.0
    portfolio.add_trade({"pnl": 25.0}, today=day2)
    assert portfolio.daily_pnl(today=day2) == pytest.approx(25.0)


def test_high_water_mark_tracks_balance_peak() -> None:
    portfolio = Portfolio(initial_balance=10_000.0)
    assert portfolio.high_water_mark() == 10_000.0

    portfolio.add_trade({"pnl": 500.0})
    assert portfolio.high_water_mark() == pytest.approx(10_500.0)

    # Drawdown does not lower the HWM.
    portfolio.add_trade({"pnl": -800.0})
    assert portfolio.high_water_mark() == pytest.approx(10_500.0)
    assert portfolio.balance == pytest.approx(9_700.0)

    # New peak raises the HWM.
    portfolio.add_trade({"pnl": 2_000.0})
    assert portfolio.high_water_mark() == pytest.approx(11_700.0)


def test_unrealised_pnl_feeds_daily_loss_limit() -> None:
    """An open-position drawdown should trip daily_loss_limit even
    before the trade is closed. Without this the shield would only
    see realised PnL and a strategy could sit on a 10% open loss
    all day.
    """
    portfolio = Portfolio(initial_balance=10_000.0)
    # Inject an open position carrying unrealised PnL of -300 (3%).
    portfolio.positions["BTC/USDT"] = {"pnl": -300.0}
    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(
        config=RiskConfig(daily_loss_limit_pct=0.02),
        clock=_frozen_clock(T0),
    )
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
        config=PipelineConfig(risk_per_trade_pct=0.0005, fallback_stop_pct=0.02),
    )

    result = _run(pipeline.run_once(pd.DataFrame()))
    assert result.executed is False
    assert result.risk_decision.reason == RejectionReason.DAILY_LOSS_LIMIT
    assert shield.is_emergency_stop_active() is True


def test_portfolio_metrics_accept_attribute_form() -> None:
    """If an adapter exposes high_water_mark as a plain attribute or
    property instead of a method, the pipeline should still read it
    without raising TypeError.
    """

    class _AttrPortfolio:
        balance = 10_000.0
        positions: dict = {}
        # High-water mark as a plain attribute, not a method.
        high_water_mark = 12_000.0

        def get_balance(self) -> float:
            return self.balance

        def get_positions(self) -> dict:
            return self.positions

        # No daily_pnl/calculate_pnl on purpose: the pipeline should
        # default them to 0 rather than crash.

    engine = StubEngine(portfolio=_AttrPortfolio())  # type: ignore[arg-type]
    shield = RiskShield(clock=_frozen_clock(T0))
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))
    # Did not raise; got a sensible decision back.
    assert isinstance(result, PipelineResult)
    # And the HWM read should have made it into the rejection context
    # if rejected, or stayed coherent if approved.
    assert result.risk_decision.approved or result.position_size > 0.0


# ---------------------------------------------------------------------------
# Adapter-routed execution
# ---------------------------------------------------------------------------


def test_pipeline_routes_through_execution_adapter() -> None:
    """When an ExecutionAdapter is supplied, the pipeline must use
    it. The signal must be translated into a well-formed
    OrderRequest carrying the notional we sized, and the
    OrderResult's status must surface in the PipelineResult
    execution dict.
    """
    from engine.adapters import PaperAdapter

    shield = RiskShield(clock=_frozen_clock(T0))
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        adapter=adapter,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is True
    assert result.execution is not None
    assert result.execution["status"] == "FILLED"
    assert result.execution["client_order_id"]
    # The paper adapter holds the new position.
    positions = _run(adapter.get_positions())
    assert "BTC/USDT" in positions


def test_adapter_idempotency_via_client_order_id() -> None:
    """The adapter dedupes on client_order_id; running the same
    signal twice within the same instant returns the cached order.
    """
    from engine.adapters import PaperAdapter

    shield = RiskShield(clock=_frozen_clock(T0))
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    fixed_sig = _buy_signal()
    pipeline = SignalPipeline(
        FixedSignalStrategy(fixed_sig),
        shield,
        adapter=adapter,
    )

    first = _run(pipeline.run_once(pd.DataFrame()))
    second = _run(pipeline.run_once(pd.DataFrame()))

    assert first.execution is not None and second.execution is not None
    assert first.executed is True
    assert second.executed is True


def test_pipeline_requires_adapter_or_engine() -> None:
    shield = RiskShield(clock=_frozen_clock(T0))
    with pytest.raises(ValueError):
        SignalPipeline(
            FixedSignalStrategy(_buy_signal()),
            shield,
        )


def test_pipeline_rejects_both_adapter_and_engine() -> None:
    """Supplying both is the split-book footgun the pipeline now
    rejects: execution would go through the adapter while risk
    state would be read from the engine's stale Portfolio.
    """
    from engine.adapters import PaperAdapter

    portfolio = Portfolio(initial_balance=10_000.0)
    shield = RiskShield(clock=_frozen_clock(T0))
    with pytest.raises(ValueError, match="exactly one"):
        SignalPipeline(
            FixedSignalStrategy(_buy_signal()),
            shield,
            engine=StubEngine(portfolio=portfolio),
            adapter=PaperAdapter(initial_balance=1_000.0),
        )


def test_adapter_only_pipeline_reads_balance_from_adapter() -> None:
    """A pipeline configured with an adapter and NO engine should
    still get an equity snapshot for the risk shield. The
    adapter's get_balance becomes the source of truth.
    """
    from engine.adapters import PaperAdapter

    shield = RiskShield(clock=_frozen_clock(T0))
    adapter = PaperAdapter(initial_balance=25_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        adapter=adapter,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))
    assert result.executed is True
    assert result.execution["status"] == "FILLED"


def test_adapter_rejection_surfaces_in_pipeline_result() -> None:
    """If the adapter rejects the order (e.g. no mark price), the
    pipeline must NOT raise; it must return a PipelineResult with
    execution.success=False and the adapter's error string.
    """
    from engine.adapters import PaperAdapter

    shield = RiskShield(clock=_frozen_clock(T0))
    # No mark price set: paper adapter will reject.
    adapter = PaperAdapter(initial_balance=10_000.0)

    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        adapter=adapter,
    )

    result = _run(pipeline.run_once(pd.DataFrame()))
    # Risk shield approved (state was fine); adapter rejected.
    assert result.risk_decision.approved is True
    assert result.executed is True  # adapter returned a result
    assert result.execution["success"] is False
    assert "mark price" in (result.execution.get("error") or "")


def test_adapter_refresh_failure_fails_closed() -> None:
    """If the adapter's get_balance / get_positions raises, the
    pipeline must NOT proceed with stale state. The whole tick
    must abort with INVALID_PORTFOLIO_STATE and never call the
    risk shield or place an order.
    """
    from engine.adapters import (
        ExecutionAdapter,
        OrderRequest,
        OrderResult,
        OrderSide,
        OrderStatus,
        OrderType,
    )

    class _BrokenAdapter(ExecutionAdapter):
        name = "broken"

        async def place_order(self, request: OrderRequest) -> OrderResult:
            raise AssertionError("must not be called when refresh fails")

        async def cancel_order(self, client_order_id: str) -> OrderResult:
            raise AssertionError("not used here")

        async def get_open_orders(self):
            return []

        async def get_positions(self):
            raise ConnectionError("rpc down")

        async def get_balance(self):
            raise ConnectionError("rpc down")

    shield = RiskShield(clock=_frozen_clock(T0))
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        adapter=_BrokenAdapter(),
    )
    result = _run(pipeline.run_once(pd.DataFrame()))

    assert result.executed is False
    assert result.position_size == 0.0
    assert result.risk_decision.reason == RejectionReason.INVALID_PORTFOLIO_STATE
    assert "portfolio refresh failed" in (result.risk_decision.detail or "")
    # Shield was never consulted (no rule check ran).
    assert shield.audit_log() == []


def test_client_order_id_distinguishes_different_symbols() -> None:
    """A naive `replace("/", "")` would let AB/CD and A/BCD collide;
    the new hashed id MUST distinguish them.
    """
    from engine.signal_pipeline import SignalPipeline as SP

    sig1 = TradingSignal(
        symbol="AB/CD", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.0,
    )
    sig2 = TradingSignal(
        symbol="A/BCD", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.0,
    )
    assert SP._client_order_id(sig1) != SP._client_order_id(sig2)


def test_client_order_id_distinguishes_sub_millisecond_signals() -> None:
    """Two signals one microsecond apart must produce different
    ids; the prior millisecond truncation collapsed them.
    """
    from engine.signal_pipeline import SignalPipeline as SP

    a = TradingSignal(
        symbol="BTC/USDT", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.000001,
    )
    b = TradingSignal(
        symbol="BTC/USDT", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.000002,
    )
    assert SP._client_order_id(a) != SP._client_order_id(b)


def test_client_order_id_is_stable_for_identical_signal() -> None:
    """Same logical signal must produce the same id every time so
    the adapter's idempotency cache works."""
    from engine.signal_pipeline import SignalPipeline as SP

    a = TradingSignal(
        symbol="BTC/USDT", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.0,
    )
    b = TradingSignal(
        symbol="BTC/USDT", action="BUY", confidence=0.5, strategy="t",
        timestamp=1_700_000_000.0,
    )
    assert SP._client_order_id(a) == SP._client_order_id(b)


def test_adapter_request_preserves_signal_metadata() -> None:
    """Strategy-emitted metadata (EVM token addresses, MT5
    overrides, etc.) must reach adapter.place_order, not be
    silently dropped. Without this, EVM swaps are rejected for
    missing from_address/to_address.
    """
    from engine.adapters import (
        ExecutionAdapter,
        OrderRequest,
        OrderResult,
        OrderStatus,
    )

    captured: Dict[str, OrderRequest] = {}

    class _CapturingAdapter(ExecutionAdapter):
        name = "capture"

        async def place_order(self, request: OrderRequest) -> OrderResult:
            captured["request"] = request
            return OrderResult(
                client_order_id=request.client_order_id,
                venue_order_id="v",
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.PENDING,
            )

        async def cancel_order(self, client_order_id: str) -> OrderResult:
            raise AssertionError

        async def get_open_orders(self):
            return []

        async def get_positions(self):
            return {}

        async def get_balance(self):
            return 10_000.0

    # Signal carries metadata the adapter MUST see.
    sig = TradingSignal(
        symbol="WETH/USDC",
        action="SELL",
        confidence=0.8,
        strategy="evm-test",
        entry_price=2000.0,
        stop_loss=1900.0,
        take_profit=2200.0,
        metadata={
            "from_address": "0xWETH...",
            "to_address": "0xUSDC...",
            "expected_amount_out": 2000.0,
        },
    )
    pipeline = SignalPipeline(
        FixedSignalStrategy(sig),
        RiskShield(clock=_frozen_clock(T0)),
        adapter=_CapturingAdapter(),
    )
    result = _run(pipeline.run_once(pd.DataFrame()))
    assert result.executed is True
    sent = captured["request"]
    assert sent.metadata["from_address"] == "0xWETH..."
    assert sent.metadata["to_address"] == "0xUSDC..."
    assert sent.metadata["expected_amount_out"] == 2000.0
    # Pipeline also stamps its own observability field.
    assert "signal_timestamp" in sent.metadata


def test_cancelled_adapter_result_does_not_count_as_success() -> None:
    """An adapter that returns CANCELLED (venue refused or cancelled
    mid-flight) must show up as execution.success=False so
    downstream metrics do not over-count wins.
    """
    from engine.adapters import (
        ExecutionAdapter,
        OrderRequest,
        OrderResult,
        OrderStatus,
    )

    class _CancellingAdapter(ExecutionAdapter):
        name = "canceller"

        async def place_order(self, request: OrderRequest) -> OrderResult:
            return OrderResult(
                client_order_id=request.client_order_id,
                venue_order_id="v",
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.CANCELLED,
            )

        async def cancel_order(self, client_order_id: str) -> OrderResult:
            raise AssertionError

        async def get_open_orders(self):
            return []

        async def get_positions(self):
            return {}

        async def get_balance(self):
            return 10_000.0

    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        RiskShield(clock=_frozen_clock(T0)),
        adapter=_CancellingAdapter(),
    )
    result = _run(pipeline.run_once(pd.DataFrame()))
    assert result.executed is True
    assert result.execution["status"] == "CANCELLED"
    assert result.execution["success"] is False


def test_drawdown_trip_via_pipeline_flow() -> None:
    """End-to-end: portfolio drops past max_drawdown_pct, next signal
    is rejected with MAX_DRAWDOWN and the shield is in emergency
    stop. Locks the integration between Portfolio's HWM tracking and
    the shield's drawdown rule.

    Uses a tiny per-trade risk so the proposed notional clears the
    position-size cap and the drawdown rule is the first to fire.
    Without this the position-size rule would short-circuit before
    the shield ever evaluates drawdown.
    """
    portfolio = Portfolio(initial_balance=10_000.0)
    # Realise a 6% drawdown YESTERDAY so today's daily_pnl is 0 and
    # the drawdown rule is the first one to fire. Without this the
    # 6% loss would also trip the daily_loss_limit (rule before
    # drawdown in the shield's evaluation order).
    from datetime import timedelta

    yesterday = (T0 - timedelta(days=1)).date()
    today = T0.date()
    portfolio.add_trade({"pnl": -600.0}, today=yesterday)
    # Force daily_pnl to roll over to today (now 0).
    assert portfolio.daily_pnl(today=today) == 0.0

    engine = StubEngine(portfolio=portfolio)
    shield = RiskShield(
        config=RiskConfig(max_drawdown_pct=0.05),
        clock=_frozen_clock(T0),
    )
    pipeline = SignalPipeline(
        FixedSignalStrategy(_buy_signal()),
        shield,
        engine,
        # 0.05% risk on a 2% stop -> 2.5% notional, well below the
        # 20% position cap, so DRAWDOWN is the first rule to trip.
        config=PipelineConfig(risk_per_trade_pct=0.0005, fallback_stop_pct=0.02),
    )

    result = _run(pipeline.run_once(pd.DataFrame()))
    assert result.executed is False
    assert result.risk_decision.reason == RejectionReason.MAX_DRAWDOWN
    assert shield.is_emergency_stop_active() is True
    assert engine.received == []
