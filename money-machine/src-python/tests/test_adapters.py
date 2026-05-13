"""
Tests for the execution adapter abstraction and the paper adapter.

The paper adapter is the production-grade reference implementation
for the `ExecutionAdapter` contract. Locking down its behaviour
gives us a baseline every future adapter (MT5, CCXT, EVM, Solana)
can be tested against.

Coverage:

  - OrderRequest / OrderResult / Fill dataclass invariants and the
    central `_validate_request` helper on the base class.
  - PaperAdapter happy path: market BUY and SELL fills update
    balance and positions correctly.
  - Idempotency: same client_order_id returns the original result
    without re-debiting balance or growing the position.
  - Limit orders: park as PENDING when not fillable, then fill on
    `process_open_orders` after `set_mark_price` clears them.
  - Cancellation: cancels a pending order, no-ops on already-done
    orders, and synthesises a rejection on unknown ids.
  - Cost model integration: commission and slippage actually affect
    the recorded fill and the resulting balance.
  - Rejection paths: missing mark, invalid request, non-positive
    quantity.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.adapters import (  # noqa: E402
    AdapterError,
    ExecutionAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperAdapter,
)
from engine.backtest import FixedSlippage, FlatCommission  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _market_buy(
    client_id: str,
    *,
    symbol: str = "BTC/USDT",
    notional: Optional[float] = None,
    quantity: Optional[float] = None,
) -> OrderRequest:
    # Default to notional=1000 only when both are unset, so callers
    # that supply `quantity=` get exactly one size field set (and
    # pass the validator's XOR check).
    if notional is None and quantity is None:
        notional = 1_000.0
    return OrderRequest(
        client_order_id=client_id,
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=notional,
        quantity=quantity,
    )


def _market_sell(
    client_id: str,
    *,
    symbol: str = "BTC/USDT",
    quantity: float = 0.05,
) -> OrderRequest:
    return OrderRequest(
        client_order_id=client_id,
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=quantity,
    )


# ---------------------------------------------------------------------------
# Request validation (base class helper)
# ---------------------------------------------------------------------------


def test_validate_request_accepts_well_formed_market_order() -> None:
    req = _market_buy("abc")
    assert ExecutionAdapter._validate_request(req) is None


def test_validate_request_rejects_empty_symbol() -> None:
    req = OrderRequest(
        client_order_id="x",
        symbol="",  # empty
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "symbol" in err


def test_validate_request_rejects_whitespace_symbol() -> None:
    req = OrderRequest(
        client_order_id="x",
        symbol="   ",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "symbol" in err


def test_validate_request_rejects_non_string_symbol() -> None:
    """Non-string symbol must not raise AttributeError inside
    `.strip()`; reject with the standard error string instead.
    """
    req = OrderRequest(
        client_order_id="x",
        symbol=12345,  # type: ignore[arg-type]
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "symbol" in err


def test_validate_request_rejects_non_string_client_order_id() -> None:
    req = OrderRequest(
        client_order_id=42,  # type: ignore[arg-type]
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "client_order_id" in err


def test_validate_request_rejects_bool_as_sizing() -> None:
    """bool is a subclass of int in Python; True passes isinstance
    and yields float(True) == 1.0, which would silently sneak a
    1.0-unit order through validation. _is_positive now rejects.
    """
    req = OrderRequest(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=True,  # type: ignore[arg-type]
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None
    assert "quantity" in err


def test_validate_request_rejects_both_quantity_and_notional() -> None:
    """Setting BOTH creates ambiguous semantics; callers must pick one."""
    req = OrderRequest(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
        quantity=0.02,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "exactly one" in err


@pytest.mark.parametrize(
    "kwargs",
    [
        {"client_order_id": ""},
        {"notional": -10.0, "quantity": None},
        {"notional": 0.0, "quantity": None},
        {"notional": float("nan"), "quantity": None},
        {"notional": None, "quantity": None},
        {"notional": None, "quantity": -1.0},
        {"notional": None, "quantity": float("inf")},
    ],
)
def test_validate_request_rejects_bad_market_orders(kwargs) -> None:
    base = dict(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
        quantity=None,
    )
    base.update(kwargs)
    req = OrderRequest(**base)
    assert ExecutionAdapter._validate_request(req) is not None


def test_validate_request_requires_limit_price_for_limit_orders() -> None:
    req = OrderRequest(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        notional=1_000.0,
        limit_price=None,
    )
    err = ExecutionAdapter._validate_request(req)
    assert err is not None and "LIMIT" in err


def test_validate_request_rejects_invalid_stop_or_take() -> None:
    base = dict(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=1_000.0,
    )
    err = ExecutionAdapter._validate_request(
        OrderRequest(stop_loss=-1.0, **base)
    )
    assert err is not None
    err = ExecutionAdapter._validate_request(
        OrderRequest(take_profit=float("nan"), **base)
    )
    assert err is not None


# ---------------------------------------------------------------------------
# OrderResult invariants
# ---------------------------------------------------------------------------


def test_order_result_open_and_done_flags() -> None:
    pending = OrderResult(
        client_order_id="a",
        venue_order_id="v",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
    )
    filled = OrderResult(
        client_order_id="b",
        venue_order_id="v",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
    )
    assert pending.is_open and not pending.is_done
    assert filled.is_done and not filled.is_open


# ---------------------------------------------------------------------------
# PaperAdapter: market orders happy path
# ---------------------------------------------------------------------------


def test_paper_adapter_market_buy_updates_balance_and_positions() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        result = await adapter.place_order(_market_buy("o1", notional=5_000.0))

        assert result.status is OrderStatus.FILLED
        assert result.symbol == "BTC/USDT"
        # 5000 / 50000 = 0.1 BTC
        assert result.filled_quantity == pytest.approx(0.1)
        assert result.average_fill_price == pytest.approx(50_000.0)

        # Balance dropped by 5000, position increased by 0.1.
        assert (await adapter.get_balance()) == pytest.approx(5_000.0)
        positions = await adapter.get_positions()
        assert positions == {"BTC/USDT": pytest.approx(0.1)}

    _run(scenario())


def test_paper_adapter_market_sell_closes_long_position() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        await adapter.place_order(_market_buy("o1", notional=5_000.0))
        result = await adapter.place_order(_market_sell("o2", quantity=0.1))

        assert result.status is OrderStatus.FILLED
        # Sold 0.1 at 50k -> +5000 cash, position back to 0.
        assert (await adapter.get_balance()) == pytest.approx(10_000.0)
        assert (await adapter.get_positions()) == {}

    _run(scenario())


def test_paper_adapter_requires_mark_price() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        result = await adapter.place_order(_market_buy("o1"))
        assert result.status is OrderStatus.REJECTED
        assert "mark price" in (result.error or "")
        # Balance untouched.
        assert (await adapter.get_balance()) == pytest.approx(10_000.0)

    _run(scenario())


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_paper_adapter_is_idempotent_on_client_order_id() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        first = await adapter.place_order(_market_buy("o1", notional=5_000.0))
        # Re-submit the same client_order_id; must return the same result
        # without touching balance or positions again.
        second = await adapter.place_order(_market_buy("o1", notional=5_000.0))

        assert first is second or first == second
        assert (await adapter.get_balance()) == pytest.approx(5_000.0)
        assert (await adapter.get_positions()) == {"BTC/USDT": pytest.approx(0.1)}

    _run(scenario())


# ---------------------------------------------------------------------------
# Limit orders, pending fills, cancellation
# ---------------------------------------------------------------------------


def test_limit_buy_parks_pending_then_fills_after_mark_drop() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        # Buy limit at 48000 with mark at 50000: should park PENDING.
        req = OrderRequest(
            client_order_id="lim1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            notional=4_800.0,
            limit_price=48_000.0,
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.PENDING
        open_orders = await adapter.get_open_orders()
        assert len(open_orders) == 1
        # Balance untouched while pending.
        assert (await adapter.get_balance()) == pytest.approx(10_000.0)

        # Mark drops to 47500 (below the limit). Process should fill it.
        adapter.set_mark_price("BTC/USDT", 47_500.0)
        filled = await adapter.process_open_orders()
        assert len(filled) == 1
        assert filled[0].status is OrderStatus.FILLED
        # Buy limit fills at min(limit, mark) = 47500.
        assert filled[0].average_fill_price == pytest.approx(47_500.0)
        # Open orders is now empty.
        assert (await adapter.get_open_orders()) == []

    _run(scenario())


def test_cancel_pending_limit_order_returns_cancelled() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)
        req = OrderRequest(
            client_order_id="lim1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            notional=1_000.0,
            limit_price=40_000.0,
        )
        await adapter.place_order(req)

        cancelled = await adapter.cancel_order("lim1")
        assert cancelled.status is OrderStatus.CANCELLED
        assert (await adapter.get_open_orders()) == []

        # Cancelling again is a no-op that returns the same CANCELLED.
        again = await adapter.cancel_order("lim1")
        assert again.status is OrderStatus.CANCELLED

    _run(scenario())


def test_cancel_unknown_order_id_returns_rejection() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        result = await adapter.cancel_order("does-not-exist")
        assert result.status is OrderStatus.REJECTED
        assert "unknown" in (result.error or "").lower()

    _run(scenario())


def test_cancel_already_filled_order_is_noop() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)
        await adapter.place_order(_market_buy("o1", notional=1_000.0))

        result = await adapter.cancel_order("o1")
        # The original FILLED state is preserved; no CANCELLED override.
        assert result.status is OrderStatus.FILLED

    _run(scenario())


# ---------------------------------------------------------------------------
# Cost model integration
# ---------------------------------------------------------------------------


def test_paper_adapter_charges_commission_on_fill() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(
            initial_balance=10_000.0,
            commission=FlatCommission(rate=0.001),  # 10 bps
            slippage=FixedSlippage(bps=0.0),
        )
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        await adapter.place_order(_market_buy("o1", notional=5_000.0))

        # Notional 5000 + commission 5 = 5005 debit.
        assert (await adapter.get_balance()) == pytest.approx(4_995.0)

    _run(scenario())


def test_paper_adapter_applies_slippage_to_fill_price() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(
            initial_balance=10_000.0,
            commission=FlatCommission(rate=0.0),
            slippage=FixedSlippage(bps=50.0),  # 50 bps = 0.5%
        )
        adapter.set_mark_price("BTC/USDT", 100.0)

        result = await adapter.place_order(_market_buy("o1", quantity=10.0))
        # BUY slips up by 0.5%: fill at 100.5
        assert result.average_fill_price == pytest.approx(100.5)
        # Balance debited by 10 * 100.5 = 1005
        assert (await adapter.get_balance()) == pytest.approx(8_995.0)

    _run(scenario())


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_invalid_request_rejected_without_balance_change() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        bad = OrderRequest(
            client_order_id="bad",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=None,
            quantity=None,  # neither size given
        )
        result = await adapter.place_order(bad)
        assert result.status is OrderStatus.REJECTED
        assert "either quantity or notional" in (result.error or "")
        assert (await adapter.get_balance()) == pytest.approx(10_000.0)

    _run(scenario())


def test_negative_initial_balance_raises() -> None:
    with pytest.raises(AdapterError):
        PaperAdapter(initial_balance=-1.0)


def test_set_mark_price_rejects_non_positive() -> None:
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(AdapterError):
        adapter.set_mark_price("BTC/USDT", 0.0)
    with pytest.raises(AdapterError):
        adapter.set_mark_price("BTC/USDT", -5.0)


def test_set_mark_price_rejects_nan_and_inf() -> None:
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(AdapterError):
        adapter.set_mark_price("BTC/USDT", float("nan"))
    with pytest.raises(AdapterError):
        adapter.set_mark_price("BTC/USDT", float("inf"))
    with pytest.raises(AdapterError):
        adapter.set_mark_price("BTC/USDT", float("-inf"))


def test_paper_adapter_rejects_buy_when_insufficient_balance() -> None:
    """A spot-style BUY that cannot be funded must be rejected the
    same way a real exchange rejects it. Otherwise the balance
    silently goes negative and downstream risk math lies.
    """
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=100.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)
        result = await adapter.place_order(
            _market_buy("oversize", notional=10_000.0)
        )
        assert result.status is OrderStatus.REJECTED
        assert "insufficient balance" in (result.error or "")
        # Balance untouched.
        assert (await adapter.get_balance()) == pytest.approx(100.0)
        # No position opened.
        assert (await adapter.get_positions()) == {}

    _run(scenario())


def test_limit_buy_fill_clamped_to_limit_price_after_slippage() -> None:
    """A BUY limit fill must never go above the limit price, even when
    the slippage model would push it higher. The limit is a strict
    cap by definition.
    """
    async def scenario() -> None:
        adapter = PaperAdapter(
            initial_balance=10_000.0,
            commission=FlatCommission(rate=0.0),
            slippage=FixedSlippage(bps=100.0),  # 1% upward push on buys
        )
        adapter.set_mark_price("BTC/USDT", 100.0)
        # Limit BUY at 100; mark is 100 so the limit can fill. The
        # slippage model would push the fill to 101, but the clamp
        # holds it at 100.
        req = OrderRequest(
            client_order_id="lim",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=100.0,
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.FILLED
        assert result.average_fill_price == pytest.approx(100.0)

    _run(scenario())


def test_limit_sell_fill_clamped_to_limit_price_after_slippage() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(
            initial_balance=10_000.0,
            commission=FlatCommission(rate=0.0),
            slippage=FixedSlippage(bps=100.0),
        )
        adapter.set_mark_price("BTC/USDT", 100.0)
        # Seed a position to sell.
        adapter._positions["BTC/USDT"] = 1.0  # type: ignore[attr-defined]
        req = OrderRequest(
            client_order_id="lim-s",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=100.0,
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.FILLED
        # Slippage would push the sell fill DOWN to 99; clamp keeps it at 100.
        assert result.average_fill_price == pytest.approx(100.0)

    _run(scenario())


# ---------------------------------------------------------------------------
# PaperAdapter.reset() — new method added in this PR
# ---------------------------------------------------------------------------


def test_reset_without_balance_arg_preserves_balance() -> None:
    """reset() without `initial_balance` must clear positions and orders
    but leave the balance at whatever it was before the call.
    """
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)
        await adapter.place_order(_market_buy("o1", notional=3_000.0))
        balance_after_buy = await adapter.get_balance()
        assert balance_after_buy < 10_000.0

        adapter.reset()  # no initial_balance kwarg

        # Balance unchanged, positions and order history cleared.
        assert (await adapter.get_balance()) == pytest.approx(balance_after_buy)
        assert (await adapter.get_positions()) == {}
        assert (await adapter.get_open_orders()) == []

    _run(scenario())


def test_reset_with_zero_initial_balance_is_valid() -> None:
    """Zero is a valid initial_balance; the adapter should start
    with an empty account and reject any BUY order immediately.
    """
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.reset(initial_balance=0.0)
    assert adapter._balance == pytest.approx(0.0)  # type: ignore[attr-defined]


def test_reset_with_negative_initial_balance_raises() -> None:
    """A negative initial_balance must raise AdapterError — the same
    guard that the constructor applies.
    """
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(AdapterError):
        adapter.reset(initial_balance=-1.0)


def test_reset_preserves_mark_prices_and_cost_models() -> None:
    """Mark prices and the injected commission/slippage models must
    survive reset() so the adapter is immediately usable without
    reconfiguring the mark oracle.
    """
    adapter = PaperAdapter(
        initial_balance=5_000.0,
        commission=FlatCommission(rate=0.001),
        slippage=FixedSlippage(bps=10.0),
    )
    adapter.set_mark_price("BTC/USDT", 50_000.0)
    adapter.set_mark_price("ETH/USDT", 3_000.0)

    adapter.reset(initial_balance=10_000.0)

    # Both marks still accessible after reset.
    assert adapter._mark_prices.get("BTC/USDT") == pytest.approx(50_000.0)  # type: ignore[attr-defined]
    assert adapter._mark_prices.get("ETH/USDT") == pytest.approx(3_000.0)  # type: ignore[attr-defined]
    # Cost models preserved too.
    assert adapter.commission is not None
    assert adapter.slippage is not None


def test_reset_clears_pending_limit_orders() -> None:
    """Any pending (open) limit orders must be purged by reset().
    The adapter must not process them after a reset; they belong
    to the previous episode's context.
    """
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        # Park a limit order that cannot fill at current mark.
        req = OrderRequest(
            client_order_id="lim-ep1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            notional=1_000.0,
            limit_price=40_000.0,  # below current mark -> PENDING
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.PENDING
        assert len(await adapter.get_open_orders()) == 1

        adapter.reset(initial_balance=10_000.0)

        # Open orders list must be empty after reset.
        assert (await adapter.get_open_orders()) == []

    _run(scenario())


def test_reset_clears_order_history_so_id_can_be_reused() -> None:
    """After reset(), a previously-used client_order_id must no
    longer be in the history. The idempotency cache is episode-
    scoped; old ids must not bleed into the next episode.

    Passing `initial_balance` to reset() restarts equity from
    that value, so each post-reset BUY costs against the fresh
    balance, not the carry-over from the first fill.
    """
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 50_000.0)

        first = await adapter.place_order(_market_buy("reused-id", notional=1_000.0))
        assert first.status is OrderStatus.FILLED

        adapter.reset(initial_balance=10_000.0)

        # Re-submit the same id; it must fill again (history is clear).
        second = await adapter.place_order(_market_buy("reused-id", notional=1_000.0))
        assert second.status is OrderStatus.FILLED
        # Balance after reset = 10_000; one fill of 1_000 = 9_000.
        balance = await adapter.get_balance()
        assert balance == pytest.approx(9_000.0)

    _run(scenario())


# ---------------------------------------------------------------------------
# PaperAdapter.place_order_sync() — new method added in this PR
# ---------------------------------------------------------------------------


def test_place_order_sync_market_buy_fills_and_updates_balance() -> None:
    """Basic synchronous BUY: balance decremented, position updated."""
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    result = adapter.place_order_sync(_market_buy("s1", notional=5_000.0))

    assert result.status is OrderStatus.FILLED
    assert result.filled_quantity == pytest.approx(0.1)
    assert result.average_fill_price == pytest.approx(50_000.0)
    # Balance debited synchronously.
    assert adapter._balance == pytest.approx(5_000.0)  # type: ignore[attr-defined]
    assert adapter._positions.get("BTC/USDT") == pytest.approx(0.1)  # type: ignore[attr-defined]


def test_place_order_sync_market_sell_fills_and_credits_balance() -> None:
    """SELL fills synchronously and credits the balance."""
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)
    # Seed a long position directly.
    adapter._positions["BTC/USDT"] = 0.2  # type: ignore[attr-defined]

    result = adapter.place_order_sync(_market_sell("s2", quantity=0.1))

    assert result.status is OrderStatus.FILLED
    # Cash credited by 0.1 * 50000 = 5000.
    assert adapter._balance == pytest.approx(15_000.0)  # type: ignore[attr-defined]
    assert adapter._positions.get("BTC/USDT") == pytest.approx(0.1)  # type: ignore[attr-defined]


def test_place_order_sync_is_idempotent_on_client_order_id() -> None:
    """Submitting the same client_order_id twice must return the
    cached result without re-filling or double-debiting.
    """
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    first = adapter.place_order_sync(_market_buy("idem-sync", notional=1_000.0))
    second = adapter.place_order_sync(_market_buy("idem-sync", notional=1_000.0))

    assert first.status is OrderStatus.FILLED
    assert second.status is OrderStatus.FILLED
    # Same result object (or at least same content).
    assert first.client_order_id == second.client_order_id
    # Balance debited only once.
    assert adapter._balance == pytest.approx(9_000.0)  # type: ignore[attr-defined]


def test_place_order_sync_rejects_when_no_mark_price() -> None:
    """Without a mark price the sync path must reject gracefully."""
    adapter = PaperAdapter(initial_balance=10_000.0)
    # Deliberately do NOT call set_mark_price.

    result = adapter.place_order_sync(_market_buy("no-mark", notional=1_000.0))

    assert result.status is OrderStatus.REJECTED
    assert "mark price" in (result.error or "")
    assert adapter._balance == pytest.approx(10_000.0)  # type: ignore[attr-defined]


def test_place_order_sync_rejects_invalid_request() -> None:
    """An order with no sizing (neither quantity nor notional) must
    be rejected and stored in order history.
    """
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    bad = OrderRequest(
        client_order_id="bad-sync",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=None,
        quantity=None,
    )
    result = adapter.place_order_sync(bad)

    assert result.status is OrderStatus.REJECTED
    assert adapter._balance == pytest.approx(10_000.0)  # type: ignore[attr-defined]


def test_place_order_sync_rejects_insufficient_balance() -> None:
    """A BUY whose notional exceeds the available balance must be
    rejected without mutating balance or positions.
    """
    adapter = PaperAdapter(initial_balance=100.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    result = adapter.place_order_sync(_market_buy("oversize-sync", notional=10_000.0))

    assert result.status is OrderStatus.REJECTED
    assert "insufficient balance" in (result.error or "")
    assert adapter._balance == pytest.approx(100.0)  # type: ignore[attr-defined]
    assert adapter._positions == {}  # type: ignore[attr-defined]


def test_place_order_sync_fills_fillable_limit_order() -> None:
    """A LIMIT BUY whose limit_price >= mark should fill immediately
    via the sync path (it does not need to park PENDING).
    """
    adapter = PaperAdapter(
        initial_balance=10_000.0,
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
    )
    adapter.set_mark_price("BTC/USDT", 100.0)

    # BUY limit at 110 with mark at 100: mark <= limit -> can fill.
    req = OrderRequest(
        client_order_id="lim-sync-fill",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        limit_price=110.0,
    )
    result = adapter.place_order_sync(req)

    assert result.status is OrderStatus.FILLED
    # Fills at min(limit, mark) = 100.
    assert result.average_fill_price == pytest.approx(100.0)


def test_place_order_sync_rejects_unfillable_limit_order() -> None:
    """An unfillable LIMIT order must be REJECTED (not PENDING),
    since the sync path cannot manage PENDING state. The caller
    must use the async place_order path for PENDING management.
    """
    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    # BUY limit at 40000 with mark at 50000: mark > limit -> unfillable.
    req = OrderRequest(
        client_order_id="lim-sync-nofill",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        notional=1_000.0,
        limit_price=40_000.0,
    )
    result = adapter.place_order_sync(req)

    assert result.status is OrderStatus.REJECTED
    assert "place_order_sync" in (result.error or "") or "async" in (result.error or "")
    # Balance unchanged.
    assert adapter._balance == pytest.approx(10_000.0)  # type: ignore[attr-defined]
    # Must NOT appear in open orders (which is async-only).
    assert adapter._open_orders == {}  # type: ignore[attr-defined]


def test_place_order_sync_works_inside_running_event_loop() -> None:
    """place_order_sync must not call asyncio.run internally; it
    must be callable from within an already-running event loop
    without raising RuntimeError.
    """
    import asyncio

    adapter = PaperAdapter(initial_balance=10_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)

    async def drive() -> OrderResult:
        return adapter.place_order_sync(_market_buy("loop-sync", notional=1_000.0))

    result = asyncio.run(drive())
    assert result.status is OrderStatus.FILLED
