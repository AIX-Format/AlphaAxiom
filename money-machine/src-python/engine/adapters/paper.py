"""
In-memory paper-trading adapter.

`PaperAdapter` is the production-grade replacement for the ad-hoc
stubs that `tests/test_signal_flow.py` uses. It simulates a venue
with an injected mark-price oracle: every order fills at the latest
mark for the symbol (plus the configured slippage), the balance is
tracked in-memory, and positions accumulate the same way a real
exchange would report them.

Design choices:

- Idempotent on `client_order_id`. Re-submitting the same id returns
  the original `OrderResult` without touching balance or positions.
- LIMIT orders that cannot fill at the current mark are PENDING and
  parked in `_open_orders`. The caller can simulate price movement
  by calling `set_mark_price()` and `process_open_orders()` to
  re-evaluate pending fills.
- Commissions and slippage are pluggable through the same
  `CommissionModel` and `SlippageModel` interfaces the backtest uses,
  so the paper run sees the same execution-cost model as the
  backtest that approved the strategy.
- The adapter is async-only so it lines up with the signature
  `SignalPipeline` and any future CCXT-backed adapter will share.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..backtest import CommissionModel, FixedSlippage, FlatCommission, SlippageModel
from .base import (
    AdapterError,
    ExecutionAdapter,
    Fill,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class PaperAdapter(ExecutionAdapter):
    """In-memory simulator for end-to-end testing and paper trading."""

    name = "paper"

    def __init__(
        self,
        *,
        initial_balance: float = 10_000.0,
        commission: Optional[CommissionModel] = None,
        slippage: Optional[SlippageModel] = None,
    ) -> None:
        if initial_balance < 0:
            raise AdapterError(
                f"initial_balance must be >= 0, got {initial_balance!r}"
            )
        self._balance = float(initial_balance)
        self._positions: Dict[str, float] = {}
        self._mark_prices: Dict[str, float] = {}
        self._order_history: Dict[str, OrderResult] = {}
        self._open_orders: Dict[str, OrderRequest] = {}
        self.commission = commission or FlatCommission(rate=0.0)
        self.slippage = slippage or FixedSlippage(bps=0.0)
        # Two locks, one per concurrency model:
        # - `_lock` (asyncio): protects the async `place_order` /
        #   `cancel_order` / `process_open_orders` paths against
        #   coroutine interleaving.
        # - `_sync_lock` (threading): protects the sync paths
        #   `place_order_sync` and `reset` against thread-vs-thread
        #   races. Used by short-lived hosts (RL env step, vectorised
        #   backtests) that must not pay the asyncio.run overhead
        #   per call.
        # Mixing async and sync entry points on the SAME adapter
        # instance is not supported; the two locks do not coordinate
        # with each other. Pick one concurrency model per adapter.
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Test/dev helpers
    # ------------------------------------------------------------------

    def set_mark_price(self, symbol: str, price: float) -> None:
        """
        Update the mark price used for fills for the given symbol.
        
        Converts `price` to float and validates it is finite and greater than zero before storing it.
        
        Parameters:
            symbol (str): Symbol whose mark price will be set.
            price (float): New mark price (will be converted to float).
        
        Raises:
            AdapterError: If `price` is not convertible to float, is not finite, or is not greater than zero.
        """
        try:
            value = float(price)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"mark price must be a number, got {price!r}") from exc
        import math

        if not math.isfinite(value) or value <= 0:
            raise AdapterError(
                f"mark price must be a finite positive number, got {price!r}"
            )
        self._mark_prices[symbol] = value

    def reset(self, *, initial_balance: Optional[float] = None) -> None:
        """Wipe order history, positions, and balance back to a
        clean state. Mark prices and the configured cost models
        are preserved.

        Used by short-lived simulations (RL training, walk-forward
        sweeps) that need a fresh adapter at the top of every
        episode without paying the cost of re-instantiation. The
        public method replaces the previous pattern of poking at
        the underscore-prefixed attributes from outside, which
        broke encapsulation and made the adapter fragile to
        bookkeeping changes.

        Thread safety: serialised against other sync callers via
        `_sync_lock`. NOT coordinated with the async `_lock`; do
        not call `reset()` while async `place_order` /
        `cancel_order` / `process_open_orders` calls are in
        flight on the same adapter instance.

        Args:
            initial_balance: Optional new starting cash balance
                (must be >= 0). When None, the existing balance is
                preserved while positions and history are cleared.

        Raises:
            AdapterError: If `initial_balance` is provided and
                negative.
        """
        if initial_balance is not None:
            try:
                candidate = float(initial_balance)
            except (TypeError, ValueError) as exc:
                raise AdapterError(
                    f"initial_balance must be numeric, got {initial_balance!r}"
                ) from exc
            import math

            if not math.isfinite(candidate) or candidate < 0:
                raise AdapterError(
                    "initial_balance must be a finite non-negative "
                    f"number, got {initial_balance!r}"
                )
        with self._sync_lock:
            if initial_balance is not None:
                self._balance = float(initial_balance)
            self._positions.clear()
            self._order_history.clear()
            self._open_orders.clear()

    def place_order_sync(self, request: OrderRequest) -> OrderResult:
        """Synchronous market-order entry point for non-async hosts.

        Provides the same fill logic as `place_order` without the
        asyncio overhead. Designed for tight loops (RL env step,
        vectorised backtests) that would otherwise spin up a fresh
        event loop on every call via `asyncio.run`.

        Thread safety: serialised against other sync callers via
        `_sync_lock`, so two threads invoking this method on the
        same adapter cannot tear balance / position state. The
        `_locked`-suffixed helpers below assume the caller holds
        SOME serialising mechanism; `_sync_lock` plays that role
        for this entry point, mirroring what `_lock` does for the
        async `place_order` path.

        NOT coordinated with the async `_lock`. Do not invoke
        `place_order_sync` and the async `place_order` /
        `cancel_order` / `process_open_orders` concurrently on the
        same adapter instance; pick one concurrency model per
        adapter.

        Limitations: LIMIT orders that would park as PENDING are
        rejected here (PENDING management requires the async path
        and `process_open_orders` to clear).
        """
        with self._sync_lock:
            cached = self._order_history.get(request.client_order_id)
            if cached is not None:
                return cached

            error = self._validate_request(request)
            if error is not None:
                result = self._reject(request, error)
                self._order_history[request.client_order_id] = result
                return result

            mark = self._mark_prices.get(request.symbol)
            if mark is None:
                result = self._reject(
                    request, f"no mark price set for {request.symbol}"
                )
                self._order_history[request.client_order_id] = result
                return result

            if request.order_type is OrderType.MARKET:
                result = self._fill_market_locked(request, mark)
            else:
                if self._limit_can_fill(request, mark):
                    result = self._fill_limit_locked(request, mark)
                else:
                    result = self._reject(
                        request,
                        "place_order_sync rejects unfillable LIMIT orders; "
                        "use the async place_order path for PENDING management",
                    )

            self._order_history[request.client_order_id] = result
            return result

    async def process_open_orders(self) -> List[OrderResult]:
        """
        Re-evaluate all pending LIMIT orders against current mark prices and return those that were filled.
        
        Pending LIMIT orders whose conditions are still not met remain open; orders without a known mark price are skipped.
        
        Returns:
            newly_filled (List[OrderResult]): List of OrderResult objects for orders that were filled during this call.
        """
        newly_filled: List[OrderResult] = []
        async with self._lock:
            for client_id in list(self._open_orders.keys()):
                request = self._open_orders[client_id]
                mark = self._mark_prices.get(request.symbol)
                if mark is None:
                    continue
                if not self._limit_can_fill(request, mark):
                    continue
                result = self._fill_limit_locked(request, mark)
                self._order_history[client_id] = result
                self._open_orders.pop(client_id, None)
                newly_filled.append(result)
        return newly_filled

    # ------------------------------------------------------------------
    # ExecutionAdapter API
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        async with self._lock:
            # Idempotency: same client_order_id returns the prior result.
            cached = self._order_history.get(request.client_order_id)
            if cached is not None:
                logger.debug(
                    "place_order idempotent hit for %s", request.client_order_id
                )
                return cached

            error = self._validate_request(request)
            if error is not None:
                result = self._reject(request, error)
                self._order_history[request.client_order_id] = result
                return result

            mark = self._mark_prices.get(request.symbol)
            if mark is None:
                result = self._reject(
                    request, f"no mark price set for {request.symbol}"
                )
                self._order_history[request.client_order_id] = result
                return result

            if request.order_type is OrderType.MARKET:
                result = self._fill_market_locked(request, mark)
            else:
                if self._limit_can_fill(request, mark):
                    result = self._fill_limit_locked(request, mark)
                else:
                    result = self._park_pending_locked(request)
                    self._open_orders[request.client_order_id] = request

            self._order_history[request.client_order_id] = result
            return result

    async def cancel_order(self, client_order_id: str) -> OrderResult:
        async with self._lock:
            existing = self._order_history.get(client_order_id)
            if existing is None:
                # Cancelling an unknown order: return a synthetic
                # rejection so the caller has a uniform response.
                return OrderResult(
                    client_order_id=client_order_id,
                    venue_order_id=None,
                    symbol="",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    error="unknown client_order_id",
                )
            if existing.status is OrderStatus.CANCELLED:
                return existing
            if existing.is_done:
                # Already filled or rejected: cancellation is a no-op
                # but we return the current state so the caller knows.
                return existing
            cancelled = OrderResult(
                client_order_id=existing.client_order_id,
                venue_order_id=existing.venue_order_id,
                symbol=existing.symbol,
                side=existing.side,
                order_type=existing.order_type,
                status=OrderStatus.CANCELLED,
                requested_quantity=existing.requested_quantity,
                filled_quantity=existing.filled_quantity,
                average_fill_price=existing.average_fill_price,
                fills=list(existing.fills),
            )
            self._order_history[client_order_id] = cancelled
            self._open_orders.pop(client_order_id, None)
            return cancelled

    async def get_open_orders(self) -> List[OrderResult]:
        async with self._lock:
            return [
                self._order_history[cid]
                for cid in self._open_orders.keys()
                if cid in self._order_history
            ]

    async def get_positions(self) -> Dict[str, float]:
        async with self._lock:
            return {
                sym: qty for sym, qty in self._positions.items() if qty != 0.0
            }

    async def get_balance(self) -> float:
        async with self._lock:
            return self._balance

    # ------------------------------------------------------------------
    # Fill / reject helpers (must be called with the lock held)
    # ------------------------------------------------------------------

    def _fill_market_locked(
        self, request: OrderRequest, mark: float
    ) -> OrderResult:
        side_str = "buy" if request.side is OrderSide.BUY else "sell"
        fill_price = self.slippage.apply(mark, side_str)
        return self._record_fill_locked(request, fill_price)

    def _fill_limit_locked(
        self, request: OrderRequest, mark: float
    ) -> OrderResult:
        # Limit orders fill at the better of the limit price and the
        # current mark, then have slippage applied on the worse side.
        # The fill price is then clamped back to the limit so a noisy
        # slippage model cannot push a BUY above its limit (or a SELL
        # below): a limit order is a strict price cap by definition.
        limit = float(request.limit_price)
        if request.side is OrderSide.BUY:
            base_price = min(limit, mark)
        else:
            base_price = max(limit, mark)
        side_str = "buy" if request.side is OrderSide.BUY else "sell"
        slipped = self.slippage.apply(base_price, side_str)
        if request.side is OrderSide.BUY:
            fill_price = min(slipped, limit)
        else:
            fill_price = max(slipped, limit)
        return self._record_fill_locked(request, fill_price)

    def _record_fill_locked(
        self, request: OrderRequest, fill_price: float
    ) -> OrderResult:
        qty = self._resolve_quantity(request, fill_price)
        if qty <= 0:
            return self._reject(request, "resolved quantity is non-positive")

        notional = qty * fill_price
        commission = float(self.commission.apply(notional))

        # Long buys and short sells: balance debited by notional + fee.
        # Long sells and short buys: balance credited by notional - fee.
        if request.side is OrderSide.BUY:
            cash_change = -(notional + commission)
            position_change = qty
            # Spot-style guardrail: a paper BUY that cannot be funded
            # is rejected the same way a real exchange would reject
            # it. Without this the balance would silently go
            # negative and downstream risk calculations would lie.
            required = notional + commission
            if required > self._balance:
                return self._reject(
                    request,
                    f"insufficient balance: need {required:.2f}, have {self._balance:.2f}",
                )
        else:
            cash_change = notional - commission
            position_change = -qty

        self._balance += cash_change
        prior = self._positions.get(request.symbol, 0.0)
        self._positions[request.symbol] = prior + position_change

        fill = Fill(
            price=fill_price,
            quantity=qty,
            timestamp=datetime.now(tz=timezone.utc),
            commission=commission,
        )
        return OrderResult(
            client_order_id=request.client_order_id,
            venue_order_id=f"paper-{uuid.uuid4()}",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.FILLED,
            requested_quantity=qty,
            filled_quantity=qty,
            average_fill_price=fill_price,
            fills=[fill],
            metadata={"strategy": request.strategy, "cash_change": cash_change},
        )

    def _park_pending_locked(self, request: OrderRequest) -> OrderResult:
        qty = self._resolve_quantity(
            request, float(request.limit_price) if request.limit_price else 0.0
        )
        return OrderResult(
            client_order_id=request.client_order_id,
            venue_order_id=f"paper-{uuid.uuid4()}",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.PENDING,
            requested_quantity=qty if qty > 0 else None,
            filled_quantity=0.0,
            average_fill_price=None,
            metadata={"strategy": request.strategy},
        )

    def _reject(self, request: OrderRequest, error: str) -> OrderResult:
        logger.info(
            "PaperAdapter rejected %s %s: %s",
            request.symbol,
            request.side.value if isinstance(request.side, OrderSide) else request.side,
            error,
        )
        return OrderResult(
            client_order_id=request.client_order_id,
            venue_order_id=None,
            symbol=request.symbol,
            side=request.side if isinstance(request.side, OrderSide) else OrderSide.BUY,
            order_type=request.order_type if isinstance(request.order_type, OrderType) else OrderType.MARKET,
            status=OrderStatus.REJECTED,
            error=error,
            metadata={"strategy": request.strategy},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_quantity(request: OrderRequest, price: float) -> float:
        if request.quantity is not None:
            return float(request.quantity)
        if request.notional is not None and price > 0:
            return float(request.notional) / float(price)
        return 0.0

    @staticmethod
    def _limit_can_fill(request: OrderRequest, mark: float) -> bool:
        if request.limit_price is None:
            return False
        if request.side is OrderSide.BUY:
            # Buy limit fills when mark drops to or below the limit.
            return mark <= float(request.limit_price)
        # Sell limit fills when mark rises to or above the limit.
        return mark >= float(request.limit_price)
