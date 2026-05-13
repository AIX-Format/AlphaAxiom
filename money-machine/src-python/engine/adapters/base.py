"""
Execution adapter abstract base.

`ExecutionAdapter` is the seam between the trading engine and any
order-routing venue: paper trading, CCXT-backed centralized exchanges,
MT5 via a Cloudflare Worker, EVM DEX aggregators, Solana DEX, etc.

Every adapter exposes the same async surface:

    await adapter.place_order(OrderRequest(...))   -> OrderResult
    await adapter.cancel_order(client_order_id)    -> OrderResult
    await adapter.get_open_orders()                -> list[OrderResult]
    await adapter.get_positions()                  -> dict[symbol -> qty]
    await adapter.get_balance()                    -> float

Adapters are idempotent on `client_order_id`: if a caller retries an
order with the same client id (typical after a network blip), the
adapter MUST return the original `OrderResult` instead of submitting a
duplicate order. This is the only safe pattern for retrying network
requests against money-moving APIs.

Adapters never raise on routine failures. Network errors, venue
rejections, and invalid input all surface as an `OrderResult` with
`status=OrderStatus.REJECTED` and an `error` string. Only truly
exceptional events (programmer error, the adapter not initialised,
hardware faults) should raise `AdapterError`.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """Raised for exceptional, non-recoverable adapter failures.

    Routine failures (venue rejection, insufficient balance, network
    blip, malformed input) surface through `OrderResult.error`, not as
    an exception. Reserve this for programmer error or hardware faults
    where the caller has no useful recovery path.
    """


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"     # accepted by venue, not yet filled
    FILLED = "FILLED"       # fully filled
    PARTIAL = "PARTIAL"     # partially filled, still resting
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"   # never reached the venue or venue refused


@dataclass(frozen=True)
class OrderRequest:
    """A single order to place.

    `client_order_id` is the caller's idempotency key. The same value
    on two calls must yield the same `OrderResult`. The adapter MUST
    NOT submit two orders for one client id.

    `notional` is the amount in quote currency the caller wants to
    deploy. Adapters convert to `quantity` using the fill price they
    observe; both fields end up populated on the `OrderResult` for
    auditability.
    """

    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    notional: Optional[float] = None        # in quote currency
    quantity: Optional[float] = None        # in base currency
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fill:
    """A single execution against an order."""

    price: float
    quantity: float
    timestamp: datetime
    commission: float = 0.0


@dataclass
class OrderResult:
    """Outcome of placing or modifying an order.

    `status` is the canonical signal: `FILLED`/`PARTIAL` means the
    venue acted, `PENDING` means accepted-but-resting,
    `REJECTED`/`CANCELLED` means no exposure. `fills` is the list of
    individual executions (most adapters produce one for MARKET orders
    and one or more for LIMIT). `error` is human-readable and never
    None when status is REJECTED.
    """

    client_order_id: str
    venue_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    requested_quantity: Optional[float] = None
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fills: List[Fill] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    @property
    def is_done(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExecutionAdapter(abc.ABC):
    """Abstract base for every order-routing venue."""

    #: Short stable identifier. Surfaced in logs and dashboards.
    name: str = "abstract-adapter"

    @abc.abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order. MUST be idempotent on `client_order_id`."""

    @abc.abstractmethod
    async def cancel_order(self, client_order_id: str) -> OrderResult:
        """Cancel a resting order. Idempotent: re-cancelling a
        cancelled order returns the same CANCELLED result.
        """

    @abc.abstractmethod
    async def get_open_orders(self) -> List[OrderResult]:
        """All orders that are PENDING or PARTIAL."""

    @abc.abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Current position size per symbol, in base-currency units.
        Long positions are positive, shorts negative. Flat symbols
        may be absent or zero.
        """

    @abc.abstractmethod
    async def get_balance(self) -> float:
        """Free quote-currency balance available for new orders."""

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: OrderRequest) -> Optional[str]:
        """Return a human-readable error or None if the request is sane.

        Centralised so every concrete adapter applies the same first-
        line sanity checks before talking to the venue: a non-empty
        client_order_id, a known side and type, a positive quantity or
        notional, finite numbers in the price fields.
        """
        if not request.client_order_id:
            return "client_order_id is required"
        if not isinstance(request.side, OrderSide):
            return f"side must be an OrderSide enum, got {type(request.side).__name__}"
        if not isinstance(request.order_type, OrderType):
            return f"order_type must be an OrderType enum, got {type(request.order_type).__name__}"

        size_given = sum(
            1
            for v in (request.quantity, request.notional)
            if v is not None
        )
        if size_given == 0:
            return "either quantity or notional must be set"
        if request.quantity is not None and not _is_positive(request.quantity):
            return f"quantity must be a positive finite number, got {request.quantity!r}"
        if request.notional is not None and not _is_positive(request.notional):
            return f"notional must be a positive finite number, got {request.notional!r}"

        if request.order_type is OrderType.LIMIT:
            if request.limit_price is None or not _is_positive(request.limit_price):
                return "LIMIT orders require a positive limit_price"

        for name, value in (
            ("stop_loss", request.stop_loss),
            ("take_profit", request.take_profit),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                return f"{name} must be a positive finite number when set, got {value!r}"

        return None


def _is_positive(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0
