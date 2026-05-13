"""Typed I/O contracts between signal, risk, and execution stages.

These models are the canonical wire contracts passed between the
SignalPipeline and execution adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .adapters.base import ExecutionAdapter, OrderRequest, OrderSide, OrderType
from .risk_shield import RiskDecision
from .strategies.base import TradingSignal


@dataclass(frozen=True)
class SignalContract:
    symbol: str
    action: str
    confidence: float
    strategy: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trading_signal(cls, signal: TradingSignal) -> "SignalContract":
        return cls(**signal.to_dict())

    def validate(self) -> None:
        TradingSignal(**self.__dict__)


@dataclass(frozen=True)
class RiskDecisionContract:
    approved: bool
    reason: Optional[str] = None
    detail: str = ""

    @classmethod
    def from_risk_decision(cls, decision: RiskDecision) -> "RiskDecisionContract":
        reason = decision.reason.value if decision.reason is not None else None
        return cls(approved=decision.approved, reason=reason, detail=decision.detail)

    def validate(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be bool")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason must be str when set")


@dataclass(frozen=True)
class OrderIntentContract:
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    notional: Optional[float] = None
    quantity: Optional[float] = None
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_order_request(self) -> OrderRequest:
        req = OrderRequest(**self.__dict__)
        err = ExecutionAdapter._validate_request(req)
        if err is not None:
            raise ValueError(err)
        return req


@dataclass(frozen=True)
class SignedOrderContract:
    client_order_id: str
    payload: bytes
    signature: bytes
    key_id: str

    def validate(self) -> None:
        if not self.client_order_id.strip():
            raise ValueError("client_order_id is required")
        if not self.payload:
            raise ValueError("payload is required")
        if not self.signature:
            raise ValueError("signature is required")
        if not self.key_id.strip():
            raise ValueError("key_id is required")


class ContractValidator:
    """Centralized validation before pipeline stage transitions."""

    @staticmethod
    def signal(signal: TradingSignal) -> SignalContract:
        typed = SignalContract.from_trading_signal(signal)
        typed.validate()
        return typed

    @staticmethod
    def risk(decision: RiskDecision) -> RiskDecisionContract:
        typed = RiskDecisionContract.from_risk_decision(decision)
        typed.validate()
        return typed

    @staticmethod
    def order_intent(intent: OrderIntentContract) -> OrderRequest:
        return intent.to_order_request()
