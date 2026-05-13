"""
Execution adapters.

Every venue (paper trading, CCXT-backed exchange, MT5 via Cloudflare
Worker, EVM DEX, Solana DEX) ships as a concrete subclass of
`ExecutionAdapter`. Upstream code (`SignalPipeline`, the dashboard,
backtest live mode) talks to the adapter through that interface and
never imports a venue-specific module directly. The pipeline tests
in `tests/test_signal_flow.py` already use this shape via a stub;
this package gives those stubs a real home and adds a production-
quality paper-trading adapter that can drop into the same slot.
"""

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
from .mt5 import (
    HttpResponse,
    MT5Adapter,
    MT5Config,
    SigningFunction,
    canonical_payload,
    inline_signer,
    keychain_signer,
)
from .paper import PaperAdapter

__all__ = [
    "AdapterError",
    "ExecutionAdapter",
    "Fill",
    "HttpResponse",
    "MT5Adapter",
    "MT5Config",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperAdapter",
    "SigningFunction",
    "canonical_payload",
    "inline_signer",
    "keychain_signer",
]
