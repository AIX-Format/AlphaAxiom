"""
Execution adapters.

Every venue (paper trading, CCXT-backed exchange, MT5 via Cloudflare
Worker, EVM DEX, Solana DEX) ships as a concrete subclass of
`ExecutionAdapter`. Upstream code (`SignalPipeline`, the dashboard,
backtest live mode) talks to the adapter through that interface and
never imports a venue-specific module directly.

Symbols are exported lazily via `__getattr__` so importing a single
venue does not pull in every other venue's transitive dependencies.
In particular, `PaperAdapter` transitively imports `engine.backtest`
and pandas/numpy; an MT5-only or EVM-only runtime that does not need
those heavy deps can `from engine.adapters import MT5Adapter` without
paying for them.
"""

from __future__ import annotations

from typing import Any

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

__all__ = [
    "AdapterError",
    "EVMAdapter",
    "EVMAdapterConfig",
    "EVMChainConfig",
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
    "TxReceipt",
    "canonical_payload",
    "inline_signer",
    "keychain_signer",
]


# Map of lazily-imported names to their submodule. Each tuple is
# (module_path_under_engine.adapters, attribute_name_in_submodule).
_LAZY_EXPORTS = {
    "MT5Adapter":        ("mt5",   "MT5Adapter"),
    "MT5Config":         ("mt5",   "MT5Config"),
    "HttpResponse":      ("mt5",   "HttpResponse"),
    "SigningFunction":   ("mt5",   "SigningFunction"),
    "canonical_payload": ("mt5",   "canonical_payload"),
    "inline_signer":     ("mt5",   "inline_signer"),
    "keychain_signer":   ("mt5",   "keychain_signer"),
    "PaperAdapter":      ("paper", "PaperAdapter"),
    "EVMAdapter":        ("evm",   "EVMAdapter"),
    "EVMAdapterConfig":  ("evm",   "EVMAdapterConfig"),
    "EVMChainConfig":    ("evm",   "EVMChainConfig"),
    "TxReceipt":         ("evm",   "TxReceipt"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute lookup for adapter exports.

    Importing only the symbols you actually use avoids dragging in
    transitive dependencies (pandas/numpy via PaperAdapter,
    eth-account via EVMAdapter, cryptography via MT5Adapter signer)
    when a runtime does not need them.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(f".{module_name}", package=__name__)
    value = getattr(module, attr_name)
    # Cache on the package so the next access hits a normal attribute
    # lookup instead of __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
