"""
Trading strategies package.

Each module under `engine.strategies` exports a single concrete
subclass of `Strategy` (see `base.py`). Strategies are intentionally
pure: they take a price DataFrame in, return a `TradingSignal` out,
and never touch the network, the portfolio, or the risk shield.
Composition with risk and execution happens upstream in
`engine.trading_core`.
"""

from .base import Strategy, TradingSignal
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

__all__ = [
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "Strategy",
    "TradingSignal",
]
