"""
Strategy base class and the canonical TradingSignal dataclass.

A strategy is anything that turns a window of OHLCV data into a
`TradingSignal`. Subclasses implement `generate_signal(df)` and
override `name` to a stable identifier used in logs and audit trails.

The contract is intentionally narrow:

- Input is a pandas DataFrame with at least the columns `open`,
  `high`, `low`, `close`, `volume`, in chronological order, indexed
  by a UTC DatetimeIndex.
- Output is always a `TradingSignal`. Strategies must never raise on
  short or noisy data; they should return a HOLD signal with
  confidence `0.0` and a reason string instead. The risk shield and
  the position sizer downstream rely on always receiving a valid
  object.
- Strategies are stateless across calls. Any state (e.g. previous
  signal, indicator caches) lives on `self` and is reset by
  re-instantiating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

# Required OHLCV columns. Strategies should fail fast if any of these
# are missing.
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class TradingSignal:
    """A single trading decision produced by a strategy.

    The shape is a superset of what the existing AI signal generator
    in `engine/signal_generator.py` returns, so downstream consumers
    (risk shield, position sizer, executor) only ever need to handle
    one type.
    """

    symbol: str
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float  # 0.0 to 1.0
    strategy: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    amount: Optional[float] = None
    reasoning: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    VALID_ACTIONS = ("BUY", "SELL", "HOLD")

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = datetime.now(tz=timezone.utc).timestamp()
        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {self.VALID_ACTIONS}, got {self.action!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )

    def is_actionable(self) -> bool:
        """True if this signal would result in an order being placed.

        HOLD signals and BUY/SELL with zero confidence are filtered
        out by the risk shield before reaching the executor.
        """
        return self.action != "HOLD" and self.confidence > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    #: Stable identifier surfaced in audit logs and the dashboard.
    #: Concrete subclasses must override this.
    name: str = "abstract-strategy"

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Produce a signal from the given OHLCV window.

        Implementations should never raise. On insufficient or
        malformed data, return a HOLD signal with confidence 0.0 and
        a reason string explaining why.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all subclasses
    # ------------------------------------------------------------------

    def _hold(self, reason: str, *, price: Optional[float] = None) -> TradingSignal:
        return TradingSignal(
            symbol=self.symbol,
            action="HOLD",
            confidence=0.0,
            strategy=self.name,
            entry_price=price,
            reasoning=reason,
        )

    def _validate_frame(self, df: pd.DataFrame, min_rows: int) -> Optional[str]:
        """Return a human-readable error if the frame is unusable, else None."""
        if df is None or df.empty:
            return "empty dataframe"
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return f"missing columns: {missing}"
        if len(df) < min_rows:
            return f"insufficient rows: have {len(df)}, need >= {min_rows}"
        return None
