"""
Gymnasium-compatible trading environment.

`TradingEnv` walks bar-by-bar through a historical OHLCV DataFrame
and lets a policy take one of three actions: HOLD, BUY, or SELL.
Each action is translated into an order against an injected
`PaperAdapter` (which already has the full execution semantics:
slippage, commission, idempotency, balance bookkeeping). The
observation is a window of recent OHLCV + a small set of
indicators + position state.

The env is intentionally simple by design:

- Single-asset, single-position long-only-by-default. Going short
  is supported when `allow_short=True`.
- Discrete action space `{0: HOLD, 1: BUY, 2: SELL}`. Continuous
  position sizing is a follow-up.
- One step = one bar. Fills happen at the next bar's open (no
  look-ahead), with the slippage/commission models defined on the
  adapter.
- `done` fires at end-of-data or when equity falls below
  `min_equity_fraction * initial_equity`.

The whole module imports only `gymnasium`, `numpy`, and `pandas`.
No torch, no stable_baselines3 - those land in the agent slice.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "engine.rl.env requires gymnasium. Install with `pip install gymnasium`."
    ) from exc

from ..adapters import OrderRequest, OrderSide, OrderStatus, OrderType, PaperAdapter
from ..indicators import atr, ema, rsi
from .reward import PnLReward, RewardFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationConfig:
    """Shape of the observation vector.

    The observation is a 1-D array assembled from:
      - `window_size` rolling closes, normalised by the latest close
      - 3 indicators on the latest bar: RSI(14), EMA(20)/close,
        ATR(14)/close
      - 2 position-state scalars: signed position (units of base),
        equity normalised by `initial_equity`

    Total dimension = window_size + 5. Float32 throughout.
    """

    window_size: int = 64
    rsi_period: int = 14
    ema_period: int = 20
    atr_period: int = 14


@dataclass
class EnvConfig:
    """Runtime tunables for the env."""

    symbol: str = "BTC/USDT"
    initial_equity: float = 10_000.0
    # Fraction of equity deployed per BUY/SELL action.
    position_fraction: float = 0.1
    # Stop the episode when equity drops below this fraction of
    # `initial_equity`. Acts as the env's risk-off circuit breaker.
    min_equity_fraction: float = 0.5
    allow_short: bool = False
    # Index of the first bar the policy gets to act on. Must be >=
    # ObservationConfig.window_size so the obs window is full.
    warmup_bars: int = 64
    # Seed for env determinism.
    seed: Optional[int] = None
    observation: ObservationConfig = field(default_factory=ObservationConfig)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


# Discrete action ids.
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


class TradingEnv(gym.Env):
    """Single-asset bar-stepped trading env.

    Constructor:
      data:    OHLCV DataFrame with columns
               ['open', 'high', 'low', 'close', 'volume'] indexed
               by timestamp.
      adapter: PaperAdapter (or any ExecutionAdapter that supports
               set_mark_price + place_order).
      config:  EnvConfig.
      reward:  RewardFunction. Defaults to `PnLReward(as_return=True)`.

    Spaces:
      action_space:      Discrete(3)
      observation_space: Box(window_size + 5,), float32, range [-10, 10]

    The Box bounds are intentionally wide because the observation
    is a normalised mix of magnitudes; clipping inside the env
    would mask agent edge cases.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        adapter: PaperAdapter,
        config: Optional[EnvConfig] = None,
        reward: Optional[RewardFunction] = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self._validate_data(data)
        self._data = data.reset_index(drop=False)
        self._index_col = self._data.columns[0]
        self.adapter = adapter
        self.reward_fn: RewardFunction = reward or PnLReward(as_return=True)

        if self.config.warmup_bars < self.config.observation.window_size:
            raise ValueError(
                "warmup_bars must be >= observation.window_size "
                f"({self.config.observation.window_size}); "
                f"got {self.config.warmup_bars}"
            )
        if len(self._data) <= self.config.warmup_bars + 1:
            raise ValueError(
                f"data has only {len(self._data)} rows; need at least "
                f"{self.config.warmup_bars + 2} for one step after warm-up"
            )

        self.action_space = spaces.Discrete(3)
        obs_dim = self.config.observation.window_size + 5
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Mutable episode state. reset() resets everything.
        self._current_bar: int = self.config.warmup_bars
        self._initial_equity: float = float(self.config.initial_equity)
        self._equity: float = self._initial_equity
        self._peak_equity: float = self._initial_equity
        self._position: float = 0.0
        self._cumulative_steps: int = 0
        self._step_id: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._current_bar = self.config.warmup_bars
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._position = 0.0
        self._cumulative_steps = 0
        self._step_id = 0

        # Reset the paper adapter to a clean balance + mark.
        self.adapter._balance = self._initial_equity  # type: ignore[attr-defined]
        self.adapter._positions.clear()  # type: ignore[attr-defined]
        self.adapter._order_history.clear()  # type: ignore[attr-defined]
        self.adapter._open_orders.clear()  # type: ignore[attr-defined]
        self._set_mark(self._current_bar)

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "step": 0,
            "equity": self._equity,
            "position": self._position,
            "bar_index": self._current_bar,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not 0 <= int(action) <= 2:
            raise ValueError(f"action must be in {{0, 1, 2}}, got {action!r}")

        prev_equity = self._equity
        prev_position = self._position
        prev_peak = self._peak_equity

        # Advance to the bar where the order would fill (next bar's
        # open). The adapter is given the next-bar open as the mark
        # so its market-order semantics fill there.
        next_bar = min(self._current_bar + 1, len(self._data) - 1)
        next_open = float(self._data["open"].iloc[next_bar])
        self.adapter.set_mark_price(self.config.symbol, next_open)

        # Translate the discrete action into an OrderRequest.
        order_notional = self._equity * self.config.position_fraction
        fill = self._apply_action(int(action), notional=order_notional, mark=next_open)

        # Advance the env clock and mark-to-market at the new bar's
        # close so the reward sees a coherent equity transition.
        self._current_bar = next_bar
        close_price = float(self._data["close"].iloc[self._current_bar])
        # `adapter.get_balance()` is async; we know PaperAdapter's
        # internal _balance is sync and consistent, so read it
        # directly. A more general adapter wrapper is a follow-up.
        self._equity = float(self.adapter._balance) + self._position_value(  # type: ignore[attr-defined]
            close_price
        )
        new_peak = max(prev_peak, self._equity)
        self._peak_equity = new_peak

        step_return = (
            (self._equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        )

        info: Dict[str, Any] = {
            "step": self._cumulative_steps + 1,
            "equity": self._equity,
            "position": self._position,
            "bar_index": self._current_bar,
            "prev_peak_equity": prev_peak,
            "new_peak_equity": new_peak,
            "step_return": step_return,
            "fill": fill,
        }
        reward = float(
            self.reward_fn(
                prev_equity=prev_equity,
                new_equity=self._equity,
                prev_position=prev_position,
                new_position=self._position,
                info=info,
            )
        )

        # Episode termination conditions.
        terminated = self._equity < self._initial_equity * self.config.min_equity_fraction
        truncated = self._current_bar >= len(self._data) - 1

        self._cumulative_steps += 1
        self._step_id += 1

        obs = self._build_observation()
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self) -> None:  # pragma: no cover - cosmetic
        logger.info(
            "bar=%d equity=%.2f position=%.6f peak=%.2f",
            self._current_bar, self._equity, self._position, self._peak_equity,
        )

    def close(self) -> None:  # pragma: no cover
        return None

    # ------------------------------------------------------------------
    # Observation + action helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        cfg = self.config.observation
        end = self._current_bar + 1
        start = max(0, end - cfg.window_size)
        closes = self._data["close"].iloc[start:end].to_numpy(dtype=np.float32)
        # Right-pad with the leftmost value if we are early in the
        # series (should not happen after warmup_bars but defensive).
        if len(closes) < cfg.window_size:
            pad = np.full(cfg.window_size - len(closes), closes[0], dtype=np.float32)
            closes = np.concatenate([pad, closes])
        latest = closes[-1] if closes[-1] != 0 else 1.0
        normalised = closes / latest - 1.0  # mean 0, scale ~ 0.01-0.05

        # Indicators on the full historical slice up to current bar.
        hist = self._data.iloc[: end]
        rsi_value = self._safe_last(rsi(hist["close"], period=cfg.rsi_period), 50.0) / 100.0
        ema_ratio = (
            self._safe_last(ema(hist["close"], period=cfg.ema_period), latest) / latest
            - 1.0
        )
        atr_ratio = (
            self._safe_last(atr(hist["high"], hist["low"], hist["close"], period=cfg.atr_period), 0.0)
            / latest
        )

        position_norm = float(self._position)
        equity_norm = (self._equity / self._initial_equity) - 1.0

        obs = np.concatenate(
            [
                normalised.astype(np.float32),
                np.array(
                    [rsi_value, ema_ratio, atr_ratio, position_norm, equity_norm],
                    dtype=np.float32,
                ),
            ]
        )
        # Clip to the declared Box bounds so a one-off numerical
        # outlier (e.g. a stale indicator NaN coerced through
        # _safe_last) cannot violate the observation_space contract.
        return np.clip(obs, -10.0, 10.0)

    @staticmethod
    def _safe_last(series: pd.Series, fallback: float) -> float:
        try:
            value = float(series.iloc[-1])
        except (IndexError, KeyError, ValueError, TypeError):
            return fallback
        if not np.isfinite(value):
            return fallback
        return value

    def _apply_action(
        self, action: int, *, notional: float, mark: float
    ) -> Dict[str, Any]:
        """Translate the discrete action into a paper order.

        Returns a small dict with the fill summary so the reward
        function can inspect it via `info["fill"]`.
        """
        if action == ACTION_HOLD or notional <= 0 or mark <= 0:
            return {"status": "HOLD"}

        if action == ACTION_BUY:
            side = OrderSide.BUY
        else:  # ACTION_SELL
            if self._position <= 0 and not self.config.allow_short:
                # Cannot open a short when shorts are disabled.
                return {"status": "BLOCKED_SHORT_DISABLED"}
            side = OrderSide.SELL

        request = OrderRequest(
            client_order_id=f"rl-{self._step_id}-{uuid.uuid4().hex[:6]}",
            symbol=self.config.symbol,
            side=side,
            order_type=OrderType.MARKET,
            notional=float(notional),
        )

        # PaperAdapter rejects oversized BUY if balance is short.
        # `get_balance` is async but we can use a synchronous shim:
        # the test suite uses asyncio.run; here we drive the call
        # via the adapter's internal sync state for performance.
        # A cleaner async wrapper is the natural follow-up.
        import asyncio

        result = asyncio.run(self.adapter.place_order(request))
        if result.status == OrderStatus.FILLED:
            qty = float(result.filled_quantity or 0.0)
            if side is OrderSide.BUY:
                self._position += qty
            else:
                self._position -= qty
            return {
                "status": "FILLED",
                "fill_price": float(result.average_fill_price or mark),
                "quantity": qty,
            }
        return {"status": result.status.value, "error": result.error}

    def _position_value(self, price: float) -> float:
        """Mark-to-market value of the current position at `price`."""
        return float(self._position) * float(price)

    def _set_mark(self, bar_idx: int) -> None:
        price = float(self._data["close"].iloc[bar_idx])
        if price > 0:
            self.adapter.set_mark_price(self.config.symbol, price)

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        if data is None or data.empty:
            raise ValueError("TradingEnv requires a non-empty OHLCV DataFrame")
        missing = [c for c in ("open", "high", "low", "close", "volume") if c not in data.columns]
        if missing:
            raise ValueError(f"missing OHLCV columns: {missing}")
