"""
Gymnasium-compatible trading environment.

`TradingEnv` walks bar-by-bar through a historical OHLCV DataFrame
and lets a policy take one of three actions: HOLD, BUY, or SELL.
Each action is translated into an order against an injected
`PaperAdapter` (which already has the full execution semantics:
slippage, commission, idempotency, balance bookkeeping). The
observation is a window of recent OHLCV + a small set of
indicators + position state, normalised so it stays inside the
declared Box bounds for any reasonable asset.

Design notes the reviews surfaced and fixed:

- Indicators are computed ONCE in `__init__` and indexed during
  `step`. The previous version recomputed RSI/EMA/ATR over the
  full historical slice on every step, giving the environment
  O(T^2) cost for an episode of length T.
- The fill path uses `PaperAdapter.place_order_sync` so there is
  no `asyncio.run` overhead per step (and the env runs cleanly
  inside an existing event loop, e.g. a Jupyter cell or an RLlib
  worker).
- `PaperAdapter.reset()` replaces the prior pattern of poking at
  the adapter's underscore attributes, keeping the env honest
  about the adapter contract.
- Position is observed as a scale-invariant weight
  `(position * mark) / equity` so the same env works for BTC and
  SHIB without busting the Box bounds.
- A SELL with `allow_short=False` is capped to the current long's
  notional so it cannot flip the position into a short.

The whole module imports only `gymnasium`, `numpy`, and `pandas`.
No torch, no stable_baselines3 - those land in the agent slice.
"""

from __future__ import annotations

import logging
import math
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
      - 3 indicators on the latest bar: RSI(14)/100, EMA(20)/close-1,
        ATR(14)/close
      - 2 position-state scalars: position weight (notional / equity),
        equity / initial_equity - 1

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
    """Single-asset bar-stepped trading env."""

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

        # Precompute the static indicator series once. Indexing
        # them during step keeps each tick O(1) instead of
        # recomputing over the full historical slice (O(T) per step
        # which was O(T^2) over a full episode).
        self._indicator_cache = self._precompute_indicators()

        # Numpy views for the hot path in step / observation
        # assembly. These never change after construction.
        self._open_arr = self._data["open"].to_numpy(dtype=np.float64)
        self._close_arr = self._data["close"].to_numpy(dtype=np.float64)
        self._n_bars = len(self._data)

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
        # Honour `EnvConfig.seed` when the caller does not pass an
        # explicit seed. Most RL training loops set the seed once
        # at construction time via the config and then call
        # `env.reset()` without arguments; without this fallback
        # the config knob is silently ignored.
        effective_seed = self.config.seed if seed is None else seed
        super().reset(seed=effective_seed)
        self._current_bar = self.config.warmup_bars
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._position = 0.0
        self._cumulative_steps = 0
        self._step_id = 0

        # Use the adapter's public reset hook instead of reaching
        # into its underscore-prefixed state.
        self.adapter.reset(initial_balance=self._initial_equity)
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
        # Use the Gymnasium space's own `contains` predicate so
        # non-integer inputs (e.g. 1.9) are rejected instead of
        # being silently coerced to a valid action via `int()`.
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be a member of {self.action_space!r}, got {action!r}"
            )

        prev_equity = self._equity
        prev_position = self._position
        prev_peak = self._peak_equity

        # Advance to the bar where the order would fill (next bar's
        # open). The adapter is given the next-bar open as the mark
        # so its market-order semantics fill there.
        next_bar = min(self._current_bar + 1, self._n_bars - 1)
        next_open = float(self._open_arr[next_bar])
        self.adapter.set_mark_price(self.config.symbol, next_open)

        # Translate the discrete action into an OrderRequest.
        order_notional = self._equity * self.config.position_fraction
        fill = self._apply_action(int(action), notional=order_notional, mark=next_open)

        # Advance the env clock and mark-to-market at the new bar's
        # close so the reward sees a coherent equity transition.
        self._current_bar = next_bar
        close_price = float(self._close_arr[self._current_bar])
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
        truncated = self._current_bar >= self._n_bars - 1

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

    def _precompute_indicators(self) -> Dict[str, np.ndarray]:
        """Run RSI/EMA/ATR once over the full historical series.

        Each indicator returns a Series aligned to the input index;
        we drop them straight to numpy arrays so `_build_observation`
        is a constant-time lookup.
        """
        cfg = self.config.observation
        close = self._data["close"]
        high = self._data["high"]
        low = self._data["low"]
        rsi_series = rsi(close, period=cfg.rsi_period)
        ema_series = ema(close, period=cfg.ema_period)
        atr_series = atr(high, low, close, period=cfg.atr_period)
        return {
            "rsi": rsi_series.to_numpy(dtype=np.float64),
            "ema": ema_series.to_numpy(dtype=np.float64),
            "atr": atr_series.to_numpy(dtype=np.float64),
        }

    def _build_observation(self) -> np.ndarray:
        cfg = self.config.observation
        end = self._current_bar + 1
        start = max(0, end - cfg.window_size)
        closes = self._close_arr[start:end].astype(np.float32)
        if len(closes) < cfg.window_size:
            pad = np.full(cfg.window_size - len(closes), closes[0], dtype=np.float32)
            closes = np.concatenate([pad, closes])
        latest = float(closes[-1]) if closes[-1] != 0 else 1.0
        normalised = closes / latest - 1.0

        bar = self._current_bar
        rsi_v = self._safe_indicator(self._indicator_cache["rsi"], bar, fallback=50.0) / 100.0
        ema_v = self._safe_indicator(self._indicator_cache["ema"], bar, fallback=latest)
        atr_v = self._safe_indicator(self._indicator_cache["atr"], bar, fallback=0.0)
        ema_ratio = (ema_v / latest) - 1.0 if latest > 0 else 0.0
        atr_ratio = atr_v / latest if latest > 0 else 0.0

        # Position is observed as a SCALE-INVARIANT weight:
        # notional in equity-currency / equity. For BTC at 50k with
        # a 0.05 long on a 10k account the weight is 0.25; for SHIB
        # at 1e-5 with 5e7 units on the same account the weight is
        # also 0.5. Raw position quantity would clip immediately
        # for assets with extreme unit sizes.
        if self._equity > 0:
            position_weight = (self._position * latest) / self._equity
        else:
            position_weight = 0.0
        equity_norm = (self._equity / self._initial_equity) - 1.0

        obs = np.concatenate(
            [
                normalised.astype(np.float32),
                np.array(
                    [rsi_v, ema_ratio, atr_ratio, position_weight, equity_norm],
                    dtype=np.float32,
                ),
            ]
        )
        return np.clip(obs, -10.0, 10.0)

    @staticmethod
    def _safe_indicator(arr: np.ndarray, bar: int, *, fallback: float) -> float:
        if 0 <= bar < arr.shape[0]:
            value = arr[bar]
            if not math.isnan(value) and not math.isinf(value):
                return float(value)
        return fallback

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
            if not self.config.allow_short:
                if self._position <= 0:
                    # Cannot open a short when shorts are disabled.
                    return {"status": "BLOCKED_SHORT_DISABLED"}
                # Cap notional to the current long's value so the
                # SELL closes (or partially closes) the position
                # without flipping it into a short.
                max_close_notional = self._position * mark
                if max_close_notional <= 0:
                    return {"status": "BLOCKED_SHORT_DISABLED"}
                notional = min(notional, max_close_notional)
            side = OrderSide.SELL

        request = OrderRequest(
            client_order_id=f"rl-{self._step_id}-{uuid.uuid4().hex[:6]}",
            symbol=self.config.symbol,
            side=side,
            order_type=OrderType.MARKET,
            notional=float(notional),
        )

        # Synchronous fill path: no asyncio.run, no event-loop
        # creation, and works inside an already-running loop (RLlib,
        # Jupyter, asyncio test harnesses).
        result = self.adapter.place_order_sync(request)
        if result.status == OrderStatus.FILLED:
            qty = float(result.filled_quantity or 0.0)
            if side is OrderSide.BUY:
                self._position += qty
            else:
                self._position -= qty
                # Post-fill clamp: with slippage, a SELL whose
                # notional was pre-capped to `position * mark` can
                # still produce a quantity slightly larger than
                # `self._position` (the fill price ends up below
                # `mark`, so `qty = notional / fill_price >
                # position`). Without this clamp, `allow_short=
                # False` could be silently violated. We clip the
                # position back to zero and surface the deviation
                # in the fill info so reward functions can see it.
                if not self.config.allow_short and self._position < 0:
                    overshoot = -self._position
                    self._position = 0.0
                    return {
                        "status": "FILLED",
                        "fill_price": float(result.average_fill_price or mark),
                        "quantity": qty,
                        "short_overshoot_clamped": overshoot,
                    }
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
        price = float(self._close_arr[bar_idx])
        if price > 0:
            self.adapter.set_mark_price(self.config.symbol, price)

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        if data is None or data.empty:
            raise ValueError("TradingEnv requires a non-empty OHLCV DataFrame")
        missing = [c for c in ("open", "high", "low", "close", "volume") if c not in data.columns]
        if missing:
            raise ValueError(f"missing OHLCV columns: {missing}")
