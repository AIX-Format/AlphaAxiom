"""
Reward functions for the trading RL environment.

A reward function maps one `(prev_state, action, new_state)`
transition to a scalar. The env wires the chosen reward into
`TradingEnv.step` and never imports `torch` / `numpy` beyond the
basic math we already have, so the unit tests stay fast and
hardware-free.

Three built-ins cover the most common combinations:

- `PnLReward`: mark-to-market PnL change in absolute or
  return-normalised units. Honest, sparse, the easiest baseline.
- `DrawdownPenaltyReward`: penalises every step that pushes the
  running peak-to-trough deeper. Pushes the agent away from
  high-variance bets even when they look profitable on average.
- `TurnoverPenaltyReward`: subtracts a fixed cost per side switch
  so the agent learns commission discipline.
- `SharpeRatioReward`: rolling Sharpe of the last N step returns,
  scaled. Smooth, dense, hard to game.

`composite(...)` stacks them: any caller can mix and match with
explicit per-component weights, which is the right design for an
RL training loop where reward shaping is the main lever.

Conventions:
  - State is whatever the env stores; the protocol only requires
    a few duck-typed attributes (`equity`, `peak_equity`).
  - Reward is `float`. NaN/inf are coerced to 0.0 so a numerical
    spike cannot brick training.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Protocol


class RewardFunction(Protocol):
    """Map one env step into a scalar reward.

    `prev_equity` and `new_equity` are the agent's portfolio
    value before and after the step's fills. `prev_position`
    and `new_position` are signed quantities (long > 0, short
    < 0). `info` is a free-form dict the env populates with
    extra context (current ATR, drawdown, etc.).
    """

    def __call__(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        ...


def _safe_float(value: float, default: float = 0.0) -> float:
    """Coerce non-finite reward components to a safe default."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


# ---------------------------------------------------------------------------
# Built-in reward components
# ---------------------------------------------------------------------------


@dataclass
class PnLReward:
    """Mark-to-market PnL change per step.

    `as_return=True` divides by `prev_equity` so the magnitude
    stays in a stable range (typically [-0.05, 0.05] per bar) and
    is independent of the absolute capital base; this is what most
    RL training loops want.
    """

    as_return: bool = True
    scale: float = 1.0

    def __call__(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        diff = new_equity - prev_equity
        if self.as_return:
            if prev_equity <= 0:
                return 0.0
            diff = diff / prev_equity
        return _safe_float(self.scale * diff)


@dataclass
class DrawdownPenaltyReward:
    """Subtract `weight * additional_drawdown_fraction` per step.

    `additional_drawdown` is the increase in `(peak - equity) /
    peak` this step. We only charge for *new* drawdown so steady
    bleeding hurts more than a one-time mark-down already paid for.
    """

    weight: float = 1.0

    def __call__(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        prev_peak = float(info.get("prev_peak_equity", prev_equity))
        new_peak = float(info.get("new_peak_equity", max(prev_peak, new_equity)))
        if new_peak <= 0:
            return 0.0
        prev_dd = max(0.0, (prev_peak - prev_equity) / prev_peak) if prev_peak > 0 else 0.0
        new_dd = max(0.0, (new_peak - new_equity) / new_peak)
        delta = max(0.0, new_dd - prev_dd)
        return _safe_float(-self.weight * delta)


@dataclass
class TurnoverPenaltyReward:
    """Charge a flat penalty when the agent flips direction.

    "Flip" means crossing zero: long -> short or short -> long.
    Opening from flat (0 -> long), scaling in or out, and closing
    to flat (long -> 0) all return 0; charging those would
    discourage normal position management instead of curbing the
    long/short churn we actually care about.

    Useful when commissions are small enough that PnL alone does
    not teach the agent to stay in a trade.
    """

    penalty: float = 0.001

    def __call__(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        # A direction flip requires both sides to be non-zero with
        # opposite signs. `prev * new < 0` captures exactly that.
        if prev_position * new_position >= 0:
            return 0.0
        return _safe_float(-self.penalty)


@dataclass
class SharpeRatioReward:
    """Rolling-window Sharpe of step returns, scaled.

    A short window (~20 steps) makes the signal dense enough for
    on-policy learning. Returns are pulled from
    `info["step_return"]`, which the env populates.

    Updates run in O(1) per step via running sum + sum-of-squares
    instead of recomputing the full mean / variance over the
    window. For window=20 this is a ~10x speedup over the naive
    pass; for window=1000 (longer-horizon training) it is the
    difference between viable and unusable.
    """

    window: int = 20
    scale: float = 1.0
    _returns: Deque[float] = field(init=False)
    _running_sum: float = field(init=False, default=0.0)
    _running_sumsq: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        # Size the deque to the configured window. `init=False`
        # keeps callers from constructing a deque with the wrong
        # maxlen by accident.
        self._returns = deque(maxlen=self.window)
        self._running_sum = 0.0
        self._running_sumsq = 0.0

    def __call__(
        self,
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        step_return = float(info.get("step_return", 0.0))
        if len(self._returns) == self._returns.maxlen:
            # About to evict the oldest value; subtract it from
            # the running totals so the maths stays O(1).
            oldest = self._returns[0]
            self._running_sum -= oldest
            self._running_sumsq -= oldest * oldest
        self._returns.append(step_return)
        self._running_sum += step_return
        self._running_sumsq += step_return * step_return

        n = len(self._returns)
        if n < 2:
            return 0.0
        mean = self._running_sum / n
        # Population variance via the running totals. The
        # "sum-of-squares minus square-of-mean" identity can produce
        # tiny negative or float-noise values when the underlying
        # returns are exactly constant (e.g. mean*mean rounds
        # slightly differently than sumsq/n). Treat anything inside
        # a small epsilon as zero-variance so constant streams give
        # 0 reward instead of an astronomical Sharpe from sqrt of
        # numerical noise.
        raw_var = self._running_sumsq / n - mean * mean
        epsilon = max(abs(mean), 1.0) * 1e-12
        if raw_var <= epsilon:
            return 0.0
        sharpe = mean / math.sqrt(raw_var)
        return _safe_float(self.scale * sharpe)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def composite(*parts: RewardFunction) -> RewardFunction:
    """Sum any number of reward functions into one.

    Callers usually want `composite(PnLReward(), DrawdownPenaltyReward(
    weight=0.5), TurnoverPenaltyReward(penalty=0.0005))`.
    """

    parts_tuple = tuple(parts)

    def _combined(
        *,
        prev_equity: float,
        new_equity: float,
        prev_position: float,
        new_position: float,
        info: dict,
    ) -> float:
        total = 0.0
        for fn in parts_tuple:
            # Coerce each component to a finite value before
            # accumulating; otherwise one NaN/inf would taint the
            # running sum and `_safe_float(total)` would collapse
            # the WHOLE composite to 0, discarding the other
            # components that returned valid values.
            total += _safe_float(
                fn(
                    prev_equity=prev_equity,
                    new_equity=new_equity,
                    prev_position=prev_position,
                    new_position=new_position,
                    info=info,
                )
            )
        return _safe_float(total)

    return _combined
