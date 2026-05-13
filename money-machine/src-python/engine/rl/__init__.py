"""
Reinforcement-learning surface for AlphaAxiom.

This package wraps the existing trading engine into the standard
Gymnasium contract so any off-the-shelf RL algorithm (PPO, SAC,
DQN, A2C via stable-baselines3, Ray RLlib, CleanRL, etc.) can be
trained against AlphaAxiom's market simulator without touching the
engine internals.

Layers (each opt-in):

- `env.TradingEnv`: a `gymnasium.Env` that consumes a historical
  OHLCV `DataFrame` and a `PaperAdapter` and exposes a discrete
  action space `{HOLD, BUY, SELL}` plus an observation of the last
  N candles, derived indicators, and the current position state.
- `reward.RewardFunction`: small protocol for pluggable reward
  shaping (mark-to-market PnL, Sharpe-shaped, drawdown-penalised,
  turnover-penalised). Strategies compose these to get realistic
  optimisation signals without re-implementing the trading loop.

The package itself does not pull in `stable_baselines3` or `torch`;
those land in the next slice (`engine/rl/agent.py`). Keeping the
env surface dependency-light means a CI run that only exercises
the env contract does not pay for the full RL toolchain.
"""

from .env import EnvConfig, ObservationConfig, TradingEnv
from .reward import (
    DrawdownPenaltyReward,
    PnLReward,
    RewardFunction,
    SharpeRatioReward,
    TurnoverPenaltyReward,
    composite,
)

__all__ = [
    "DrawdownPenaltyReward",
    "EnvConfig",
    "ObservationConfig",
    "PnLReward",
    "RewardFunction",
    "SharpeRatioReward",
    "TradingEnv",
    "TurnoverPenaltyReward",
    "composite",
]
