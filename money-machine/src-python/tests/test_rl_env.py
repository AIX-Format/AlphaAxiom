"""
Tests for engine.rl.env.TradingEnv and engine.rl.reward.

Coverage:

  - reset() returns an observation of the right shape and dtype,
    and an info dict with the documented keys.
  - step() advances the bar index, updates equity/position
    according to the action, and returns (obs, reward, done,
    truncated, info) in the right shape.
  - The Gymnasium standard checker (gym.utils.env_checker.check_env)
    runs end-to-end without raising.
  - Reward functions: PnLReward returns mark-to-market change,
    DrawdownPenaltyReward penalises new drawdown, Turnover
    PenaltyReward charges on direction flip, SharpeRatioReward
    smooths a stable signal across a window, composite() sums.
  - Episode termination: equity below the configured floor
    terminates; end-of-data truncates.
  - BUY without enough balance is rejected by the underlying
    PaperAdapter and does not mutate position.
  - Short orders are blocked when allow_short=False.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

pytest.importorskip("gymnasium")

from engine.adapters import PaperAdapter  # noqa: E402
from engine.backtest import FixedSlippage, FlatCommission  # noqa: E402
from engine.rl.env import (  # noqa: E402
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    EnvConfig,
    ObservationConfig,
    TradingEnv,
)
from engine.rl.reward import (  # noqa: E402
    DrawdownPenaltyReward,
    PnLReward,
    SharpeRatioReward,
    TurnoverPenaltyReward,
    composite,
)


def _build_df(closes: List[float], *, atr_pct: float = 0.005) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    opens = [closes[0]] + closes[:-1]
    highs, lows = [], []
    for op, cl in zip(opens, closes):
        mid = (op + cl) / 2.0
        spread = abs(mid) * atr_pct
        highs.append(max(op, cl) + spread)
        lows.append(min(op, cl) - spread)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _build_env(
    *,
    closes: List[float],
    initial_equity: float = 10_000.0,
    position_fraction: float = 0.1,
    allow_short: bool = False,
    window_size: int = 16,
    warmup_bars: int = 20,
) -> TradingEnv:
    df = _build_df(closes)
    adapter = PaperAdapter(
        initial_balance=initial_equity,
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
    )
    config = EnvConfig(
        symbol="BTC/USDT",
        initial_equity=initial_equity,
        position_fraction=position_fraction,
        allow_short=allow_short,
        warmup_bars=warmup_bars,
        observation=ObservationConfig(window_size=window_size),
    )
    return TradingEnv(df, adapter, config)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def test_pnl_reward_returns_relative_change() -> None:
    r = PnLReward(as_return=True)
    assert r(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=0.0, new_position=0.5, info={},
    ) == pytest.approx(0.01)
    assert r(
        prev_equity=10_000.0, new_equity=9_950.0,
        prev_position=0.5, new_position=0.5, info={},
    ) == pytest.approx(-0.005)


def test_pnl_reward_absolute_mode() -> None:
    r = PnLReward(as_return=False, scale=2.0)
    out = r(
        prev_equity=10_000.0, new_equity=10_005.0,
        prev_position=0.0, new_position=0.0, info={},
    )
    assert out == pytest.approx(10.0)


def test_drawdown_penalty_charges_new_drawdown_only() -> None:
    r = DrawdownPenaltyReward(weight=1.0)
    info = {"prev_peak_equity": 10_000.0, "new_peak_equity": 10_000.0}
    # Equity drops 5%: drawdown went from 0 -> 0.05; penalty = -0.05.
    out = r(prev_equity=10_000.0, new_equity=9_500.0,
            prev_position=0.0, new_position=0.0, info=info)
    assert out == pytest.approx(-0.05, rel=1e-6)
    # Recovering reduces the drawdown but is not a positive reward;
    # we only charge for new drawdown.
    out = r(prev_equity=9_500.0, new_equity=9_800.0,
            prev_position=0.0, new_position=0.0, info=info)
    assert out == 0.0


def test_turnover_penalty_charges_direction_flip_only() -> None:
    """Direction flip = sign change across zero. Opening from flat,
    scaling in / out, and closing to flat are all unchanged."""
    r = TurnoverPenaltyReward(penalty=0.01)

    # Same direction (long -> bigger long) -> no penalty.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=1.0, new_position=2.0, info={}) == 0.0
    # Flat -> long: NOT a flip, no penalty. Encourages normal
    # position management.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=0.0, new_position=1.0, info={}) == 0.0
    # Long -> flat: NOT a flip, no penalty.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=1.0, new_position=0.0, info={}) == 0.0
    # Flat -> short: NOT a flip, no penalty.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=0.0, new_position=-1.0, info={}) == 0.0
    # Long -> short: REAL flip, charged.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=1.0, new_position=-1.0, info={}) == pytest.approx(-0.01)
    # Short -> long: REAL flip, charged.
    assert r(prev_equity=1.0, new_equity=1.0,
             prev_position=-1.0, new_position=1.0, info={}) == pytest.approx(-0.01)


def test_sharpe_reward_smooths_over_window() -> None:
    r = SharpeRatioReward(window=10, scale=1.0)
    # Feed a stable 1% return stream; Sharpe should be a finite,
    # large positive number after a few steps.
    out = 0.0
    for _ in range(11):
        out = r(
            prev_equity=1.0, new_equity=1.01,
            prev_position=0.0, new_position=0.0,
            info={"step_return": 0.01},
        )
    # Constant returns -> zero variance -> Sharpe returns 0 by the
    # safety branch (we cannot divide by zero).
    assert out == 0.0

    # Now feed alternating returns so variance is non-zero.
    r2 = SharpeRatioReward(window=10, scale=1.0)
    seq = [0.01, -0.005, 0.012, -0.003, 0.008, -0.004, 0.011, -0.002, 0.009, -0.001]
    for s in seq:
        last = r2(
            prev_equity=1.0, new_equity=1.0,
            prev_position=0.0, new_position=0.0,
            info={"step_return": s},
        )
    assert last > 0.0
    assert np.isfinite(last)


def test_composite_sums_components() -> None:
    # Use a long->short flip to trigger the turnover penalty (since
    # the penalty now fires only on direction flips, not on
    # opening from flat).
    fn = composite(
        PnLReward(as_return=True, scale=1.0),
        TurnoverPenaltyReward(penalty=0.001),
    )
    out = fn(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=1.0, new_position=-1.0,
        info={},
    )
    # PnL = +0.01, turnover = -0.001 (long->short flip), total = +0.009.
    assert out == pytest.approx(0.009, rel=1e-6)


def test_composite_safe_coerces_each_component_independently() -> None:
    """If one component returns NaN/inf, the others must still
    contribute. Previously `_safe_float(total)` would collapse the
    whole composite to 0 once a single NaN entered the sum.
    """

    def bad_component(**kwargs: object) -> float:
        return float("nan")

    fn = composite(
        PnLReward(as_return=True, scale=1.0),
        bad_component,
    )
    out = fn(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=0.0, new_position=0.0,
        info={},
    )
    # PnL=0.01, bad=NaN -> coerced to 0 -> total=0.01.
    assert out == pytest.approx(0.01, rel=1e-6)


def test_reward_safely_coerces_non_finite() -> None:
    r = PnLReward(as_return=True)
    # prev_equity 0 short-circuits to 0.0 rather than dividing by zero.
    assert r(prev_equity=0.0, new_equity=100.0, prev_position=0.0, new_position=0.0, info={}) == 0.0


# ---------------------------------------------------------------------------
# Env: reset / step contracts
# ---------------------------------------------------------------------------


def test_env_reset_returns_box_shaped_observation() -> None:
    env = _build_env(closes=[100.0 + i for i in range(40)], window_size=16, warmup_bars=20)
    obs, info = env.reset()
    assert obs.shape == (16 + 5,)
    assert obs.dtype == np.float32
    assert info["step"] == 0
    assert info["bar_index"] == 20
    assert info["equity"] == pytest.approx(10_000.0)
    assert info["position"] == 0.0


def test_env_step_hold_action_does_not_open_position() -> None:
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert info["position"] == 0.0
    assert info["fill"]["status"] == "HOLD"
    assert obs.shape == env.observation_space.shape
    assert not terminated
    assert not truncated


def test_env_step_buy_opens_long_position() -> None:
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    _, _, _, _, info = env.step(ACTION_BUY)
    assert info["position"] > 0.0
    assert info["fill"]["status"] == "FILLED"


def test_env_step_sell_when_long_reduces_position() -> None:
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    env.step(ACTION_BUY)
    before = env._position
    _, _, _, _, info = env.step(ACTION_SELL)
    # Sell reduces (or flips) the long position.
    assert info["position"] < before


def test_env_blocks_short_when_disallowed() -> None:
    env = _build_env(
        closes=[100.0 + i for i in range(40)],
        allow_short=False,
    )
    env.reset()
    # Selling from flat should be blocked rather than opening a short.
    _, _, _, _, info = env.step(ACTION_SELL)
    assert info["position"] == 0.0
    assert info["fill"]["status"] == "BLOCKED_SHORT_DISABLED"


def test_sell_with_slippage_does_not_open_short_when_disallowed() -> None:
    """Pre-cap notional + post-fill clamp: a SELL with nonzero
    slippage might fill at a worse price and end up reducing the
    position by MORE than the long. The post-fill clamp must keep
    the position at exactly 0 in that case.
    """
    df = _build_df([100.0 + i * 0.1 for i in range(40)])
    adapter = PaperAdapter(
        initial_balance=10_000.0,
        commission=FlatCommission(rate=0.0),
        # 200 bps slippage will make a SELL fill 2% below mark.
        slippage=FixedSlippage(bps=200.0),
    )
    config = EnvConfig(
        symbol="BTC/USDT",
        initial_equity=10_000.0,
        position_fraction=0.5,
        allow_short=False,
        warmup_bars=20,
        observation=ObservationConfig(window_size=8),
    )
    env = TradingEnv(df, adapter, config)
    env.reset()
    env.step(ACTION_BUY)
    # Hammer SELL several times so the slippage-induced overshoot
    # paths are exercised.
    for _ in range(3):
        _, _, _, _, info = env.step(ACTION_SELL)
        assert env._position >= -1e-9, (
            f"position went short despite allow_short=False: "
            f"{env._position}; fill={info['fill']}"
        )


def test_sell_caps_to_long_value_when_shorts_disallowed() -> None:
    """A SELL with notional bigger than the current long must close
    the long without flipping into a short."""
    env = _build_env(
        closes=[100.0 + i for i in range(40)],
        position_fraction=0.05,  # smaller BUY so the long is small
        allow_short=False,
        window_size=8,
        warmup_bars=20,
    )
    env.reset()
    env.step(ACTION_BUY)
    long_before = env._position
    assert long_before > 0
    # Now request many SELL ticks with a position_fraction of 5%
    # equity each. The cap inside _apply_action must keep the
    # position >= 0 at every step.
    for _ in range(5):
        _, _, _, _, info = env.step(ACTION_SELL)
        assert env._position >= -1e-9, (
            f"position went short: {env._position}; fill={info['fill']}"
        )


def test_env_truncates_at_end_of_data() -> None:
    n = 30
    env = _build_env(
        closes=[100.0 + i for i in range(n)],
        window_size=8,
        warmup_bars=10,
    )
    env.reset()
    truncated = False
    for _ in range(n + 5):
        _, _, _, truncated, _ = env.step(ACTION_HOLD)
        if truncated:
            break
    assert truncated, "env should truncate when bar index hits end of data"


def test_env_terminates_on_equity_floor() -> None:
    """A leveraged long position in a crashing market should trip
    the equity floor and terminate the episode before truncation.

    Setup: flat warmup, then several BUY steps to build a large
    position, then a sharp price collapse that knocks equity past
    the 50% floor.
    """
    closes = [100.0] * 25 + [100.0, 95.0, 50.0, 20.0, 5.0, 2.0, 1.0, 1.0]
    env = _build_env(
        closes=closes,
        position_fraction=0.9,
        window_size=8,
        warmup_bars=22,
    )
    env.reset()
    # Stack 3 BUYs during the flat phase so the long is heavy when
    # the crash arrives.
    for _ in range(3):
        env.step(ACTION_BUY)
    terminated = False
    truncated = False
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(ACTION_HOLD)
        if terminated or truncated:
            break
    assert terminated, (
        f"expected termination on equity-floor breach; "
        f"got terminated={terminated} truncated={truncated} "
        f"equity={env._equity:.2f}"
    )


def test_env_invalid_action_raises() -> None:
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_env_rejects_non_integer_action() -> None:
    """Float actions must NOT be silently coerced via int(). A policy
    bug that produces 1.9 should fail fast, not execute BUY.
    """
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    with pytest.raises(ValueError):
        env.step(1.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        env.step("BUY")  # type: ignore[arg-type]


def test_env_config_seed_used_when_reset_seed_omitted() -> None:
    """Two envs with the same EnvConfig.seed should produce
    identical observation sequences from reset() (no explicit
    seed argument). The previous code only honoured the method
    argument, making the config knob a no-op for the typical
    `env.reset()` call pattern.
    """
    df = _build_df([100.0 + 0.5 * i for i in range(60)])

    def fresh_env() -> TradingEnv:
        adapter = PaperAdapter(
            initial_balance=10_000.0,
            commission=FlatCommission(rate=0.0),
            slippage=FixedSlippage(bps=0.0),
        )
        config = EnvConfig(
            symbol="BTC/USDT",
            initial_equity=10_000.0,
            warmup_bars=20,
            seed=42,
            observation=ObservationConfig(window_size=8),
        )
        return TradingEnv(df, adapter, config)

    env_a = fresh_env()
    env_b = fresh_env()
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    np.testing.assert_allclose(obs_a, obs_b)


def test_env_rejects_dataframe_missing_columns() -> None:
    df = pd.DataFrame({"close": list(range(40))})
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(ValueError, match="missing OHLCV columns"):
        TradingEnv(df, adapter, EnvConfig(warmup_bars=20))


def test_env_rejects_too_short_data() -> None:
    env = None
    df = _build_df([100.0] * 5)
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(ValueError):
        env = TradingEnv(df, adapter, EnvConfig(warmup_bars=20))
    assert env is None


# ---------------------------------------------------------------------------
# Gymnasium standard check
# ---------------------------------------------------------------------------


def test_env_runs_inside_running_event_loop() -> None:
    """The env must not call `asyncio.run` per step; otherwise it
    blows up when used inside a running event loop (RLlib worker,
    Jupyter, async test harness). Drive a few steps from inside an
    async function and confirm no `RuntimeError` is raised.
    """
    import asyncio

    env = _build_env(closes=[100.0 + i for i in range(40)], window_size=8, warmup_bars=20)

    async def drive() -> int:
        env.reset()
        # 5 alternating actions inside an active event loop.
        for action in [ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD, ACTION_BUY]:
            env.step(action)
        return env._cumulative_steps

    steps = asyncio.run(drive())
    assert steps == 5


def test_indicators_are_precomputed_once() -> None:
    """A pure observation pull after `reset` should hit the
    precomputed indicator arrays, not recompute them. We instrument
    the rsi function in `engine.rl.env` to assert it is NOT called
    on each step.
    """
    import engine.rl.env as env_module

    env = _build_env(closes=[100.0 + i for i in range(40)], window_size=8, warmup_bars=20)
    env.reset()

    call_count = [0]
    original = env_module.rsi

    def counting(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    env_module.rsi = counting  # type: ignore[assignment]
    try:
        for _ in range(10):
            env.step(ACTION_HOLD)
    finally:
        env_module.rsi = original  # type: ignore[assignment]

    # Zero calls during the step loop: indicators are read from the
    # cached numpy arrays.
    assert call_count[0] == 0


def test_position_observation_is_scale_invariant_weight() -> None:
    """Position in the observation must be `(position * price) /
    equity`, not the raw quantity. A BTC long of 0.1 at price 50k
    on a 10k equity should show position weight ≈ 0.5; a SHIB long
    of 5e7 at price 1e-5 on the same equity should also show ≈ 0.5.
    """
    # BTC-like asset: high price, small position size.
    closes_btc = [50_000.0] * 40
    env_btc = _build_env(closes=closes_btc, window_size=8, warmup_bars=20, position_fraction=0.5)
    env_btc.reset()
    env_btc.step(ACTION_BUY)
    obs_btc = env_btc._build_observation()
    pos_weight_btc = float(obs_btc[-2])

    # SHIB-like asset: tiny price, huge position size.
    closes_shib = [1e-5] * 40
    env_shib = _build_env(closes=closes_shib, window_size=8, warmup_bars=20, position_fraction=0.5)
    env_shib.reset()
    env_shib.step(ACTION_BUY)
    obs_shib = env_shib._build_observation()
    pos_weight_shib = float(obs_shib[-2])

    # Both should land near 0.5 (50% of equity deployed), regardless
    # of the asset's unit scale. The fixture has zero slippage and
    # zero commission so the math is exact up to float rounding.
    assert pos_weight_btc == pytest.approx(0.5, abs=1e-3)
    assert pos_weight_shib == pytest.approx(0.5, abs=1e-3)


def test_paper_adapter_reset_clears_state() -> None:
    """The new `PaperAdapter.reset()` replaces the old pattern of
    poking at `_balance` / `_positions` / `_order_history` from
    outside. Verify it does what the env relies on.
    """
    import asyncio

    from engine.adapters import OrderRequest, OrderSide, OrderType

    adapter = PaperAdapter(initial_balance=5_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)
    asyncio.run(adapter.place_order(OrderRequest(
        client_order_id="x",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=500.0,
    )))
    assert asyncio.run(adapter.get_balance()) < 5_000.0

    adapter.reset(initial_balance=10_000.0)
    assert asyncio.run(adapter.get_balance()) == 10_000.0
    assert asyncio.run(adapter.get_positions()) == {}
    assert asyncio.run(adapter.get_open_orders()) == []


def test_env_passes_gymnasium_env_checker() -> None:
    """The official Gymnasium env_checker runs reset/step/close,
    verifies space shapes, dtypes, and seeding behaviour. If
    anything in the contract drifts this test catches it.
    """
    from gymnasium.utils.env_checker import check_env

    env = _build_env(closes=[100.0 + i * 0.5 for i in range(80)], window_size=16, warmup_bars=20)
    # check_env wants `render_modes` metadata if render is called;
    # skip_render_check=True skips that path and focuses on the
    # core contract.
    check_env(env.unwrapped, skip_render_check=True)


# ---------------------------------------------------------------------------
# TradingEnv: additional edge cases
# ---------------------------------------------------------------------------


def test_env_rejects_warmup_bars_less_than_window_size() -> None:
    """warmup_bars < window_size must raise ValueError at construction
    time. Without this guard the observation window cannot be fully
    populated at the start of an episode.
    """
    df = _build_df([100.0 + i for i in range(40)])
    adapter = PaperAdapter(initial_balance=10_000.0)
    config = EnvConfig(
        warmup_bars=5,
        observation=ObservationConfig(window_size=16),
    )
    with pytest.raises(ValueError, match="warmup_bars must be >= observation.window_size"):
        TradingEnv(df, adapter, config)


def test_env_allows_warmup_bars_exactly_equal_to_window_size() -> None:
    """warmup_bars == window_size is the minimum valid value and must
    not raise.
    """
    df = _build_df([100.0 + i for i in range(40)])
    adapter = PaperAdapter(initial_balance=10_000.0)
    config = EnvConfig(
        warmup_bars=16,
        observation=ObservationConfig(window_size=16),
    )
    env = TradingEnv(df, adapter, config)
    obs, info = env.reset()
    assert obs.shape == (16 + 5,)


def test_env_allows_short_when_configured() -> None:
    """With allow_short=True, a SELL from flat must open a short
    position (negative quantity) rather than being blocked.
    """
    df = _build_df([100.0 + i for i in range(40)])
    adapter = PaperAdapter(
        initial_balance=10_000.0,
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
    )
    config = EnvConfig(
        symbol="BTC/USDT",
        initial_equity=10_000.0,
        position_fraction=0.1,
        allow_short=True,
        warmup_bars=20,
        observation=ObservationConfig(window_size=16),
    )
    env = TradingEnv(df, adapter, config)
    env.reset()
    # SELL from flat: should open a short when allow_short=True.
    _, _, _, _, info = env.step(ACTION_SELL)
    assert info["fill"]["status"] == "FILLED"
    assert env._position < 0.0


def test_env_step_info_contains_all_documented_keys() -> None:
    """step() info must contain step, equity, position, bar_index,
    prev_peak_equity, new_peak_equity, step_return, and fill.
    """
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    _, _, _, _, info = env.step(ACTION_HOLD)
    required_keys = {
        "step", "equity", "position", "bar_index",
        "prev_peak_equity", "new_peak_equity", "step_return", "fill",
    }
    assert required_keys.issubset(info.keys()), (
        f"Missing keys: {required_keys - info.keys()}"
    )


def test_env_peak_equity_never_decreases() -> None:
    """_peak_equity must be a running maximum: even after a price
    drop that reduces equity, the peak must remain at its high-water
    mark and never fall below it.
    """
    closes = [100.0] * 25 + [120.0, 130.0, 110.0, 90.0, 80.0]
    env = _build_env(
        closes=closes,
        window_size=8,
        warmup_bars=20,
        position_fraction=0.5,
    )
    env.reset()
    env.step(ACTION_BUY)  # Build a position before the rally

    peak_seen = env._peak_equity
    for _ in range(len(closes) - 20 - 2):
        env.step(ACTION_HOLD)
        assert env._peak_equity >= peak_seen, (
            f"peak_equity decreased: was {peak_seen}, now {env._peak_equity}"
        )
        peak_seen = env._peak_equity


def test_env_reset_restores_cumulative_steps_and_position() -> None:
    """After a sequence of steps, reset() must restore the env to
    the same state as a fresh episode: cumulative_steps=0, position=0,
    and equity back to initial.
    """
    env = _build_env(closes=[100.0 + i for i in range(40)])
    env.reset()
    for _ in range(5):
        env.step(ACTION_BUY)

    assert env._cumulative_steps == 5
    assert env._position > 0.0

    env.reset()

    assert env._cumulative_steps == 0
    assert env._position == 0.0
    assert env._equity == pytest.approx(10_000.0)
    assert env._peak_equity == pytest.approx(10_000.0)


def test_env_observation_values_clipped_to_box_bounds() -> None:
    """All observation values must lie within [-10, 10] per the
    declared Box space, even for extreme price movements that would
    otherwise produce very large normalised values.
    """
    # A sudden 100x price spike would produce extreme normalised closes
    # without clipping.
    closes = [100.0] * 25 + [10_000.0] * 15
    env = _build_env(closes=closes, window_size=8, warmup_bars=20)
    env.reset()
    for _ in range(14):
        obs, _, _, _, _ = env.step(ACTION_HOLD)
        assert np.all(obs >= -10.0) and np.all(obs <= 10.0), (
            f"observation out of [-10, 10] bounds: min={obs.min()}, max={obs.max()}"
        )


def test_env_equity_norm_zero_at_reset() -> None:
    """The equity_norm component (last element of the observation) is
    `(equity / initial_equity) - 1`. At reset the equity equals
    initial_equity, so equity_norm must be exactly 0.
    """
    env = _build_env(closes=[100.0 + i for i in range(40)])
    obs, _ = env.reset()
    equity_norm = float(obs[-1])
    assert equity_norm == pytest.approx(0.0, abs=1e-6)


def test_safe_indicator_returns_fallback_for_nan() -> None:
    """_safe_indicator must return the fallback when the array
    contains NaN at the requested bar index.
    """
    arr = np.array([1.0, float("nan"), 3.0])
    result = TradingEnv._safe_indicator(arr, bar=1, fallback=99.0)
    assert result == pytest.approx(99.0)


def test_safe_indicator_returns_fallback_for_inf() -> None:
    """_safe_indicator must return the fallback when the value is inf."""
    arr = np.array([1.0, float("inf"), 3.0])
    result = TradingEnv._safe_indicator(arr, bar=1, fallback=42.0)
    assert result == pytest.approx(42.0)


def test_safe_indicator_returns_fallback_for_out_of_range_bar() -> None:
    """Bar indices out of [0, array_length) must return the fallback."""
    arr = np.array([1.0, 2.0, 3.0])
    assert TradingEnv._safe_indicator(arr, bar=-1, fallback=7.0) == pytest.approx(7.0)
    assert TradingEnv._safe_indicator(arr, bar=10, fallback=7.0) == pytest.approx(7.0)


def test_safe_indicator_returns_actual_value_for_valid_bar() -> None:
    """For a valid bar with a finite value the indicator itself is returned."""
    arr = np.array([10.0, 20.0, 30.0])
    assert TradingEnv._safe_indicator(arr, bar=1, fallback=0.0) == pytest.approx(20.0)


def test_env_validate_data_rejects_empty_dataframe() -> None:
    """An empty DataFrame must raise ValueError at construction."""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    adapter = PaperAdapter(initial_balance=10_000.0)
    with pytest.raises(ValueError, match="non-empty OHLCV DataFrame"):
        TradingEnv(df, adapter, EnvConfig(warmup_bars=20))


# ---------------------------------------------------------------------------
# Reward function: additional edge cases
# ---------------------------------------------------------------------------


def test_drawdown_penalty_fallback_when_peak_missing_from_info() -> None:
    """When info lacks 'prev_peak_equity', the reward function must
    fall back to `prev_equity` as the implicit peak and compute
    drawdown relative to that.
    """
    r = DrawdownPenaltyReward(weight=1.0)
    # No peak info: prev_peak defaults to prev_equity (10000), so
    # prev_dd = 0. new_peak defaults to max(prev_peak, new_equity).
    # new_equity 9000 < prev_peak 10000 -> new_dd = (10000-9000)/10000 = 0.1.
    out = r(
        prev_equity=10_000.0,
        new_equity=9_000.0,
        prev_position=0.0,
        new_position=0.0,
        info={},  # no peak keys
    )
    assert out == pytest.approx(-0.1, rel=1e-6)


def test_drawdown_penalty_weighted() -> None:
    """weight != 1.0 must scale the penalty proportionally."""
    r_unit = DrawdownPenaltyReward(weight=1.0)
    r_half = DrawdownPenaltyReward(weight=0.5)
    info = {"prev_peak_equity": 10_000.0, "new_peak_equity": 10_000.0}
    out_unit = r_unit(prev_equity=10_000.0, new_equity=9_000.0,
                      prev_position=0.0, new_position=0.0, info=info)
    out_half = r_half(prev_equity=10_000.0, new_equity=9_000.0,
                      prev_position=0.0, new_position=0.0, info=info)
    assert out_half == pytest.approx(out_unit * 0.5, rel=1e-6)


def test_drawdown_penalty_zero_on_new_equity_high() -> None:
    """If new equity exceeds the previous peak (a new high-water mark),
    there is no new drawdown, so the penalty must be 0.
    """
    r = DrawdownPenaltyReward(weight=1.0)
    # Equity rises from 10k -> 11k (new peak). No additional drawdown.
    info = {"prev_peak_equity": 10_000.0, "new_peak_equity": 11_000.0}
    out = r(
        prev_equity=10_000.0, new_equity=11_000.0,
        prev_position=0.0, new_position=0.0,
        info=info,
    )
    assert out == pytest.approx(0.0)


def test_sharpe_reward_returns_zero_with_single_sample() -> None:
    """With fewer than 2 samples there is no variance to compute;
    the reward must return 0.0 for the first step.
    """
    r = SharpeRatioReward(window=10, scale=1.0)
    out = r(
        prev_equity=1.0, new_equity=1.01,
        prev_position=0.0, new_position=0.0,
        info={"step_return": 0.01},
    )
    assert out == pytest.approx(0.0)


def test_sharpe_reward_window_eviction_updates_running_stats() -> None:
    """After the window is full, each new return must evict the oldest
    so the running sum/sumsq stay consistent with a fresh computation
    over only the last `window` entries.
    """
    import math as _math

    window = 5
    r = SharpeRatioReward(window=window, scale=1.0)
    # Fill the window with 5 values; then add one more that evicts the first.
    returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.015]
    for ret in returns:
        last = r(
            prev_equity=1.0, new_equity=1.0,
            prev_position=0.0, new_position=0.0,
            info={"step_return": ret},
        )

    # After 6 steps the window holds returns[1:6].
    window_returns = returns[1:]  # last 5
    n = len(window_returns)
    mean = sum(window_returns) / n
    var = sum((x - mean) ** 2 for x in window_returns) / n
    expected = (mean / _math.sqrt(var)) if var > 1e-12 else 0.0
    assert last == pytest.approx(expected, rel=1e-3)


def test_sharpe_reward_negative_scale_flips_sign() -> None:
    """A negative scale should produce a negative Sharpe reward for
    a positive mean return, allowing reward inversion for adversarial
    training scenarios.
    """
    r_pos = SharpeRatioReward(window=10, scale=1.0)
    r_neg = SharpeRatioReward(window=10, scale=-1.0)
    seq = [0.01, -0.005, 0.012, -0.003, 0.008, -0.004, 0.011, -0.002, 0.009, -0.001]
    out_pos, out_neg = 0.0, 0.0
    for s in seq:
        out_pos = r_pos(prev_equity=1.0, new_equity=1.0,
                        prev_position=0.0, new_position=0.0,
                        info={"step_return": s})
        out_neg = r_neg(prev_equity=1.0, new_equity=1.0,
                        prev_position=0.0, new_position=0.0,
                        info={"step_return": s})
    assert out_neg == pytest.approx(-out_pos, rel=1e-6)


def test_composite_with_zero_components_returns_zero() -> None:
    """composite() with no arguments must return 0.0 (empty sum)."""
    fn = composite()
    out = fn(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=0.0, new_position=0.0,
        info={},
    )
    assert out == pytest.approx(0.0)


def test_composite_with_inf_component_coerces_to_zero() -> None:
    """A component returning +inf must be coerced to 0 so the other
    components' contributions are not tainted.
    """
    def inf_component(**kwargs: object) -> float:
        return float("inf")

    fn = composite(
        PnLReward(as_return=True, scale=1.0),
        inf_component,
    )
    out = fn(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=0.0, new_position=0.0,
        info={},
    )
    # PnL = 0.01, inf -> 0.0, total = 0.01.
    assert out == pytest.approx(0.01, rel=1e-6)


def test_pnl_reward_scale_applied_in_return_mode() -> None:
    """scale must multiply the return even when as_return=True."""
    r = PnLReward(as_return=True, scale=10.0)
    out = r(
        prev_equity=10_000.0, new_equity=10_100.0,
        prev_position=0.0, new_position=0.0, info={},
    )
    # return = 0.01, scaled = 0.1.
    assert out == pytest.approx(0.1, rel=1e-6)
