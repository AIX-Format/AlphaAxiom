"""
Tests for engine.strategies.

The acceptance bar from Sprint 1 Task 1.2 is "positive Sharpe on 6
months of historical data", which depends on the backtest framework
landing in Task 1.6. Until then this suite locks in the structural
contract: each strategy returns a well-formed `TradingSignal`,
respects warm-up requirements, never raises on edge cases, and emits
the expected action on hand-crafted scenarios.

When the backtest framework lands, the Sharpe assertion will be added
to a separate `test_strategies_backtest.py` so we keep these fast
unit tests deterministic and offline.
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

from engine.strategies import (  # noqa: E402
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
    TradingSignal,
)


# ---------------------------------------------------------------------------
# Helpers to build deterministic OHLCV frames
# ---------------------------------------------------------------------------


def _frame_from_closes(closes: List[float], *, atr_pct: float = 0.005) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a list of closes.

    The high/low spread is `atr_pct` of the bar's midpoint, which is
    enough variance for ATR-based strategies to compute non-zero
    volatility. The index is hourly UTC starting 2024-01-01.
    """
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    opens = [closes[0]] + closes[:-1]
    highs = []
    lows = []
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


# ---------------------------------------------------------------------------
# TradingSignal contract
# ---------------------------------------------------------------------------


def test_trading_signal_rejects_invalid_action() -> None:
    with pytest.raises(ValueError):
        TradingSignal(symbol="BTC/USDT", action="LONG", confidence=0.5)


def test_trading_signal_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        TradingSignal(symbol="BTC/USDT", action="BUY", confidence=1.5)
    with pytest.raises(ValueError):
        TradingSignal(symbol="BTC/USDT", action="BUY", confidence=-0.1)


def test_trading_signal_is_actionable_only_for_non_hold() -> None:
    hold = TradingSignal(symbol="BTC/USDT", action="HOLD", confidence=0.0)
    buy = TradingSignal(symbol="BTC/USDT", action="BUY", confidence=0.5)
    sell_zero = TradingSignal(symbol="BTC/USDT", action="SELL", confidence=0.0)
    assert hold.is_actionable() is False
    assert buy.is_actionable() is True
    assert sell_zero.is_actionable() is False


# ---------------------------------------------------------------------------
# Common strategy contract: shape, warm-up, edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanReversionStrategy("BTC/USDT"),
        lambda: MomentumStrategy("BTC/USDT"),
        lambda: BreakoutStrategy("BTC/USDT"),
    ],
)
def test_strategy_handles_empty_frame(factory) -> None:
    s = factory()
    out = s.generate_signal(pd.DataFrame())
    assert out.action == "HOLD"
    assert out.confidence == 0.0
    assert isinstance(out.strategy, str) and out.strategy != ""


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanReversionStrategy("BTC/USDT"),
        lambda: MomentumStrategy("BTC/USDT"),
        lambda: BreakoutStrategy("BTC/USDT"),
    ],
)
def test_strategy_holds_during_warmup(factory) -> None:
    s = factory()
    # Definitely too short for any of the strategies.
    df = _frame_from_closes([100.0] * 10)
    out = s.generate_signal(df)
    assert out.action == "HOLD"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanReversionStrategy("BTC/USDT"),
        lambda: MomentumStrategy("BTC/USDT"),
        lambda: BreakoutStrategy("BTC/USDT"),
    ],
)
def test_strategy_rejects_missing_columns(factory) -> None:
    s = factory()
    df = pd.DataFrame({"close": list(range(100))})
    out = s.generate_signal(df)
    assert out.action == "HOLD"
    assert "missing columns" in out.reasoning


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanReversionStrategy("BTC/USDT"),
        lambda: MomentumStrategy("BTC/USDT"),
        lambda: BreakoutStrategy("BTC/USDT"),
    ],
)
def test_strategy_holds_on_flat_market(factory) -> None:
    """A perfectly flat market should never produce an actionable signal."""
    s = factory()
    df = _frame_from_closes([100.0] * 120, atr_pct=0.0001)
    out = s.generate_signal(df)
    assert out.action == "HOLD"


def test_strategy_is_subclass_of_base() -> None:
    assert issubclass(MeanReversionStrategy, Strategy)
    assert issubclass(MomentumStrategy, Strategy)
    assert issubclass(BreakoutStrategy, Strategy)


# ---------------------------------------------------------------------------
# Mean reversion: targeted scenarios
# ---------------------------------------------------------------------------


def test_mean_reversion_buys_on_oversold_plus_lower_band() -> None:
    # Flat at 100 for 25 bars to settle the BB middle, then a sharp
    # drop to push close below the lower band and RSI below 30.
    closes = [100.0] * 25 + [99.0, 97.0, 95.0, 92.0, 89.0, 86.0, 83.0, 80.0]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = MeanReversionStrategy(
        "BTC/USDT", bb_period=20, rsi_period=14, atr_period=14
    )
    out = s.generate_signal(df)
    assert out.action == "BUY", out.reasoning
    assert out.entry_price is not None
    assert out.stop_loss is not None and out.stop_loss < out.entry_price
    assert out.take_profit is not None and out.take_profit > out.entry_price
    assert out.metadata["rsi"] <= 30.0


def test_mean_reversion_sells_on_overbought_plus_upper_band() -> None:
    # Flat at 100, then a sharp rise.
    closes = [100.0] * 25 + [101.0, 103.0, 105.0, 108.0, 111.0, 114.0, 117.0, 120.0]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = MeanReversionStrategy(
        "BTC/USDT", bb_period=20, rsi_period=14, atr_period=14
    )
    out = s.generate_signal(df)
    assert out.action == "SELL", out.reasoning
    assert out.entry_price is not None
    assert out.stop_loss is not None and out.stop_loss > out.entry_price
    assert out.take_profit is not None and out.take_profit < out.entry_price
    assert out.metadata["rsi"] >= 70.0


def test_mean_reversion_holds_on_mild_drift() -> None:
    # Gentle drift that should not push past the bands or trigger RSI.
    rng = np.random.default_rng(seed=42)
    closes = [100.0 + 0.1 * i + rng.normal(0, 0.05) for i in range(60)]
    df = _frame_from_closes(closes, atr_pct=0.005)
    s = MeanReversionStrategy("BTC/USDT")
    out = s.generate_signal(df)
    assert out.action == "HOLD"


# ---------------------------------------------------------------------------
# Momentum: targeted scenarios
# ---------------------------------------------------------------------------


def test_momentum_buys_on_bullish_cross_with_uptrend() -> None:
    # Strong uptrend with a periodic ripple. The trend keeps close
    # above the slower EMA throughout, while the ripple gives the
    # MACD pair enough oscillation to cross and recross many times.
    # On the bullish crossovers (line crosses above signal) the
    # uptrend filter is automatically satisfied, so the strategy
    # should fire BUY at least once.
    import math

    n = 140
    closes = [
        80.0 + 0.4 * i + 1.5 * math.sin(2 * math.pi * i / 15.0)
        for i in range(n)
    ]
    df = _frame_from_closes(closes, atr_pct=0.003)
    s = MomentumStrategy("BTC/USDT", trend_period=30)
    fired = None
    for end in range(60, len(df) + 1):
        sub = df.iloc[:end]
        sig = s.generate_signal(sub)
        if sig.action == "BUY":
            fired = sig
            break
    assert fired is not None, "expected at least one BUY across the rippled uptrend"
    assert fired.metadata["macd"] > fired.metadata["macd_signal"]
    assert float(fired.entry_price) > fired.metadata["trend_ema"]
    assert fired.stop_loss is not None and fired.stop_loss < fired.entry_price
    assert fired.take_profit is not None and fired.take_profit > fired.entry_price


def test_momentum_sells_on_bearish_cross_with_downtrend() -> None:
    import math

    n = 140
    closes = [
        140.0 - 0.4 * i + 1.5 * math.sin(2 * math.pi * i / 15.0)
        for i in range(n)
    ]
    df = _frame_from_closes(closes, atr_pct=0.003)
    s = MomentumStrategy("BTC/USDT", trend_period=30)
    fired = None
    for end in range(60, len(df) + 1):
        sub = df.iloc[:end]
        sig = s.generate_signal(sub)
        if sig.action == "SELL":
            fired = sig
            break
    assert fired is not None, "expected at least one SELL across the rippled downtrend"
    assert fired.metadata["macd"] < fired.metadata["macd_signal"]
    assert float(fired.entry_price) < fired.metadata["trend_ema"]
    assert fired.stop_loss is not None and fired.stop_loss > fired.entry_price
    assert fired.take_profit is not None and fired.take_profit < fired.entry_price


def test_momentum_holds_when_cross_disagrees_with_trend() -> None:
    # Steady uptrend, no real reversal: bullish crosses align with
    # uptrend (good) and should occasionally fire, but no SELLs.
    closes = [100.0 + i * 0.5 for i in range(120)]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = MomentumStrategy("BTC/USDT", trend_period=30)
    actions = []
    for end in range(60, len(df) + 1):
        sub = df.iloc[:end]
        actions.append(s.generate_signal(sub).action)
    assert "SELL" not in actions


# ---------------------------------------------------------------------------
# Breakout: targeted scenarios
# ---------------------------------------------------------------------------


def test_breakout_buys_on_new_high_with_atr_buffer() -> None:
    # Tight range then a sharp break above.
    closes = [100.0 + 0.1 * (i % 5) for i in range(30)] + [102.0, 105.0]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = BreakoutStrategy(
        "BTC/USDT", lookback=20, atr_period=14, breakout_atr_mult=0.5
    )
    out = s.generate_signal(df)
    assert out.action == "BUY"
    assert out.entry_price == pytest.approx(closes[-1])
    assert out.metadata["atr"] > 0


def test_breakout_sells_on_new_low_with_atr_buffer() -> None:
    closes = [100.0 - 0.1 * (i % 5) for i in range(30)] + [98.0, 95.0]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = BreakoutStrategy(
        "BTC/USDT", lookback=20, atr_period=14, breakout_atr_mult=0.5
    )
    out = s.generate_signal(df)
    assert out.action == "SELL"
    assert out.entry_price == pytest.approx(closes[-1])


def test_breakout_holds_inside_range() -> None:
    closes = [100.0 + 0.2 * (i % 5) for i in range(30)] + [100.3]
    df = _frame_from_closes(closes, atr_pct=0.002)
    s = BreakoutStrategy("BTC/USDT", lookback=20)
    out = s.generate_signal(df)
    assert out.action == "HOLD"


def test_breakout_rejects_tiny_lookback() -> None:
    with pytest.raises(ValueError):
        BreakoutStrategy("BTC/USDT", lookback=1)


# ---------------------------------------------------------------------------
# Constructor validation: __init__ should fail fast on bad config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atr_period": 0},
        {"atr_period": -3},
        {"breakout_atr_mult": -0.1},
        {"atr_stop_mult": 0},
        {"atr_stop_mult": -1.0},
        {"atr_target_mult": 0},
        {"atr_target_mult": -2.0},
    ],
)
def test_breakout_rejects_invalid_init_args(kwargs) -> None:
    with pytest.raises(ValueError):
        BreakoutStrategy("BTC/USDT", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_period": 0},
        {"slow_period": 0},
        {"signal_period": 0},
        {"trend_period": 0},
        {"atr_period": 0},
        {"fast_period": 30, "slow_period": 20},  # fast >= slow
        {"atr_stop_mult": 0},
        {"atr_stop_mult": -1.0},
        {"atr_target_mult": 0},
    ],
)
def test_momentum_rejects_invalid_init_args(kwargs) -> None:
    with pytest.raises(ValueError):
        MomentumStrategy("BTC/USDT", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rsi_period": 0},
        {"bb_period": 0},
        {"atr_period": 0},
        {"bb_num_std": 0.0},
        {"bb_num_std": -2.0},
        {"atr_stop_mult": 0.0},
        {"atr_stop_mult": -1.0},
        {"rsi_oversold": 80.0, "rsi_overbought": 20.0},  # inverted
        {"rsi_oversold": -10.0},
        {"rsi_overbought": 110.0},
    ],
)
def test_mean_reversion_rejects_invalid_init_args(kwargs) -> None:
    with pytest.raises(ValueError):
        MeanReversionStrategy("BTC/USDT", **kwargs)


# ---------------------------------------------------------------------------
# Self-runner for environments without pytest.
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import traceback

    failures = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if hasattr(fn, "pytestmark"):
            # parametrized test, skip in self-runner.
            continue
        try:
            fn()
            print(f"[ok] {name}")
        except Exception:
            failures += 1
            print(f"[fail] {name}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
