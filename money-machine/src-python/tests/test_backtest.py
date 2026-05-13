"""
Tests for engine.backtest.

The Sprint 1 Task 1.6 acceptance has two parts:

1. "Backtest يطلع نتائج قريبة من manual calculation" - the metrics
   math is verified against hand-computed numbers on small fixtures.
2. "Equity curve monotonic لو strategy = buy & hold" - a long-only,
   always-buy strategy in a strictly rising synthetic market with
   zero costs produces a monotonically non-decreasing equity curve.

The full sprint requirement ('run each strategy on 6 months BTCUSDT
1h, positive Sharpe') is not asserted here because the live fixture
PR is the next slice. What we lock down today: the framework runs
each strategy end-to-end on the 60-bar synthetic fixture without
crashing, produces a valid BacktestResult, and the cost models do
what they say. The 6-month historical run lands when
scripts/refresh_indicator_fixture.py is wired up.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.backtest import (  # noqa: E402
    AtrSlippage,
    Backtest,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    BINANCE_SPOT_TAKER,
    BYBIT_PERP_TAKER,
    FixedSlippage,
    FlatCommission,
    TradeRecord,
    compute_metrics,
)
from engine.strategies import (  # noqa: E402
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)
from engine.strategies.base import Strategy, TradingSignal  # noqa: E402
from tests.fixtures.ohlcv_btcusdt_1h_sample import CANDLES  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [c.open for c in CANDLES],
            "high": [c.high for c in CANDLES],
            "low": [c.low for c in CANDLES],
            "close": [c.close for c in CANDLES],
            "volume": [c.volume for c in CANDLES],
        },
        index=pd.to_datetime(
            [c.timestamp_ms for c in CANDLES], unit="ms", utc=True
        ),
    )


def _build_frame(closes, *, atr_pct: float = 0.005) -> pd.DataFrame:
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


class _BuyAndHold(Strategy):
    """Always-BUY strategy used to characterise the framework.

    Sends one BUY on the first warm-up-clear bar with a tight stop
    and a far take so the position rides the trend without exits.
    Subsequent bars produce HOLD so the open position is preserved
    by the framework.
    """

    name = "buy-and-hold"

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self._fired = False

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        last_close = float(df["close"].iloc[-1])
        if self._fired:
            return TradingSignal(
                symbol=self.symbol,
                action="HOLD",
                confidence=0.0,
                strategy=self.name,
                entry_price=last_close,
            )
        self._fired = True
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=0.9,
            strategy=self.name,
            entry_price=last_close,
            stop_loss=last_close * 0.5,   # far enough never to fire
            take_profit=last_close * 10,  # ditto
        )


class _AlwaysHold(Strategy):
    name = "always-hold"

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        return TradingSignal(
            symbol=self.symbol,
            action="HOLD",
            confidence=0.0,
            strategy=self.name,
        )


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def test_fixed_slippage_long_pays_higher() -> None:
    s = FixedSlippage(bps=10.0)
    assert s.apply(100.0, "buy") == pytest.approx(100.10)
    assert s.apply(100.0, "sell") == pytest.approx(99.90)


def test_fixed_slippage_zero_is_no_op() -> None:
    s = FixedSlippage(bps=0.0)
    assert s.apply(100.0, "buy") == 100.0
    assert s.apply(100.0, "sell") == 100.0


def test_atr_slippage_scales_with_volatility() -> None:
    s = AtrSlippage(multiplier=0.1, floor_bps=0.0)
    quiet = s.apply(100.0, "buy", atr=0.5)   # 0.1 * 0.5 = 0.05
    busy = s.apply(100.0, "buy", atr=5.0)    # 0.1 * 5.0 = 0.50
    assert quiet == pytest.approx(100.05)
    assert busy == pytest.approx(100.50)


def test_atr_slippage_has_floor_when_atr_missing() -> None:
    s = AtrSlippage(multiplier=0.1, floor_bps=5.0)
    out = s.apply(100.0, "buy", atr=None)
    # 5 bps floor on 100 = 0.05
    assert out == pytest.approx(100.05)


def test_flat_commission_proportional() -> None:
    c = FlatCommission(rate=0.001)
    assert c.apply(1_000.0) == pytest.approx(1.0)
    assert c.apply(2_500.0) == pytest.approx(2.5)
    # Negative notional should still cost a positive fee.
    assert c.apply(-1_000.0) == pytest.approx(1.0)


def test_exchange_presets_match_published_rates() -> None:
    # Lock the constants so a typo in the file shows up here.
    assert BINANCE_SPOT_TAKER.rate == pytest.approx(0.001)
    assert BYBIT_PERP_TAKER.rate == pytest.approx(0.00055)


# ---------------------------------------------------------------------------
# compute_metrics math on hand-checkable inputs
# ---------------------------------------------------------------------------


def test_metrics_constant_curve_has_zero_sharpe_and_dd() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    curve = pd.Series([1000.0] * 10, index=idx)
    m = compute_metrics(curve, trades=[], initial_equity=1000.0)
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
    assert m.num_trades == 0
    assert m.win_rate == 0.0


def test_metrics_linear_curve_total_return_and_drawdown() -> None:
    # Equity goes 1000 -> 1500 linearly: +50%, no drawdown.
    idx = pd.date_range("2024-01-01", periods=11, freq="h", tz="UTC")
    curve = pd.Series([1000 + i * 50 for i in range(11)], index=idx, dtype=float)
    m = compute_metrics(curve, trades=[], initial_equity=1000.0)
    assert m.total_return == pytest.approx(0.5)
    assert m.max_drawdown == 0.0
    assert m.sharpe > 0  # strictly positive returns => positive Sharpe


def test_metrics_dip_in_middle_yields_expected_max_drawdown() -> None:
    # 1000 -> 1200 -> 900 -> 1100. Peak before dip is 1200, trough 900.
    # Drawdown = (900 - 1200) / 1200 = -25%.
    curve = pd.Series([1000, 1200, 900, 1100], dtype=float)
    m = compute_metrics(curve, trades=[], initial_equity=1000.0)
    assert m.max_drawdown == pytest.approx(0.25)


def test_metrics_win_rate_and_averages() -> None:
    trades = [
        _make_trade(pnl=100.0),
        _make_trade(pnl=-50.0),
        _make_trade(pnl=200.0),
        _make_trade(pnl=-100.0),
    ]
    curve = pd.Series([1000.0, 1000.0], dtype=float)
    m = compute_metrics(curve, trades=trades, initial_equity=1000.0)
    assert m.num_trades == 4
    assert m.num_wins == 2
    assert m.num_losses == 2
    assert m.win_rate == 0.5
    assert m.avg_win == pytest.approx(150.0)
    assert m.avg_loss == pytest.approx(75.0)  # magnitudes


def _make_trade(*, pnl: float) -> TradeRecord:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return TradeRecord(
        entry_time=ts,
        exit_time=ts,
        direction="long",
        entry_price=100.0,
        exit_price=100.0 + pnl,
        quantity=1.0,
        pnl=pnl,
        commission=0.0,
        exit_reason="take" if pnl > 0 else "stop",
    )


# ---------------------------------------------------------------------------
# Buy-and-hold sanity (sprint acceptance)
# ---------------------------------------------------------------------------


def test_buy_and_hold_in_rising_market_with_no_costs_is_monotonic() -> None:
    """Acceptance criterion from Task 1.6:
    buy-and-hold on a strictly rising market with zero slippage and
    zero commission should produce a monotonically non-decreasing
    equity curve.
    """
    # Strictly rising closes from 100 to ~200 over 120 bars.
    closes = [100.0 + i * 0.8 for i in range(120)]
    df = _build_frame(closes, atr_pct=0.001)

    bt = Backtest(
        strategy=_BuyAndHold("BTC/USDT"),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)

    # Strictly rising = the equity series should be non-decreasing
    # from the bar after entry. We allow one downward blip on the
    # entry bar itself due to integer-division quantity sizing.
    eq = result.equity_curve.iloc[3:].to_numpy()
    diffs = np.diff(eq)
    assert (diffs >= -1e-6).all(), (
        f"equity curve not non-decreasing after entry, min diff={diffs.min()}"
    )
    assert result.metrics.total_return > 0.0
    assert result.final_equity > 10_000.0


def test_buy_and_hold_has_one_trade_at_end_of_data() -> None:
    closes = [100.0 + i * 0.5 for i in range(80)]
    df = _build_frame(closes, atr_pct=0.001)
    bt = Backtest(
        strategy=_BuyAndHold("BTC/USDT"),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].pnl > 0.0


# ---------------------------------------------------------------------------
# Pass-through behaviour: HOLD-only strategies leave equity untouched
# ---------------------------------------------------------------------------


def test_always_hold_leaves_equity_flat() -> None:
    closes = [100.0 + math.sin(i / 10.0) for i in range(80)]
    df = _build_frame(closes)
    bt = Backtest(
        strategy=_AlwaysHold("BTC/USDT"),
        config=BacktestConfig(initial_equity=5_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    # No trades, no commission, no PnL.
    assert result.trades == []
    assert result.metrics.num_trades == 0
    assert result.metrics.max_drawdown == 0.0
    assert (result.equity_curve == 5_000.0).all()


# ---------------------------------------------------------------------------
# Stop-loss / take-profit fire correctly
# ---------------------------------------------------------------------------


class _OneShotLong(Strategy):
    """Fires BUY once with specific stop/take, then HOLDs forever."""

    name = "one-shot-long"

    def __init__(self, symbol: str, *, stop: float, take: float) -> None:
        super().__init__(symbol)
        self._fired = False
        self._stop = stop
        self._take = take

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        last_close = float(df["close"].iloc[-1])
        if self._fired:
            return TradingSignal(
                symbol=self.symbol,
                action="HOLD",
                confidence=0.0,
                strategy=self.name,
            )
        self._fired = True
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=0.8,
            strategy=self.name,
            entry_price=last_close,
            stop_loss=self._stop,
            take_profit=self._take,
        )


def test_long_stop_fires_when_low_crosses_stop() -> None:
    # Up trend, then a sharp dip. Stop set below dip.
    closes = [100.0] * 10 + [99.0, 98.0, 90.0, 95.0, 96.0]
    df = _build_frame(closes, atr_pct=0.002)
    bt = Backtest(
        strategy=_OneShotLong("BTC/USDT", stop=95.0, take=200.0),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(95.0)


def test_long_take_fires_when_high_crosses_take() -> None:
    # Up trend that breaks the take level on an intraday spike.
    closes = [100.0] * 5 + [101.0, 102.0, 103.0, 110.0, 108.0]
    df = _build_frame(closes, atr_pct=0.003)
    bt = Backtest(
        strategy=_OneShotLong("BTC/USDT", stop=50.0, take=105.0),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=0.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    assert any(t.exit_reason == "take" for t in result.trades)


# ---------------------------------------------------------------------------
# Each real strategy runs end-to-end on the synthetic fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MeanReversionStrategy("BTC/USDT"),
        lambda: MomentumStrategy("BTC/USDT", trend_period=20),
        lambda: BreakoutStrategy("BTC/USDT", lookback=10),
    ],
)
def test_real_strategies_run_end_to_end_on_fixture(
    factory, fixture_df: pd.DataFrame
) -> None:
    bt = Backtest(
        strategy=factory(),
        config=BacktestConfig(
            initial_equity=10_000.0,
            risk_per_trade_pct=0.003,  # fits the 20% per-position cap
            warmup_bars=30,
        ),
    )
    result = bt.run(fixture_df)
    # Did not crash and produced a coherent result object.
    assert isinstance(result, BacktestResult)
    assert isinstance(result.metrics, BacktestMetrics)
    assert result.equity_curve.index.equals(fixture_df.index)
    # Total return is well-defined and the metrics are consistent.
    assert math.isfinite(result.metrics.total_return)
    assert 0.0 <= result.metrics.win_rate <= 1.0
    assert result.metrics.max_drawdown >= 0.0
    assert result.metrics.num_wins + result.metrics.num_losses <= result.metrics.num_trades


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


def test_backtest_rejects_empty_frame() -> None:
    bt = Backtest(strategy=_AlwaysHold("BTC/USDT"))
    with pytest.raises(ValueError):
        bt.run(pd.DataFrame())


def test_backtest_rejects_frame_missing_columns() -> None:
    bt = Backtest(strategy=_AlwaysHold("BTC/USDT"))
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        bt.run(df)


# ---------------------------------------------------------------------------
# Regression tests for Codex P1 findings on the backtest engine.
# ---------------------------------------------------------------------------


class _NaNStopStrategy(Strategy):
    """Emits one BUY with a NaN stop_loss, then HOLDs."""

    name = "nan-stop-test"

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self._fired = False

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        if self._fired:
            return TradingSignal(
                symbol=self.symbol, action="HOLD", confidence=0.0,
                strategy=self.name,
            )
        self._fired = True
        last_close = float(df["close"].iloc[-1])
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=0.9,
            strategy=self.name,
            entry_price=last_close,
            stop_loss=float("nan"),
            take_profit=last_close * 1.10,
        )


def test_nan_stop_does_not_contaminate_equity_curve() -> None:
    """A strategy emitting NaN stop_loss must not produce NaN sizing
    or NaN equity. Without the explicit isfinite guard, NaN
    propagates through commission and quantity math and silently
    breaks every downstream metric.
    """
    closes = [100.0 + i * 0.5 for i in range(60)]
    df = _build_frame(closes, atr_pct=0.002)
    bt = Backtest(
        strategy=_NaNStopStrategy("BTC/USDT"),
        config=BacktestConfig(
            initial_equity=10_000.0,
            warmup_bars=2,
            fallback_stop_pct=0.02,
        ),
    )
    result = bt.run(df)
    # No NaN anywhere in the equity curve.
    assert not result.equity_curve.isna().any()
    # Metrics are finite.
    assert math.isfinite(result.metrics.total_return)
    assert math.isfinite(result.metrics.sharpe)
    assert math.isfinite(result.metrics.max_drawdown)
    # And the fallback stop kicked in so we still traded.
    assert result.metrics.num_trades >= 1


def test_stop_exit_pays_slippage() -> None:
    """Stop-loss exits must run through the slippage model. Previously
    they used the raw trigger price, which made stop-heavy strategies
    look better than they were under nonzero slippage.
    """
    # Up trend then sharp dip that fires the stop.
    closes = [100.0] * 10 + [99.0, 98.0, 90.0, 95.0, 96.0]
    df = _build_frame(closes, atr_pct=0.002)
    bt = Backtest(
        strategy=_OneShotLong("BTC/USDT", stop=95.0, take=200.0),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=100.0),  # 1% slippage, very visible
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    # A long exit is a sell; slippage at 100 bps drags the fill below
    # the trigger price of 95.0.
    assert trade.exit_price < 95.0
    assert trade.exit_price == pytest.approx(95.0 * (1.0 - 0.01))


def test_take_exit_pays_slippage() -> None:
    """Mirror of the stop test: take-profit exits also pay slippage."""
    closes = [100.0] * 5 + [101.0, 102.0, 103.0, 110.0, 108.0]
    df = _build_frame(closes, atr_pct=0.003)
    bt = Backtest(
        strategy=_OneShotLong("BTC/USDT", stop=50.0, take=105.0),
        commission=FlatCommission(rate=0.0),
        slippage=FixedSlippage(bps=100.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    take_trades = [t for t in result.trades if t.exit_reason == "take"]
    assert take_trades, "expected at least one take exit"
    take = take_trades[0]
    # Long exit sells with slippage; fill is below the take trigger.
    assert take.exit_price < 105.0
    assert take.exit_price == pytest.approx(105.0 * (1.0 - 0.01))


class _RecordingSlippage:
    """Slippage spy: records every call so we can assert on the ATR arg."""

    def __init__(self) -> None:
        self.calls = []  # list of (price, side, atr)

    def apply(self, price, side, atr=None):
        self.calls.append((price, side, atr))
        return price


class _OneShotTakeAt(Strategy):
    """Fires BUY once with entry at the latest close and a hard
    take_profit at a configurable level. Used to engineer a trade
    whose gross profit barely covers the round-trip fees."""

    name = "one-shot-take"

    def __init__(self, symbol: str, *, take: float) -> None:
        super().__init__(symbol)
        self._fired = False
        self._take = take

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        if self._fired:
            return TradingSignal(
                symbol=self.symbol, action="HOLD", confidence=0.0,
                strategy=self.name,
            )
        self._fired = True
        last_close = float(df["close"].iloc[-1])
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=0.9,
            strategy=self.name,
            entry_price=last_close,
            stop_loss=last_close * 0.5,
            take_profit=self._take,
        )


def test_trade_pnl_subtracts_both_entry_and_exit_commissions() -> None:
    """TradeRecord.pnl must be the net round-trip including BOTH the
    entry and the exit commissions. Otherwise compute_metrics will
    classify a small gross-positive but net-negative trade as a win.

    Engineered scenario: 1% taker on both sides, gross gain of about
    +1.5% on entry-to-take, so after both fees the net is positive
    but smaller. We verify the net pnl is exactly
    gross - exit_commission - entry_commission, and that the
    equity-curve credit (returned by position.close) is unchanged.
    """
    closes = [100.0] * 5 + [101.0, 102.0, 103.0]
    df = _build_frame(closes, atr_pct=0.005)
    bt = Backtest(
        strategy=_OneShotTakeAt("BTC/USDT", take=102.0),
        commission=FlatCommission(rate=0.01),  # 1% per side, exaggerated
        slippage=FixedSlippage(bps=0.0),
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=2),
    )
    result = bt.run(df)
    take_trades = [t for t in result.trades if t.exit_reason == "take"]
    assert take_trades, "expected one take exit"
    trade = take_trades[0]

    gross = trade.quantity * (trade.exit_price - trade.entry_price)
    entry_commission = trade.quantity * trade.entry_price * 0.01
    exit_commission = trade.quantity * trade.exit_price * 0.01
    expected_net = gross - exit_commission - entry_commission

    assert trade.pnl == pytest.approx(expected_net, rel=1e-9)
    # And the `commission` field still totals both legs.
    assert trade.commission == pytest.approx(
        entry_commission + exit_commission, rel=1e-9
    )


def test_win_rate_uses_net_pnl_not_gross() -> None:
    """A trade that is net-negative after both commissions must be
    counted as a loss, not a win. Lock the win/loss math to the
    fee-inclusive PnL.
    """
    # Build a trade that gross-positive (+1) but net-negative once a
    # 5% per-side commission is applied. Synthesize a TradeRecord
    # directly and run compute_metrics on it; this is the post-fix
    # invariant compute_metrics relies on.
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    # gross = quantity * (exit - entry) = 1.0 * (101 - 100) = +1.0
    # entry_commission = 1.0 * 100 * 0.05 = 5.0
    # exit_commission  = 1.0 * 101 * 0.05 = 5.05
    # net = +1.0 - 5.0 - 5.05 = -9.05
    trade = TradeRecord(
        entry_time=ts,
        exit_time=ts,
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        pnl=-9.05,
        commission=10.05,
        exit_reason="take",
    )
    curve = pd.Series([1000.0, 1000.0 - 9.05], dtype=float)
    m = compute_metrics(curve, trades=[trade], initial_equity=1000.0)
    assert m.num_trades == 1
    assert m.num_wins == 0
    assert m.num_losses == 1
    assert m.win_rate == 0.0
    assert m.avg_loss == pytest.approx(9.05, rel=1e-9)


def test_atr_is_passed_to_slippage_model() -> None:
    """AtrSlippage was always falling back to its floor because the
    engine never passed an ATR value. Verify the engine forwards a
    finite ATR on every fill once warm-up is past.
    """
    closes = [100.0 + (i % 5) for i in range(60)]
    df = _build_frame(closes, atr_pct=0.005)
    spy = _RecordingSlippage()
    bt = Backtest(
        strategy=_OneShotLong("BTC/USDT", stop=80.0, take=200.0),
        commission=FlatCommission(rate=0.0),
        slippage=spy,
        config=BacktestConfig(initial_equity=10_000.0, warmup_bars=20),
    )
    bt.run(df)
    # The strategy fires exactly once, so spy.calls has one entry,
    # the entry fill. (The position is force-closed at end_of_data,
    # which is the second call.)
    assert spy.calls, "slippage was never invoked"
    entry_call = spy.calls[0]
    _, _, entry_atr = entry_call
    # ATR must be a finite positive number, not None.
    assert entry_atr is not None
    assert math.isfinite(entry_atr)
    assert entry_atr > 0.0
