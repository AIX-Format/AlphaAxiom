"""
Single-position backtest framework.

Given a `Strategy`, an OHLCV DataFrame, a commission model, and a
slippage model, `Backtest.run()` walks bar by bar and simulates what
would have happened if the strategy had been live.

The model is deliberately simple:

- One open position at a time (long or short). The risk shield and
  position-sizing layers from Task 1.3 / 1.4 keep size honest; this
  layer just runs the strategy.
- Signals are generated on the close of bar `t` and entered on the
  open of bar `t + 1`, with slippage applied. This avoids the
  classic look-ahead bias of entering on the same bar the signal
  was generated on.
- During the holding period each bar's high/low is checked against
  the stop and take levels. If both could fire in the same bar (high
  hits take AND low hits stop) we assume the stop fired first, which
  is the conservative choice and matches how most realistic backtests
  handle the unknown intra-bar path.
- An opposite signal closes the current position and immediately
  flips into the new one on the next bar.

Outputs:

- `equity_curve`: pandas Series indexed by the input DataFrame's
  index. Starts at `initial_equity`.
- `trades`: list of `TradeRecord` dicts with entry / exit / pnl / etc.
- `metrics`: `BacktestMetrics` with Sharpe, max drawdown, win rate,
  total return, and trade counts.

The Sharpe calculation annualises by the inferred bar frequency. For
hourly data the factor is `sqrt(24 * 365)`. The caller can override
`periods_per_year` if the inference is wrong (e.g. trading hours only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from .indicators import atr as _atr_series
from .strategies.base import REQUIRED_COLUMNS, Strategy, TradingSignal


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


class SlippageModel(Protocol):
    """Apply slippage to a fill price.

    `side` is `"buy"` for longs entering / shorts exiting (we pay the
    ask) and `"sell"` for the mirror. `atr` is passed through when the
    model wants to scale slippage to volatility; models that don't
    care should ignore it.
    """

    def apply(self, price: float, side: str, atr: Optional[float] = None) -> float:
        ...


@dataclass
class FixedSlippage:
    """Constant bps slippage (default 10 bps = 0.10%).

    Long entries fill higher than the printed price; short entries
    fill lower; exits the opposite. The factor is the same on every
    bar regardless of volatility.
    """

    bps: float = 10.0

    def apply(self, price: float, side: str, atr: Optional[float] = None) -> float:
        factor = self.bps / 10_000.0
        if side == "buy":
            return price * (1.0 + factor)
        return price * (1.0 - factor)


@dataclass
class AtrSlippage:
    """Volatility-aware slippage: `multiplier * ATR` added to the fill.

    Useful when modeling thin or volatile markets where the realised
    slippage scales with bar range. Falls back to a small floor when
    ATR is missing so the model never returns the raw printed price
    silently.
    """

    multiplier: float = 0.1
    floor_bps: float = 1.0

    def apply(self, price: float, side: str, atr: Optional[float] = None) -> float:
        atr_slip = self.multiplier * float(atr) if atr and math.isfinite(atr) else 0.0
        floor_slip = price * (self.floor_bps / 10_000.0)
        total = max(atr_slip, floor_slip)
        if side == "buy":
            return price + total
        return price - total


class CommissionModel(Protocol):
    """Return the commission cost (always positive) for a notional fill."""

    def apply(self, notional: float) -> float:
        ...


@dataclass
class FlatCommission:
    """Constant per-side rate, e.g. 0.001 for 10 bps (Binance taker)."""

    rate: float = 0.001

    def apply(self, notional: float) -> float:
        return abs(float(notional)) * float(self.rate)


#: Common presets. Update these alongside the exchange fee schedules.
BINANCE_SPOT_TAKER = FlatCommission(rate=0.001)
BYBIT_PERP_TAKER = FlatCommission(rate=0.00055)


# ---------------------------------------------------------------------------
# Trade records and metrics
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """One round-trip trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    exit_reason: str  # "stop", "take", "signal_flip", "end_of_data"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "commission": self.commission,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestMetrics:
    """Summary statistics for a completed backtest."""

    total_return: float          # final / initial - 1
    sharpe: float                # annualised Sharpe ratio
    max_drawdown: float          # most negative peak-to-trough drop, fraction
    win_rate: float              # winning trades / total trades, in [0, 1]
    num_trades: int
    num_wins: int
    num_losses: int
    avg_win: float
    avg_loss: float              # positive number, the magnitude of an avg loss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[TradeRecord]
    metrics: BacktestMetrics

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    fallback_stop_pct: float = 0.02
    #: Override the inferred bar frequency. Hourly: 24*365=8760.
    #: Daily: 252. None means infer from the index.
    periods_per_year: Optional[float] = None
    #: Minimum number of bars the strategy needs before we start
    #: asking it for signals. Acts as a guard against the warm-up
    #: window producing noise.
    warmup_bars: int = 50


class Backtest:
    """Walk-forward, single-position simulator."""

    def __init__(
        self,
        strategy: Strategy,
        commission: Optional[CommissionModel] = None,
        slippage: Optional[SlippageModel] = None,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        # Defaults are materialised here so the public signature stays
        # None-friendly and we never share mutable default instances.
        self.strategy = strategy
        self.commission = commission if commission is not None else BINANCE_SPOT_TAKER
        self.slippage = slippage if slippage is not None else FixedSlippage()
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        _validate_frame(df)

        n = len(df)
        equity = self.config.initial_equity
        equity_curve = np.full(n, equity, dtype=float)

        trades: List[TradeRecord] = []
        position: Optional[_OpenPosition] = None

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        index = df.index

        # Precompute ATR once so AtrSlippage actually sees volatility
        # instead of always falling back to its floor. The 14-bar
        # default lines up with what the strategies use for their own
        # stops, so the slippage model and the strategy share a
        # consistent volatility view.
        try:
            atr_values = _atr_series(
                df["high"], df["low"], df["close"], period=14
            ).to_numpy(dtype=float)
        except Exception:
            atr_values = np.full(n, np.nan, dtype=float)

        def _bar_atr(i: int) -> Optional[float]:
            if i < 0 or i >= n:
                return None
            v = atr_values[i]
            return float(v) if math.isfinite(v) else None

        for t in range(n):
            # 1) Mark-to-market any open position at this bar's close.
            #    Also check whether stop/take fired intra-bar.
            if position is not None:
                exit_info = position.check_intrabar(highs[t], lows[t])
                if exit_info is not None:
                    trigger_price, reason = exit_info
                    # Stop/take fills experience slippage just like any
                    # other order. Longs sell to exit (slip down); shorts
                    # buy to exit (slip up).
                    exit_side = "sell" if position.direction == "long" else "buy"
                    exit_price = self.slippage.apply(
                        trigger_price, exit_side, atr=_bar_atr(t)
                    )
                    realised = position.close(
                        exit_price, index[t], reason, self.commission
                    )
                    equity += realised
                    trades.append(position.record)
                    position = None

            # 2) Record equity at this bar's close.
            mark = equity
            if position is not None:
                mark += position.unrealised(closes[t])
            equity_curve[t] = mark

            # 3) Ask the strategy for a fresh signal once we have enough
            #    warm-up. The frame we hand the strategy ends at bar t
            #    inclusive; entry will be on bar t+1's open below.
            if t < max(self.config.warmup_bars - 1, 0) or t == n - 1:
                continue

            sub = df.iloc[: t + 1]
            try:
                signal = self.strategy.generate_signal(sub)
            except Exception:
                # The strategy contract is "never raise"; treat any
                # escape as HOLD instead of crashing the backtest.
                continue

            if not signal.is_actionable():
                continue

            next_atr = _bar_atr(t + 1)

            # 4) Apply the signal on the NEXT bar's open. If we have an
            #    open position in the opposite direction, close it first.
            if position is not None and not _same_direction(position, signal):
                close_open = opens[t + 1]
                slipped = self.slippage.apply(
                    close_open,
                    "sell" if position.direction == "long" else "buy",
                    atr=next_atr,
                )
                realised = position.close(
                    slipped, index[t + 1], "signal_flip", self.commission
                )
                equity += realised
                trades.append(position.record)
                position = None

            # If after the close we are flat, open a fresh position.
            if position is None:
                size_notional = self._compute_size(signal, equity)
                if size_notional <= 0 or not math.isfinite(size_notional):
                    continue
                fill_side = "buy" if signal.action == "BUY" else "sell"
                fill_price = self.slippage.apply(
                    opens[t + 1], fill_side, atr=next_atr
                )
                qty = size_notional / fill_price if fill_price > 0 else 0.0
                if qty <= 0 or not math.isfinite(qty):
                    continue
                entry_commission = self.commission.apply(size_notional)
                equity -= entry_commission
                position = _OpenPosition(
                    direction="long" if signal.action == "BUY" else "short",
                    entry_time=index[t + 1],
                    entry_price=fill_price,
                    quantity=qty,
                    stop=signal.stop_loss,
                    take=signal.take_profit,
                    entry_commission=entry_commission,
                )

        # 5) Force-close anything still open at the final bar's close.
        if position is not None:
            final_atr = _bar_atr(n - 1)
            exit_side = "sell" if position.direction == "long" else "buy"
            slipped_close = self.slippage.apply(
                closes[-1], exit_side, atr=final_atr
            )
            realised = position.close(
                slipped_close, index[-1], "end_of_data", self.commission
            )
            equity += realised
            trades.append(position.record)
            # Update final equity mark.
            equity_curve[-1] = equity

        eq_series = pd.Series(equity_curve, index=index, name="equity")
        metrics = compute_metrics(
            eq_series,
            trades,
            initial_equity=self.config.initial_equity,
            periods_per_year=self.config.periods_per_year,
        )
        return BacktestResult(equity_curve=eq_series, trades=trades, metrics=metrics)

    def _compute_size(self, signal: TradingSignal, equity: float) -> float:
        # NaN sneaks past plain `<= 0` and `if entry and stop` checks
        # (NaN is truthy) so we coerce and explicitly reject any
        # non-finite value. Without this a strategy that emits a
        # NaN stop would contaminate the equity curve with NaN.
        if not math.isfinite(equity) or equity <= 0:
            return 0.0

        try:
            entry = float(signal.entry_price) if signal.entry_price is not None else None
            stop = float(signal.stop_loss) if signal.stop_loss is not None else None
        except (TypeError, ValueError):
            entry, stop = None, None

        if (
            entry is not None
            and stop is not None
            and math.isfinite(entry)
            and math.isfinite(stop)
            and entry > 0
        ):
            stop_pct = abs(entry - stop) / entry
            if not math.isfinite(stop_pct) or stop_pct <= 0:
                stop_pct = self.config.fallback_stop_pct
        else:
            stop_pct = self.config.fallback_stop_pct

        size = equity * self.config.risk_per_trade_pct / stop_pct
        return size if math.isfinite(size) else 0.0


# ---------------------------------------------------------------------------
# Open-position bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    direction: str  # "long" | "short"
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop: Optional[float]
    take: Optional[float]
    entry_commission: float

    # Filled in when the position closes.
    record: Optional[TradeRecord] = None

    def check_intrabar(
        self, high: float, low: float
    ) -> Optional[tuple]:
        """Return (exit_price, reason) if stop/take fires this bar."""
        if self.direction == "long":
            # Conservative: if both could fire, the stop wins.
            if self.stop is not None and low <= self.stop:
                return (float(self.stop), "stop")
            if self.take is not None and high >= self.take:
                return (float(self.take), "take")
        else:
            if self.stop is not None and high >= self.stop:
                return (float(self.stop), "stop")
            if self.take is not None and low <= self.take:
                return (float(self.take), "take")
        return None

    def unrealised(self, mark_price: float) -> float:
        if self.direction == "long":
            return self.quantity * (mark_price - self.entry_price)
        return self.quantity * (self.entry_price - mark_price)

    def close(
        self,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
        commission_model: CommissionModel,
    ) -> float:
        """Close the position and stamp the TradeRecord.

        Returns the value to credit back to equity, which is
        `gross - exit_commission`. The entry commission was already
        debited from equity when the position opened, so we must NOT
        return it again here or equity would be double-charged.

        TradeRecord.pnl, by contrast, is the true round-trip net,
        `gross - exit_commission - entry_commission`. compute_metrics
        classifies wins and losses by `t.pnl`, so it has to see the
        fee-inclusive number; otherwise a small gross gain that does
        not cover both legs of commission would still register as a
        win and skew win-rate and avg-win/avg-loss.
        """
        gross = self.unrealised(exit_price)
        exit_notional = self.quantity * exit_price
        exit_commission = commission_model.apply(exit_notional)
        realised_to_equity = gross - exit_commission
        net_pnl = realised_to_equity - self.entry_commission
        self.record = TradeRecord(
            entry_time=self.entry_time,
            exit_time=exit_time,
            direction=self.direction,
            entry_price=self.entry_price,
            exit_price=exit_price,
            quantity=self.quantity,
            pnl=net_pnl,
            commission=self.entry_commission + exit_commission,
            exit_reason=reason,
        )
        return realised_to_equity


def _same_direction(position: _OpenPosition, signal: TradingSignal) -> bool:
    if position.direction == "long" and signal.action == "BUY":
        return True
    if position.direction == "short" and signal.action == "SELL":
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_frame(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Backtest requires a non-empty OHLCV DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest frame missing columns: {missing}")


def _infer_periods_per_year(index: pd.Index) -> float:
    """Best-effort guess at the annualisation factor.

    Looks at the median bar spacing and maps to common defaults
    (hourly, daily, minutely). Falls back to 252 (daily) when the
    index is not a DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 252.0
    median_seconds = float(diffs.median().total_seconds())
    if median_seconds <= 0:
        return 252.0
    # Bars per year, treating a year as 365.25 days.
    return (365.25 * 24 * 3600.0) / median_seconds


def compute_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    *,
    initial_equity: float,
    periods_per_year: Optional[float] = None,
) -> BacktestMetrics:
    """Standalone metric computation: exposed for tests and reuse."""
    if equity_curve.empty:
        return BacktestMetrics(
            total_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            num_wins=0,
            num_losses=0,
            avg_win=0.0,
            avg_loss=0.0,
        )

    final = float(equity_curve.iloc[-1])
    total_return = (final / initial_equity) - 1.0 if initial_equity > 0 else 0.0

    # Per-bar returns from the equity curve.
    returns = equity_curve.pct_change().dropna()
    ann = periods_per_year or _infer_periods_per_year(equity_curve.index)
    if len(returns) > 1 and returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(ann))
    else:
        sharpe = 0.0

    # Max drawdown from the running peak.
    running_peak = equity_curve.cummax()
    drawdowns = (equity_curve - running_peak) / running_peak.where(running_peak > 0, np.nan)
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
    if math.isnan(max_dd):
        max_dd = 0.0
    max_dd = abs(min(max_dd, 0.0))

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    num_trades = len(trades)
    win_rate = (len(wins) / num_trades) if num_trades else 0.0
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([-t.pnl for t in losses])) if losses else 0.0

    return BacktestMetrics(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
        num_wins=len(wins),
        num_losses=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
