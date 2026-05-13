"""
Momentum strategy: MACD crossover gated by an EMA trend filter.

Premise: trends persist. A bullish MACD crossover (`macd_line` crosses
above `signal_line`) is taken as a BUY when the longer-term EMA shows
the market is also trending up (`close > ema(close, trend_period)`).
The mirror setup produces SELL. Without trend agreement the
crossover is ignored.

Default parameters (TradingView-conventional):

- MACD 12 / 26 / 9.
- Trend EMA period 50.
- Stop-loss = entry - ATR * `atr_stop_mult` (for BUY) or
  entry + ATR * `atr_stop_mult` (for SELL).
- Take-profit = entry +/- ATR * `atr_target_mult` so the default
  reward-to-risk is `atr_target_mult / atr_stop_mult` (= 2 by
  default).
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, ema, macd
from .base import Strategy, TradingSignal


class MomentumStrategy(Strategy):
    name = "momentum-v1"

    def __init__(
        self,
        symbol: str,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        trend_period: int = 50,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        atr_target_mult: float = 3.0,
    ) -> None:
        super().__init__(symbol)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        # Need slow + signal warm-up plus the trend EMA window. Add a
        # cushion so the crossover detection has the previous bar too.
        min_rows = self.slow_period + self.signal_period + 2
        min_rows = max(min_rows, self.trend_period + 2)
        err = self._validate_frame(df, min_rows=min_rows)
        if err:
            return self._hold(err)

        close = df["close"]
        macd_line, signal_line, _hist = macd(
            close,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
        )
        trend = ema(close, self.trend_period)
        atr_series = atr(df["high"], df["low"], close, period=self.atr_period)

        last_close = float(close.iloc[-1])
        last_trend = trend.iloc[-1]
        last_atr = atr_series.iloc[-1]
        last_macd = macd_line.iloc[-1]
        last_sig = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal_line.iloc[-2]

        if any(
            pd.isna(x)
            for x in (last_trend, last_atr, last_macd, last_sig, prev_macd, prev_sig)
        ):
            return self._hold("indicator warm-up incomplete", price=last_close)

        bullish_cross = prev_macd <= prev_sig and last_macd > last_sig
        bearish_cross = prev_macd >= prev_sig and last_macd < last_sig
        uptrend = last_close > float(last_trend)
        downtrend = last_close < float(last_trend)

        if bullish_cross and uptrend:
            stop = last_close - self.atr_stop_mult * float(last_atr)
            take = last_close + self.atr_target_mult * float(last_atr)
            confidence = self._confidence(
                cross_gap=float(last_macd - last_sig),
                trend_gap=last_close - float(last_trend),
                price=last_close,
            )
            return TradingSignal(
                symbol=self.symbol,
                action="BUY",
                confidence=confidence,
                strategy=self.name,
                entry_price=last_close,
                stop_loss=stop,
                take_profit=take,
                reasoning=(
                    f"bullish MACD crossover ({last_macd:.2f} > {last_sig:.2f}) "
                    f"with close {last_close:.2f} above trend EMA {last_trend:.2f}"
                ),
                metadata={
                    "macd": float(last_macd),
                    "macd_signal": float(last_sig),
                    "trend_ema": float(last_trend),
                    "atr": float(last_atr),
                },
            )

        if bearish_cross and downtrend:
            stop = last_close + self.atr_stop_mult * float(last_atr)
            take = last_close - self.atr_target_mult * float(last_atr)
            confidence = self._confidence(
                cross_gap=float(last_sig - last_macd),
                trend_gap=float(last_trend) - last_close,
                price=last_close,
            )
            return TradingSignal(
                symbol=self.symbol,
                action="SELL",
                confidence=confidence,
                strategy=self.name,
                entry_price=last_close,
                stop_loss=stop,
                take_profit=take,
                reasoning=(
                    f"bearish MACD crossover ({last_macd:.2f} < {last_sig:.2f}) "
                    f"with close {last_close:.2f} below trend EMA {last_trend:.2f}"
                ),
                metadata={
                    "macd": float(last_macd),
                    "macd_signal": float(last_sig),
                    "trend_ema": float(last_trend),
                    "atr": float(last_atr),
                },
            )

        return self._hold(
            f"no aligned crossover: macd {last_macd:.2f} vs sig {last_sig:.2f}",
            price=last_close,
        )

    @staticmethod
    def _confidence(*, cross_gap: float, trend_gap: float, price: float) -> float:
        # Normalise gaps as a fraction of price. Tight gaps -> low
        # conviction; wide gaps with strong trend -> high conviction.
        cross_score = min(max(cross_gap, 0.0) / max(price, 1e-9) * 200.0, 1.0)
        trend_score = min(max(trend_gap, 0.0) / max(price, 1e-9) * 50.0, 1.0)
        score = 0.5 * cross_score + 0.5 * trend_score
        return float(min(max(0.2 + 0.7 * score, 0.2), 0.9))
