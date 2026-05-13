"""
Volatility breakout strategy: ATR-scaled break of the recent range.

Premise: a sustained move beyond the recent N-bar high or low, by at
least `breakout_atr_mult` ATRs, is evidence the regime has shifted.
The strategy emits BUY on a new high break and SELL on a new low
break. The ATR multiple keeps quiet ranges from triggering on noise.

Default parameters:

- Lookback window 20 bars (excluding the current bar).
- ATR period 14, breakout threshold 0.5 ATR.
- Stop-loss = entry -/+ `atr_stop_mult` * ATR (1.5 default).
- Take-profit = entry +/- `atr_target_mult` * ATR (3.0 default,
  i.e. 2:1 reward-to-risk).

This is the simplest of the three strategies on purpose: it does not
use a trend filter, which makes it the natural counter-balance to
mean reversion on the same data.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr
from .base import Strategy, TradingSignal


class BreakoutStrategy(Strategy):
    name = "breakout-v1"

    def __init__(
        self,
        symbol: str,
        *,
        lookback: int = 20,
        atr_period: int = 14,
        breakout_atr_mult: float = 0.5,
        atr_stop_mult: float = 1.5,
        atr_target_mult: float = 3.0,
    ) -> None:
        super().__init__(symbol)
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        self.lookback = lookback
        self.atr_period = atr_period
        self.breakout_atr_mult = breakout_atr_mult
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        min_rows = max(self.lookback + 1, self.atr_period + 1)
        err = self._validate_frame(df, min_rows=min_rows)
        if err:
            return self._hold(err)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        atr_series = atr(high, low, close, period=self.atr_period)
        last_atr = atr_series.iloc[-1]
        last_close = float(close.iloc[-1])

        if pd.isna(last_atr):
            return self._hold("ATR warm-up incomplete", price=last_close)

        # Range over the prior `lookback` bars, excluding the current
        # bar so we are not comparing the close against itself.
        prior = df.iloc[-(self.lookback + 1):-1]
        range_high = float(prior["high"].max())
        range_low = float(prior["low"].min())

        threshold = self.breakout_atr_mult * float(last_atr)

        if last_close > range_high + threshold:
            stop = last_close - self.atr_stop_mult * float(last_atr)
            take = last_close + self.atr_target_mult * float(last_atr)
            confidence = self._confidence(
                gap=last_close - range_high,
                atr=float(last_atr),
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
                    f"close {last_close:.2f} > {self.lookback}-bar high "
                    f"{range_high:.2f} by more than {threshold:.2f} ATR units"
                ),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "atr": float(last_atr),
                },
            )

        if last_close < range_low - threshold:
            stop = last_close + self.atr_stop_mult * float(last_atr)
            take = last_close - self.atr_target_mult * float(last_atr)
            confidence = self._confidence(
                gap=range_low - last_close,
                atr=float(last_atr),
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
                    f"close {last_close:.2f} < {self.lookback}-bar low "
                    f"{range_low:.2f} by more than {threshold:.2f} ATR units"
                ),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "atr": float(last_atr),
                },
            )

        return self._hold(
            f"no breakout: close {last_close:.2f} within "
            f"[{range_low:.2f}, {range_high:.2f}]",
            price=last_close,
        )

    @staticmethod
    def _confidence(*, gap: float, atr: float) -> float:
        """Confidence rises with the size of the break in ATR units."""
        if atr <= 0:
            return 0.2
        atr_multiples = max(gap, 0.0) / atr
        # 2 ATR break -> ~0.9, 0.5 ATR break -> ~0.35.
        score = min(atr_multiples / 2.0, 1.0)
        return float(min(max(0.2 + 0.7 * score, 0.2), 0.9))
