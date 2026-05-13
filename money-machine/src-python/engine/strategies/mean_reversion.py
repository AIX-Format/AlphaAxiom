"""
Mean reversion strategy: RSI extremes confirmed by Bollinger Bands.

Premise: assets oscillate around a moving average. When price stretches
past the upper Bollinger Band AND the RSI shows overbought, the
strategy expects a pullback and emits SELL. The symmetric setup at
the lower band with oversold RSI emits BUY. Anything else is HOLD.

Default parameters (TradingView-conventional):

- RSI period 14, oversold below 30, overbought above 70.
- Bollinger Bands period 20, num_std 2.0.

The stop-loss sits one ATR beyond the touched band (giving the trade
room to breathe past the noise) and the take-profit aims back at the
Bollinger mid-band (the natural mean-reversion target). ATR period
follows the RSI default of 14 so the warm-ups line up.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, bollinger_bands, rsi
from .base import Strategy, TradingSignal


class MeanReversionStrategy(Strategy):
    name = "mean-reversion-v1"

    def __init__(
        self,
        symbol: str,
        *,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_num_std: float = 2.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.0,
    ) -> None:
        super().__init__(symbol)
        for name, value in (
            ("rsi_period", rsi_period),
            ("bb_period", bb_period),
            ("atr_period", atr_period),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be int >= 1, got {value!r}")
        if bb_num_std <= 0:
            raise ValueError(f"bb_num_std must be > 0, got {bb_num_std!r}")
        if atr_stop_mult <= 0:
            raise ValueError(f"atr_stop_mult must be > 0, got {atr_stop_mult!r}")
        if not 0.0 <= rsi_oversold <= rsi_overbought <= 100.0:
            raise ValueError(
                "RSI thresholds must satisfy 0 <= oversold <= overbought <= 100, "
                f"got oversold={rsi_oversold}, overbought={rsi_overbought}"
            )
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_num_std = bb_num_std
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        min_rows = max(self.bb_period, self.rsi_period, self.atr_period) + 1
        err = self._validate_frame(df, min_rows=min_rows)
        if err:
            return self._hold(err)

        close = df["close"]
        try:
            upper, middle, lower = bollinger_bands(
                close, period=self.bb_period, num_std=self.bb_num_std
            )
            rsi_series = rsi(close, period=self.rsi_period)
            atr_series = atr(
                df["high"], df["low"], close, period=self.atr_period
            )
        except ValueError as exc:
            return self._hold(f"invalid indicator parameters: {exc}")

        last_close = float(close.iloc[-1])
        last_upper = upper.iloc[-1]
        last_middle = middle.iloc[-1]
        last_lower = lower.iloc[-1]
        last_rsi = rsi_series.iloc[-1]
        last_atr = atr_series.iloc[-1]

        if any(
            pd.isna(x)
            for x in (last_upper, last_middle, last_lower, last_rsi, last_atr)
        ):
            return self._hold("indicator warm-up incomplete", price=last_close)

        # Oversold + below lower band -> expect pull back up.
        if last_close <= last_lower and last_rsi <= self.rsi_oversold:
            stop = last_close - self.atr_stop_mult * float(last_atr)
            take = float(last_middle)
            confidence = self._confidence(
                rsi_distance=self.rsi_oversold - float(last_rsi),
                band_distance=float(last_lower) - last_close,
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
                    f"price {last_close:.2f} <= lower band {last_lower:.2f} and "
                    f"RSI {last_rsi:.1f} <= {self.rsi_oversold}"
                ),
                metadata={
                    "rsi": float(last_rsi),
                    "bb_upper": float(last_upper),
                    "bb_middle": float(last_middle),
                    "bb_lower": float(last_lower),
                    "atr": float(last_atr),
                },
            )

        # Overbought + above upper band -> expect pull back down.
        if last_close >= last_upper and last_rsi >= self.rsi_overbought:
            stop = last_close + self.atr_stop_mult * float(last_atr)
            take = float(last_middle)
            confidence = self._confidence(
                rsi_distance=float(last_rsi) - self.rsi_overbought,
                band_distance=last_close - float(last_upper),
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
                    f"price {last_close:.2f} >= upper band {last_upper:.2f} and "
                    f"RSI {last_rsi:.1f} >= {self.rsi_overbought}"
                ),
                metadata={
                    "rsi": float(last_rsi),
                    "bb_upper": float(last_upper),
                    "bb_middle": float(last_middle),
                    "bb_lower": float(last_lower),
                    "atr": float(last_atr),
                },
            )

        return self._hold(
            f"no extreme: close {last_close:.2f}, RSI {last_rsi:.1f}",
            price=last_close,
        )

    @staticmethod
    def _confidence(
        *, rsi_distance: float, band_distance: float, price: float
    ) -> float:
        """Confidence in [0.1, 0.9] based on how stretched the move is.

        Both signals contribute: RSI distance past the threshold and
        the band penetration as a fraction of price. Capped to keep
        downstream sizing sane.
        """
        rsi_score = min(max(rsi_distance, 0.0) / 20.0, 1.0)  # 20-point spread
        band_score = min(max(band_distance, 0.0) / max(price, 1e-9) * 100.0, 1.0)
        score = 0.6 * rsi_score + 0.4 * band_score
        return float(min(max(0.1 + 0.8 * score, 0.1), 0.9))
