"""
Tests for the signal_generator.py changes introduced in this PR.

Key changes:
  1. TradingSignal is now imported from engine.strategies.base (the canonical
     location) rather than being defined locally in signal_generator.py.
  2. The amount field was replaced with metadata={"amount_pct": ...} so the
     pipeline contract is compatible with the canonical TradingSignal dataclass.

Tests here verify:
  - TradingSignal is NOT defined in signal_generator (it was removed).
  - The canonical TradingSignal from engine.strategies.base has a metadata field.
  - _parse_json_response() populates metadata["amount_pct"] from the AI payload.
  - When amount_pct is absent from the AI payload, metadata["amount_pct"] is None.
  - Fallback (rule-based) signals have an empty metadata dict (the dataclass default).
  - Both modules reference the exact same TradingSignal class object.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

# pandas is a required dependency; skip the entire module if it is absent
# so the test runner stays green in minimal environments (same pattern used
# by test_mt5_adapter.py for the cryptography package).
pandas = pytest.importorskip("pandas")

import engine.signal_generator as sg_module  # noqa: E402
from engine.strategies.base import TradingSignal  # noqa: E402


# ---------------------------------------------------------------------------
# TradingSignal is NOT defined locally in signal_generator
# ---------------------------------------------------------------------------


def test_trading_signal_not_defined_locally_in_signal_generator() -> None:
    """After the PR, signal_generator.py must NOT own TradingSignal; it imports it."""
    # The module-level dict must not contain a TradingSignal class that is
    # different from the one in engine.strategies.base.
    local_cls = vars(sg_module).get("TradingSignal")
    if local_cls is not None:
        # It may be re-exported (imported name), but it must be the same object.
        assert local_cls is TradingSignal, (
            "signal_generator.TradingSignal must be the canonical class from "
            "engine.strategies.base, not a separate local definition."
        )


def test_signal_generator_module_uses_canonical_trading_signal() -> None:
    """The TradingSignal accessible via signal_generator is the base class."""
    from engine.signal_generator import TradingSignal as sg_ts  # type: ignore[attr-defined]
    assert sg_ts is TradingSignal


# ---------------------------------------------------------------------------
# TradingSignal canonical dataclass – metadata field
# ---------------------------------------------------------------------------


def test_canonical_trading_signal_has_metadata_field() -> None:
    signal = TradingSignal(symbol="EURUSD", action="HOLD", confidence=0.5)
    assert hasattr(signal, "metadata")
    assert isinstance(signal.metadata, dict)


def test_canonical_trading_signal_default_metadata_is_empty_dict() -> None:
    signal = TradingSignal(symbol="BTCUSDT", action="BUY", confidence=0.7)
    assert signal.metadata == {}


def test_canonical_trading_signal_metadata_accepts_amount_pct() -> None:
    signal = TradingSignal(
        symbol="BTCUSDT",
        action="BUY",
        confidence=0.8,
        metadata={"amount_pct": 0.02},
    )
    assert signal.metadata["amount_pct"] == pytest.approx(0.02)


def test_canonical_trading_signal_metadata_is_mutable_dict() -> None:
    signal = TradingSignal(symbol="EURUSD", action="HOLD", confidence=0.0)
    signal.metadata["extra"] = "value"
    assert signal.metadata["extra"] == "value"


# ---------------------------------------------------------------------------
# _parse_json_response – metadata["amount_pct"] population
# ---------------------------------------------------------------------------

# Minimal fake market data (OHLCV): [timestamp, open, high, low, close, volume]
_MARKET_DATA: List[List] = [[1_700_000_000_000, 50_000, 51_000, 49_000, 50_500, 10.0]]


def _make_generator() -> sg_module.SignalGenerator:
    """Return a SignalGenerator without a real Gemini client."""
    gen = sg_module.SignalGenerator.__new__(sg_module.SignalGenerator)
    gen.gemini_client = None
    gen.model = None
    gen.market_context = sg_module.MarketContext()
    gen._last_signals: dict = {}
    return gen


def test_parse_json_response_sets_amount_pct_in_metadata() -> None:
    gen = _make_generator()
    json_text = '{"action":"BUY","confidence":0.75,"amount_pct":0.02,"reasoning":"test"}'
    signal = gen._parse_json_response("BTCUSDT", json_text, _MARKET_DATA)
    assert isinstance(signal, TradingSignal)
    assert signal.metadata.get("amount_pct") == pytest.approx(0.02)


def test_parse_json_response_amount_pct_none_when_missing() -> None:
    gen = _make_generator()
    json_text = '{"action":"SELL","confidence":0.6,"reasoning":"no size given"}'
    signal = gen._parse_json_response("EURUSD", json_text, _MARKET_DATA)
    assert isinstance(signal, TradingSignal)
    # amount_pct was not in the payload → should be None (data.get("amount_pct"))
    assert signal.metadata.get("amount_pct") is None


def test_parse_json_response_amount_pct_null_in_payload() -> None:
    gen = _make_generator()
    json_text = '{"action":"BUY","confidence":0.5,"amount_pct":null}'
    signal = gen._parse_json_response("BTCUSDT", json_text, _MARKET_DATA)
    assert isinstance(signal, TradingSignal)
    assert signal.metadata.get("amount_pct") is None


def test_parse_json_response_amount_pct_not_in_top_level_amount_field() -> None:
    """The old code used signal.amount; the new code stores it in metadata.
    Ensure the old 'amount' field is NOT the attribute we're checking."""
    gen = _make_generator()
    json_text = '{"action":"BUY","confidence":0.7,"amount_pct":0.03}'
    signal = gen._parse_json_response("EURUSD", json_text, _MARKET_DATA)
    # The canonical TradingSignal has no 'amount' field; amount_pct must live in metadata.
    assert not hasattr(signal, "amount"), (
        "TradingSignal must not have an 'amount' field; use metadata['amount_pct']"
    )
    assert signal.metadata["amount_pct"] == pytest.approx(0.03)


def test_parse_json_response_fallback_on_invalid_json() -> None:
    gen = _make_generator()
    signal = gen._parse_json_response("BTCUSDT", "NOT_VALID_JSON", _MARKET_DATA)
    assert isinstance(signal, TradingSignal)
    assert signal.action == "HOLD"
    # Fallback signal has default empty metadata
    assert signal.metadata == {}


def test_parse_json_response_action_is_uppercased() -> None:
    gen = _make_generator()
    json_text = '{"action":"buy","confidence":0.6,"amount_pct":0.01}'
    signal = gen._parse_json_response("EURUSD", json_text, _MARKET_DATA)
    assert signal.action == "BUY"


def test_parse_json_response_sets_correct_symbol() -> None:
    gen = _make_generator()
    json_text = '{"action":"HOLD","confidence":0.5}'
    signal = gen._parse_json_response("USDJPY", json_text, _MARKET_DATA)
    assert signal.symbol == "USDJPY"


def test_parse_json_response_uses_entry_price_from_market_data_when_missing() -> None:
    gen = _make_generator()
    json_text = '{"action":"BUY","confidence":0.7}'
    # Last OHLCV close is at index [4] of the last candle.
    signal = gen._parse_json_response("BTCUSDT", json_text, _MARKET_DATA)
    assert signal.entry_price == pytest.approx(_MARKET_DATA[-1][4])


def test_parse_json_response_stop_loss_and_take_profit_passthrough() -> None:
    gen = _make_generator()
    json_text = (
        '{"action":"SELL","confidence":0.8,'
        '"stop_loss":51000.0,"take_profit":48000.0}'
    )
    signal = gen._parse_json_response("BTCUSDT", json_text, _MARKET_DATA)
    assert signal.stop_loss == pytest.approx(51_000.0)
    assert signal.take_profit == pytest.approx(48_000.0)


# ---------------------------------------------------------------------------
# TradingSignal is the same object in both modules (import identity)
# ---------------------------------------------------------------------------


def test_trading_signal_class_identity_across_modules() -> None:
    from engine.strategies.base import TradingSignal as base_ts
    # Importing through signal_generator should give the same class.
    from engine.signal_generator import TradingSignal as sg_ts  # type: ignore[attr-defined]
    assert sg_ts is base_ts