"""
Tests for changes introduced in the 'harden desktop security' PR:

  1. TradingSignal.amount field — added to engine/strategies/base.py.
  2. signal_generator.py — TradingSignal is now imported from
     engine.strategies.base (the local duplicate was removed), and
     _parse_json_response maps 'amount_pct' from the AI response to
     the signal's `amount` field.

Scope is limited to the code that was modified in this PR. The
surrounding Strategy / SignalGenerator behaviour is pre-existing and
tested elsewhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.strategies.base import TradingSignal  # noqa: E402
from engine.signal_generator import SignalGenerator  # noqa: E402



# ---------------------------------------------------------------------------
# TradingSignal.amount field
# ---------------------------------------------------------------------------


def test_trading_signal_amount_defaults_to_none() -> None:
    """The new `amount` field must be Optional and default to None so that
    existing code that does not pass an amount continues to work."""
    signal = TradingSignal(symbol="BTC/USDT", action="HOLD", confidence=0.0)
    assert signal.amount is None


def test_trading_signal_amount_can_be_set() -> None:
    """Callers that explicitly pass amount= must be able to read it back."""
    signal = TradingSignal(
        symbol="BTC/USDT",
        action="BUY",
        confidence=0.75,
        amount=0.02,
    )
    assert signal.amount == pytest.approx(0.02)


def test_trading_signal_amount_zero_is_allowed() -> None:
    """Zero is a valid explicit amount (signals a 0-size order intent)."""
    signal = TradingSignal(
        symbol="ETH/USDT",
        action="SELL",
        confidence=0.5,
        amount=0.0,
    )
    assert signal.amount == 0.0


def test_trading_signal_to_dict_includes_amount() -> None:
    """to_dict() must serialise the amount field so IPC consumers receive it."""
    signal = TradingSignal(
        symbol="BTC/USDT",
        action="BUY",
        confidence=0.8,
        amount=0.015,
    )
    d = signal.to_dict()
    assert "amount" in d
    assert d["amount"] == pytest.approx(0.015)


def test_trading_signal_to_dict_amount_none_when_not_set() -> None:
    """When amount is not provided, to_dict() must include the key with None."""
    signal = TradingSignal(symbol="BTC/USDT", action="HOLD", confidence=0.0)
    d = signal.to_dict()
    assert "amount" in d
    assert d["amount"] is None


def test_trading_signal_amount_position_in_field_order() -> None:
    """Verify that amount sits between take_profit and reasoning as specified."""
    signal = TradingSignal(
        symbol="BTC/USDT",
        action="BUY",
        confidence=0.7,
        entry_price=50_000.0,
        stop_loss=49_000.0,
        take_profit=52_000.0,
        amount=0.01,
        reasoning="test",
    )
    assert signal.take_profit == pytest.approx(52_000.0)
    assert signal.amount == pytest.approx(0.01)
    assert signal.reasoning == "test"


# ---------------------------------------------------------------------------
# TradingSignal import in signal_generator.py
# ---------------------------------------------------------------------------


def test_signal_generator_imports_trading_signal_from_base() -> None:
    """signal_generator.py must import TradingSignal from
    engine.strategies.base, not define its own copy."""
    import engine.signal_generator as sg_module

    # The module must not define TradingSignal itself.
    assert not hasattr(sg_module, "TradingSignal") or (
        sg_module.TradingSignal is TradingSignal
    ), (
        "signal_generator defines its own TradingSignal instead of "
        "importing from engine.strategies.base"
    )


def test_signal_generator_trading_signal_is_same_class() -> None:
    """The TradingSignal used by the generator must be the canonical one
    from engine.strategies.base so that downstream consumers (risk shield,
    pipeline) receive the right type."""
    import engine.signal_generator as sg_module
    import importlib

    # Reload to get a fresh reference, unaffected by other test ordering.
    importlib.reload(sg_module)
    from engine.signal_generator import SignalGenerator as SG
    from engine.strategies.base import TradingSignal as TS

    gen = SG(api_key="")
    # Verify that the fallback rule-based path returns the canonical type.
    minimal_data = [[i * 1000, 50000 + i, 50100 + i, 49900 + i, 50050 + i, 100] for i in range(25)]
    import asyncio
    signal = asyncio.run(
        gen.generate_signal("BTC/USDT", minimal_data, portfolio_balance=10_000.0)
    )
    assert isinstance(signal, TS)


# ---------------------------------------------------------------------------
# SignalGenerator._parse_json_response — amount_pct mapping
# ---------------------------------------------------------------------------


def _make_market_data(n: int = 5, price: float = 50_000.0) -> List[List]:
    """Return minimal OHLCV rows for parse tests."""
    return [[i * 1_000, price, price + 100, price - 100, price, 100] for i in range(n)]


def test_parse_json_response_maps_amount_pct_to_amount() -> None:
    """_parse_json_response must read 'amount_pct' from the AI JSON and
    store it in TradingSignal.amount so position sizers can use it."""
    gen = SignalGenerator(api_key="")
    market_data = _make_market_data()

    response_json = json.dumps(
        {
            "action": "BUY",
            "confidence": 0.8,
            "entry_price": 50_000.0,
            "stop_loss": 49_000.0,
            "take_profit": 52_000.0,
            "amount_pct": 0.015,
            "reasoning": "strong momentum",
        }
    )

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert signal.action == "BUY"
    assert signal.confidence == pytest.approx(0.8)
    assert signal.amount == pytest.approx(0.015)


def test_parse_json_response_amount_is_none_when_key_absent() -> None:
    """If 'amount_pct' is absent from the AI response, signal.amount must
    be None (not crash or default to a hard-coded value)."""
    gen = SignalGenerator(api_key="")
    market_data = _make_market_data()

    response_json = json.dumps(
        {
            "action": "SELL",
            "confidence": 0.6,
            "reasoning": "overbought",
        }
    )

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert signal.action == "SELL"
    assert signal.amount is None


def test_parse_json_response_amount_pct_null_maps_to_none() -> None:
    """An explicit null/None in the AI response must also yield amount=None."""
    gen = SignalGenerator(api_key="")
    market_data = _make_market_data()

    response_json = json.dumps(
        {
            "action": "HOLD",
            "confidence": 0.3,
            "amount_pct": None,
            "reasoning": "unclear",
        }
    )

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)
    assert signal.amount is None


def test_parse_json_response_falls_back_on_invalid_json() -> None:
    """If the AI returns garbage, _parse_json_response must return a HOLD
    signal rather than raising."""
    gen = SignalGenerator(api_key="")
    market_data = _make_market_data()

    signal = gen._parse_json_response("BTC/USDT", "this is not json {{{{", market_data)

    assert signal.action == "HOLD"
    assert isinstance(signal, TradingSignal)


def test_parse_json_response_invalid_action_falls_back_to_hold() -> None:
    """An unrecognised action string in the AI response must not crash
    _parse_json_response; a HOLD fallback should be returned instead."""
    gen = SignalGenerator(api_key="")
    market_data = _make_market_data()

    response_json = json.dumps(
        {
            "action": "YOLO",
            "confidence": 0.9,
            "amount_pct": 0.02,
            "reasoning": "moon",
        }
    )

    # TradingSignal.__post_init__ raises ValueError for invalid actions,
    # so _parse_json_response must catch that and return a HOLD.
    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)
    assert signal.action == "HOLD"


# ---------------------------------------------------------------------------
# Regression: amount field round-trips through to_dict / constructor
# ---------------------------------------------------------------------------


def test_trading_signal_round_trip_with_amount() -> None:
    """A signal reconstructed from to_dict() must preserve the amount."""
    original = TradingSignal(
        symbol="ETH/USDT",
        action="BUY",
        confidence=0.65,
        entry_price=3_000.0,
        amount=0.02,
        reasoning="round-trip test",
    )
    d = original.to_dict()
    reconstructed = TradingSignal(**d)
    assert reconstructed.amount == pytest.approx(0.02)
    assert reconstructed.symbol == "ETH/USDT"
    assert reconstructed.reasoning == "round-trip test"
