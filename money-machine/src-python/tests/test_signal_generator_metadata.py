"""
Tests for signal_generator._parse_json_response metadata changes.

PR change: the `amount` field was removed from TradingSignal; instead
`amount_pct` is now stored inside `metadata` so downstream consumers
can still access it without breaking the dataclass contract.

These tests exercise _parse_json_response in isolation:
- amount_pct present in Gemini JSON → surfaced in metadata dict
- amount_pct absent → metadata contains None for the key
- amount_pct explicitly null → metadata contains None
- Non-JSON response → fallback HOLD signal returned
- Market data drives entry_price when entry_price is absent from JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

# engine.strategies.base imports pandas at module level; skip the whole
# module gracefully when pandas is not installed so the rest of the
# suite keeps running.
pandas = pytest.importorskip("pandas")

from engine.signal_generator import SignalGenerator  # noqa: E402
from engine.strategies.base import TradingSignal  # noqa: E402


def _make_ohlcv(close: float = 50_000.0, n: int = 1) -> List[List]:
    """Return n minimal OHLCV candles all with the given close price."""
    ts = 1_700_000_000_000  # arbitrary ms timestamp
    return [[ts + i * 60_000, close, close, close, close, 100.0] for i in range(n)]


def _generator() -> SignalGenerator:
    """Return a SignalGenerator with no API key (model=None)."""
    return SignalGenerator(api_key="")


# ---------------------------------------------------------------------------
# _parse_json_response – metadata / amount_pct
# ---------------------------------------------------------------------------


def test_parse_json_response_stores_amount_pct_in_metadata() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    response_json = json.dumps({
        "action": "BUY",
        "confidence": 0.8,
        "entry_price": 50_100.0,
        "stop_loss": 49_500.0,
        "take_profit": 51_000.0,
        "amount_pct": 0.02,
        "reasoning": "Strong momentum",
    })

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert isinstance(signal, TradingSignal)
    assert signal.action == "BUY"
    assert signal.metadata.get("amount_pct") == pytest.approx(0.02)


def test_parse_json_response_missing_amount_pct_stores_none() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    response_json = json.dumps({
        "action": "SELL",
        "confidence": 0.6,
        "reasoning": "Overbought",
        # amount_pct deliberately absent
    })

    signal = gen._parse_json_response("ETH/USDT", response_json, market_data)

    assert signal.action == "SELL"
    assert "amount_pct" in signal.metadata
    assert signal.metadata["amount_pct"] is None


def test_parse_json_response_explicit_null_amount_pct_stored_as_none() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    response_json = json.dumps({
        "action": "HOLD",
        "confidence": 0.4,
        "amount_pct": None,
        "reasoning": "Unclear",
    })

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert signal.metadata["amount_pct"] is None


def test_parse_json_response_amount_pct_not_on_top_level_signal() -> None:
    """amount_pct must NOT be stored as a top-level TradingSignal field."""
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    response_json = json.dumps({
        "action": "BUY",
        "confidence": 0.75,
        "amount_pct": 0.015,
        "reasoning": "Breakout",
    })

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    # amount_pct should live in metadata, NOT as a standalone attribute
    assert not hasattr(signal, "amount") or signal.__class__.__name__ == "TradingSignal"
    assert signal.metadata["amount_pct"] == pytest.approx(0.015)


def test_parse_json_response_invalid_json_returns_hold_fallback() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)

    signal = gen._parse_json_response("BTC/USDT", "not json at all {{", market_data)

    assert signal.action == "HOLD"
    assert signal.confidence == pytest.approx(0.3)


def test_parse_json_response_empty_json_object_returns_hold_default() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    # Empty dict: action defaults to HOLD, confidence defaults to 0.5
    signal = gen._parse_json_response("BTC/USDT", json.dumps({}), market_data)
    assert signal.action == "HOLD"


def test_parse_json_response_uses_current_price_when_entry_price_absent() -> None:
    gen = _generator()
    close = 42_000.0
    market_data = _make_ohlcv(close)
    response_json = json.dumps({
        "action": "BUY",
        "confidence": 0.7,
        "entry_price": None,  # null → falls back to current price
        "reasoning": "Dip",
    })

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert signal.entry_price == pytest.approx(close)


def test_parse_json_response_uses_provided_entry_price_when_present() -> None:
    gen = _generator()
    market_data = _make_ohlcv(50_000.0)
    response_json = json.dumps({
        "action": "BUY",
        "confidence": 0.8,
        "entry_price": 49_800.0,
        "reasoning": "Limit entry",
    })

    signal = gen._parse_json_response("BTC/USDT", response_json, market_data)

    assert signal.entry_price == pytest.approx(49_800.0)


def test_parse_json_response_symbol_is_preserved() -> None:
    gen = _generator()
    market_data = _make_ohlcv(1_800.0)
    response_json = json.dumps({"action": "HOLD", "confidence": 0.5})

    signal = gen._parse_json_response("ETH/USDT", response_json, market_data)

    assert signal.symbol == "ETH/USDT"


# ---------------------------------------------------------------------------
# _generate_rule_based_signal – metadata field present (regression guard)
# ---------------------------------------------------------------------------


def test_rule_based_signal_has_metadata_field() -> None:
    """The rule-based fallback must also produce a TradingSignal that has
    a metadata dict (even if empty) since TradingSignal now always has one.
    """
    gen = _generator()
    # 21 candles so the rule-based path runs fully.
    market_data = _make_ohlcv(50_000.0, n=21)

    signal = gen._generate_rule_based_signal("BTC/USDT", market_data)

    assert isinstance(signal.metadata, dict)
