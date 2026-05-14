"""
Comprehensive tests for validate_config_update() in engine.trading_core.

The function is new in this PR and enforces a whitelist + numeric range
policy on runtime config mutations.  Tests here cover:

  - Happy path: all three allowed keys, individually and combined.
  - Type enforcement: booleans, strings, and non-dict inputs are rejected.
  - Integer values are accepted and coerced to float.
  - Boundary values for each key, probing both inclusive and exclusive limits.
  - Error messages contain the offending key name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.trading_core import CONFIG_LIMITS, validate_config_update  # noqa: E402


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_empty_dict_returns_empty_dict() -> None:
    assert validate_config_update({}) == {}


def test_valid_initial_balance_at_minimum() -> None:
    result = validate_config_update({"initial_balance": 100.0})
    assert result == {"initial_balance": 100.0}


def test_valid_initial_balance_at_maximum() -> None:
    result = validate_config_update({"initial_balance": 1_000_000.0})
    assert result == {"initial_balance": 1_000_000.0}


def test_valid_initial_balance_midrange() -> None:
    result = validate_config_update({"initial_balance": 50_000.0})
    assert result == {"initial_balance": 50_000.0}


def test_valid_max_risk_per_trade_small_positive() -> None:
    result = validate_config_update({"max_risk_per_trade": 0.001})
    assert result == {"max_risk_per_trade": 0.001}


def test_valid_max_risk_per_trade_at_maximum() -> None:
    result = validate_config_update({"max_risk_per_trade": 0.1})
    assert result == {"max_risk_per_trade": 0.1}


def test_valid_max_daily_loss_small_positive() -> None:
    result = validate_config_update({"max_daily_loss": 0.001})
    assert result == {"max_daily_loss": 0.001}


def test_valid_max_daily_loss_at_maximum() -> None:
    result = validate_config_update({"max_daily_loss": 0.2})
    assert result == {"max_daily_loss": 0.2}


def test_multiple_valid_keys_in_single_call() -> None:
    payload = {
        "initial_balance": 5000.0,
        "max_risk_per_trade": 0.02,
        "max_daily_loss": 0.05,
    }
    result = validate_config_update(payload)
    assert result == {
        "initial_balance": 5000.0,
        "max_risk_per_trade": 0.02,
        "max_daily_loss": 0.05,
    }


def test_integer_value_is_accepted_and_coerced_to_float() -> None:
    result = validate_config_update({"initial_balance": 500})
    assert result == {"initial_balance": 500.0}
    assert isinstance(result["initial_balance"], float)


def test_integer_value_for_risk_coerced_to_float() -> None:
    # While 0 is out of range for max_risk_per_trade, 1 would be too; use a
    # valid integer that coerces to a valid float.
    # There is no valid integer for max_risk_per_trade because the only
    # integer ≤ 0.1 is 0, which violates the exclusive minimum.
    # Use initial_balance instead, which accepts integer 100 (min = 100.0 inclusive).
    result = validate_config_update({"initial_balance": 100})
    assert isinstance(result["initial_balance"], float)
    assert result["initial_balance"] == 100.0


# ---------------------------------------------------------------------------
# Type rejection
# ---------------------------------------------------------------------------


def test_non_dict_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="object"):
        validate_config_update([("initial_balance", 500.0)])  # type: ignore[arg-type]


def test_non_dict_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="object"):
        validate_config_update("initial_balance=500")  # type: ignore[arg-type]


def test_non_dict_none_raises_value_error() -> None:
    with pytest.raises(ValueError, match="object"):
        validate_config_update(None)  # type: ignore[arg-type]


def test_bool_true_rejected_even_though_isinstance_int() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": True})


def test_bool_false_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": False})


def test_string_numeric_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": "500.0"})


def test_none_value_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": None})


def test_dict_value_rejected() -> None:
    # A nested dict is not numeric.
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": {"value": 500}})


# ---------------------------------------------------------------------------
# Key whitelist
# ---------------------------------------------------------------------------


def test_unknown_key_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"unknown_key": 1.0})


def test_exchange_nested_key_raises_value_error() -> None:
    # "exchange" itself is not in CONFIG_LIMITS
    with pytest.raises(ValueError):
        validate_config_update({"exchange": 1.0})


def test_gemini_api_key_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"gemini_api_key": 1.0})


def test_error_message_contains_key_name() -> None:
    with pytest.raises(ValueError, match="unsupported_key"):
        validate_config_update({"unsupported_key": 1.0})


# ---------------------------------------------------------------------------
# Boundary values – initial_balance (inclusive minimum 100.0, max 1_000_000.0)
# ---------------------------------------------------------------------------


def test_initial_balance_below_minimum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 99.99})


def test_initial_balance_zero_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 0.0})


def test_initial_balance_negative_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": -1.0})


def test_initial_balance_above_maximum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 1_000_001.0})


def test_initial_balance_at_minimum_accepted() -> None:
    assert validate_config_update({"initial_balance": 100.0}) == {
        "initial_balance": 100.0
    }


# ---------------------------------------------------------------------------
# Boundary values – max_risk_per_trade (exclusive minimum 0.0, max 0.1)
# ---------------------------------------------------------------------------


def test_max_risk_per_trade_at_exclusive_minimum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.0})


def test_max_risk_per_trade_above_maximum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.10001})


def test_max_risk_per_trade_much_above_maximum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.5})


def test_max_risk_per_trade_negative_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": -0.01})


# ---------------------------------------------------------------------------
# Boundary values – max_daily_loss (exclusive minimum 0.0, max 0.2)
# ---------------------------------------------------------------------------


def test_max_daily_loss_at_exclusive_minimum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": 0.0})


def test_max_daily_loss_above_maximum_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": 0.201})


def test_max_daily_loss_negative_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": -0.1})


def test_max_daily_loss_at_maximum_accepted() -> None:
    assert validate_config_update({"max_daily_loss": 0.2}) == {"max_daily_loss": 0.2}


# ---------------------------------------------------------------------------
# Return value is always a new dict (not mutated from input)
# ---------------------------------------------------------------------------


def test_returned_dict_is_independent_from_input() -> None:
    original = {"initial_balance": 500.0}
    result = validate_config_update(original)
    result["initial_balance"] = 999.0
    assert original["initial_balance"] == 500.0


# ---------------------------------------------------------------------------
# CONFIG_LIMITS constant structure
# ---------------------------------------------------------------------------


def test_config_limits_has_expected_keys() -> None:
    assert set(CONFIG_LIMITS.keys()) == {"initial_balance", "max_risk_per_trade", "max_daily_loss"}


def test_config_limits_initial_balance_is_inclusive_minimum() -> None:
    _min, _max, exclusive = CONFIG_LIMITS["initial_balance"]
    assert exclusive is False  # inclusive lower bound


def test_config_limits_risk_and_loss_are_exclusive_minimum() -> None:
    for key in ("max_risk_per_trade", "max_daily_loss"):
        _min, _max, exclusive = CONFIG_LIMITS[key]
        assert exclusive is True, f"{key} should have exclusive minimum"
