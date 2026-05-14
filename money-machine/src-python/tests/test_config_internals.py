"""
Tests for the private helpers in utils/config.py that were introduced in this PR:

  - _without_secrets(value): strips keys that appear in SECRET_KEYS from any
    depth of a nested structure and from list elements.
  - _deep_merge(base, update): recursively merges two dicts in-place, falling
    back to simple overwrite for non-dict values.
  - save_config() returns False when the destination path is unwritable.

The public load_config() and save_config() integration behaviour is already
covered in test_config_security.py. This file focuses on the helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from utils.config import (  # noqa: E402
    SECRET_KEYS,
    _deep_merge,
    _without_secrets,
    save_config,
)


# ---------------------------------------------------------------------------
# _without_secrets – scalar pass-through
# ---------------------------------------------------------------------------


def test_without_secrets_passthrough_integer() -> None:
    assert _without_secrets(42) == 42


def test_without_secrets_passthrough_string() -> None:
    assert _without_secrets("hello") == "hello"


def test_without_secrets_passthrough_none() -> None:
    assert _without_secrets(None) is None


def test_without_secrets_passthrough_float() -> None:
    assert _without_secrets(3.14) == 3.14


def test_without_secrets_passthrough_bool() -> None:
    assert _without_secrets(True) is True


# ---------------------------------------------------------------------------
# _without_secrets – empty/flat dict
# ---------------------------------------------------------------------------


def test_without_secrets_empty_dict_returns_empty_dict() -> None:
    result = _without_secrets({})
    assert result == {}


def test_without_secrets_dict_without_secrets_is_unchanged() -> None:
    d = {"exchange_name": "binance", "ipc_port": 19284}
    assert _without_secrets(d) == d


def test_without_secrets_removes_api_key() -> None:
    result = _without_secrets({"api_key": "secret", "name": "binance"})
    assert "api_key" not in result
    assert result["name"] == "binance"


def test_without_secrets_removes_secret() -> None:
    result = _without_secrets({"secret": "s3cr3t", "sandbox": True})
    assert "secret" not in result
    assert result["sandbox"] is True


def test_without_secrets_removes_gemini_api_key() -> None:
    result = _without_secrets({"gemini_api_key": "gk-xxx", "model": "gemini-1.5-flash"})
    assert "gemini_api_key" not in result
    assert result["model"] == "gemini-1.5-flash"


def test_without_secrets_case_insensitive_removal() -> None:
    # SECRET_KEYS comparison is done with key.lower()
    result = _without_secrets({"API_KEY": "upper", "Secret": "mixed", "name": "ok"})
    assert "API_KEY" not in result
    assert "Secret" not in result
    assert result["name"] == "ok"


# ---------------------------------------------------------------------------
# _without_secrets – nested dicts
# ---------------------------------------------------------------------------


def test_without_secrets_nested_dict() -> None:
    data = {
        "exchange": {
            "name": "binance",
            "api_key": "should-be-gone",
            "secret": "also-gone",
        },
        "max_risk_per_trade": 0.02,
    }
    result = _without_secrets(data)
    assert result["exchange"] == {"name": "binance"}
    assert result["max_risk_per_trade"] == 0.02


def test_without_secrets_deeply_nested() -> None:
    data = {
        "level1": {
            "level2": {
                "api_key": "deep-secret",
                "allowed": "value",
            }
        }
    }
    result = _without_secrets(data)
    assert result["level1"]["level2"] == {"allowed": "value"}


def test_without_secrets_returns_new_dict_not_mutating_original() -> None:
    data = {"api_key": "secret", "name": "test"}
    result = _without_secrets(data)
    # Original is unmodified
    assert data["api_key"] == "secret"
    assert "api_key" not in result


# ---------------------------------------------------------------------------
# _without_secrets – lists
# ---------------------------------------------------------------------------


def test_without_secrets_list_of_scalars_passthrough() -> None:
    assert _without_secrets([1, 2, 3]) == [1, 2, 3]


def test_without_secrets_list_of_dicts_strips_secrets() -> None:
    data = [
        {"name": "a", "api_key": "key-a"},
        {"name": "b", "secret": "sec-b"},
    ]
    result = _without_secrets(data)
    assert result == [{"name": "a"}, {"name": "b"}]


def test_without_secrets_nested_list_inside_dict() -> None:
    data = {
        "providers": [
            {"name": "binance", "api_key": "k1"},
            {"name": "kraken", "api_key": "k2"},
        ]
    }
    result = _without_secrets(data)
    assert result == {"providers": [{"name": "binance"}, {"name": "kraken"}]}


def test_without_secrets_empty_list() -> None:
    assert _without_secrets([]) == []


# ---------------------------------------------------------------------------
# _without_secrets – SECRET_KEYS constant
# ---------------------------------------------------------------------------


def test_secret_keys_contains_expected_members() -> None:
    assert "api_key" in SECRET_KEYS
    assert "secret" in SECRET_KEYS
    assert "gemini_api_key" in SECRET_KEYS


def test_secret_keys_is_frozenset() -> None:
    assert isinstance(SECRET_KEYS, frozenset)


# ---------------------------------------------------------------------------
# _deep_merge – flat merge
# ---------------------------------------------------------------------------


def test_deep_merge_adds_new_keys_to_base() -> None:
    base: Dict[str, Any] = {"a": 1}
    _deep_merge(base, {"b": 2})
    assert base == {"a": 1, "b": 2}


def test_deep_merge_overwrites_existing_scalar() -> None:
    base: Dict[str, Any] = {"a": 1}
    _deep_merge(base, {"a": 99})
    assert base["a"] == 99


def test_deep_merge_modifies_base_in_place() -> None:
    base: Dict[str, Any] = {"x": 10}
    update = {"y": 20}
    _deep_merge(base, update)
    assert base == {"x": 10, "y": 20}
    assert update == {"y": 20}  # update is not modified


# ---------------------------------------------------------------------------
# _deep_merge – nested dict recursion
# ---------------------------------------------------------------------------


def test_deep_merge_recurses_into_nested_dict() -> None:
    base: Dict[str, Any] = {"exchange": {"name": "binance", "sandbox": True}}
    _deep_merge(base, {"exchange": {"sandbox": False}})
    assert base["exchange"]["name"] == "binance"   # preserved
    assert base["exchange"]["sandbox"] is False     # updated


def test_deep_merge_adds_new_nested_key() -> None:
    base: Dict[str, Any] = {"exchange": {"name": "binance"}}
    _deep_merge(base, {"exchange": {"region": "eu"}})
    assert base["exchange"]["name"] == "binance"
    assert base["exchange"]["region"] == "eu"


def test_deep_merge_nested_three_levels() -> None:
    base: Dict[str, Any] = {"a": {"b": {"c": 1}}}
    _deep_merge(base, {"a": {"b": {"d": 2}}})
    assert base == {"a": {"b": {"c": 1, "d": 2}}}


# ---------------------------------------------------------------------------
# _deep_merge – non-dict value overwrites (lists, scalars replacing dicts)
# ---------------------------------------------------------------------------


def test_deep_merge_list_is_replaced_not_merged() -> None:
    base: Dict[str, Any] = {"items": [1, 2, 3]}
    _deep_merge(base, {"items": [4, 5]})
    assert base["items"] == [4, 5]  # lists are not extended


def test_deep_merge_dict_replaced_by_scalar() -> None:
    base: Dict[str, Any] = {"section": {"key": "value"}}
    _deep_merge(base, {"section": 42})
    assert base["section"] == 42  # scalar wins over dict


def test_deep_merge_scalar_replaced_by_dict() -> None:
    base: Dict[str, Any] = {"section": 42}
    _deep_merge(base, {"section": {"key": "value"}})
    assert base["section"] == {"key": "value"}


def test_deep_merge_empty_update_leaves_base_unchanged() -> None:
    base: Dict[str, Any] = {"a": 1, "b": 2}
    _deep_merge(base, {})
    assert base == {"a": 1, "b": 2}


def test_deep_merge_empty_base_receives_all_update_keys() -> None:
    base: Dict[str, Any] = {}
    _deep_merge(base, {"x": 1, "y": {"z": 2}})
    assert base == {"x": 1, "y": {"z": 2}}


# ---------------------------------------------------------------------------
# save_config – error path returns False
# ---------------------------------------------------------------------------


def test_save_config_returns_false_on_unwritable_path(monkeypatch) -> None:
    """save_config() should return False (not raise) when the path is unwritable."""
    import utils.config as config_module

    # Point __file__ at a path inside a non-existent directory so the open()
    # call inside save_config() will fail with a FileNotFoundError.
    monkeypatch.setattr(
        config_module,
        "__file__",
        "/nonexistent_directory/sub/config.py",
    )
    result = save_config({"max_risk_per_trade": 0.02})
    assert result is False