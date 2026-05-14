from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.trading_core import validate_config_update  # noqa: E402
from utils import config as config_module  # noqa: E402
from utils.config import _without_secrets, _deep_merge  # noqa: E402


def test_save_config_redacts_nested_secrets(tmp_path, monkeypatch) -> None:
    fake_config_module_path = tmp_path / "utils" / "config.py"
    fake_config_module_path.parent.mkdir()
    fake_config_module_path.write_text("")
    monkeypatch.setattr(config_module, "__file__", str(fake_config_module_path))

    assert config_module.save_config(
        {
            "exchange": {
                "name": "binance",
                "api_key": "should-not-hit-disk",
                "secret": "should-not-hit-disk",
            },
            "gemini_api_key": "also-secret",
            "max_risk_per_trade": 0.02,
        }
    )

    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted == {
        "exchange": {"name": "binance"},
        "max_risk_per_trade": 0.02,
    }


def test_load_config_ignores_disk_secrets(tmp_path, monkeypatch) -> None:
    fake_config_module_path = tmp_path / "utils" / "config.py"
    fake_config_module_path.parent.mkdir()
    fake_config_module_path.write_text("")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "exchange": {"api_key": "disk-key", "secret": "disk-secret"},
                "gemini_api_key": "disk-gemini",
                "max_risk_per_trade": 0.03,
            }
        )
    )
    monkeypatch.setattr(config_module, "__file__", str(fake_config_module_path))

    loaded = config_module.load_config()
    assert loaded["max_risk_per_trade"] == 0.03
    assert loaded["gemini_api_key"] == ""
    assert loaded["exchange"]["api_key"] == ""
    assert loaded["exchange"]["secret"] == ""


def test_validate_config_update_rejects_secrets_and_unknown_keys() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"exchange": {"api_key": "nope"}})
    with pytest.raises(ValueError):
        validate_config_update({"gemini_model": "gemini-1.5-flash"})
    with pytest.raises(ValueError):
        validate_config_update({"unsafe": True})


def test_validate_config_update_enforces_ranges() -> None:
    assert validate_config_update({"max_risk_per_trade": 0.02}) == {
        "max_risk_per_trade": 0.02
    }
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.5})
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 50})
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": 0.0})


# ---------------------------------------------------------------------------
# validate_config_update – additional boundary and type checks
# ---------------------------------------------------------------------------


def test_validate_config_update_empty_dict_returns_empty() -> None:
    assert validate_config_update({}) == {}


def test_validate_config_update_non_dict_raises() -> None:
    with pytest.raises(ValueError):
        validate_config_update([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_config_update("nope")  # type: ignore[arg-type]


def test_validate_config_update_boolean_rejected() -> None:
    # bool is a subclass of int, but must be explicitly rejected.
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": True})
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": False})


def test_validate_config_update_string_value_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": "10000"})


def test_validate_config_update_initial_balance_inclusive_min_boundary() -> None:
    # 100.0 is the inclusive minimum; must be accepted.
    result = validate_config_update({"initial_balance": 100.0})
    assert result == {"initial_balance": 100.0}


def test_validate_config_update_initial_balance_at_max_boundary() -> None:
    result = validate_config_update({"initial_balance": 1_000_000.0})
    assert result == {"initial_balance": 1_000_000.0}


def test_validate_config_update_initial_balance_below_min_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 99.9})


def test_validate_config_update_initial_balance_above_max_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 1_000_000.01})


def test_validate_config_update_max_risk_per_trade_exclusive_min_rejected() -> None:
    # max_risk_per_trade has exclusive_minimum=True so 0.0 must be rejected.
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.0})


def test_validate_config_update_max_risk_per_trade_at_max_accepted() -> None:
    # 0.1 == maximum; the check is `value > maximum` so 0.1 is accepted.
    result = validate_config_update({"max_risk_per_trade": 0.1})
    assert result == {"max_risk_per_trade": 0.1}


def test_validate_config_update_max_daily_loss_at_max_accepted() -> None:
    result = validate_config_update({"max_daily_loss": 0.2})
    assert result == {"max_daily_loss": 0.2}


def test_validate_config_update_multiple_valid_keys() -> None:
    result = validate_config_update(
        {"initial_balance": 5000.0, "max_risk_per_trade": 0.05, "max_daily_loss": 0.1}
    )
    assert result == {
        "initial_balance": 5000.0,
        "max_risk_per_trade": 0.05,
        "max_daily_loss": 0.1,
    }


def test_validate_config_update_int_values_accepted_as_float() -> None:
    # int literals are allowed; the function casts them to float.
    result = validate_config_update({"initial_balance": 500})
    assert result == {"initial_balance": 500.0}
    assert isinstance(result["initial_balance"], float)


# ---------------------------------------------------------------------------
# _without_secrets – unit tests
# ---------------------------------------------------------------------------


def test_without_secrets_removes_top_level_secret_keys() -> None:
    data = {"api_key": "s", "secret": "s", "gemini_api_key": "s", "name": "binance"}
    assert _without_secrets(data) == {"name": "binance"}


def test_without_secrets_case_insensitive_key_matching() -> None:
    # Keys checked with .lower(); mixed-case variants must be stripped.
    data = {"API_KEY": "leak", "Secret": "leak", "model": "x"}
    result = _without_secrets(data)
    assert "API_KEY" not in result
    assert "Secret" not in result
    assert result.get("model") == "x"


def test_without_secrets_handles_nested_dicts() -> None:
    data = {"exchange": {"api_key": "k", "secret": "s", "name": "binance"}}
    assert _without_secrets(data) == {"exchange": {"name": "binance"}}


def test_without_secrets_handles_list_of_dicts() -> None:
    data = [{"api_key": "leak", "name": "binance"}, {"secret": "s", "label": "main"}]
    result = _without_secrets(data)
    assert result == [{"name": "binance"}, {"label": "main"}]


def test_without_secrets_primitives_pass_through() -> None:
    assert _without_secrets(42) == 42
    assert _without_secrets("hello") == "hello"
    assert _without_secrets(3.14) == 3.14
    assert _without_secrets(None) is None
    assert _without_secrets(True) is True


def test_without_secrets_empty_dict_returns_empty() -> None:
    assert _without_secrets({}) == {}


def test_without_secrets_non_secret_keys_preserved_in_nested() -> None:
    data = {"exchange": {"name": "bybit", "sandbox": True}}
    assert _without_secrets(data) == {"exchange": {"name": "bybit", "sandbox": True}}


# ---------------------------------------------------------------------------
# _deep_merge – unit tests
# ---------------------------------------------------------------------------


def test_deep_merge_nested_dicts_recursive() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 10}
    update = {"a": {"y": 99, "z": 3}}
    _deep_merge(base, update)
    assert base == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}


def test_deep_merge_overwrites_scalar_with_scalar() -> None:
    base = {"key": "old"}
    _deep_merge(base, {"key": "new"})
    assert base["key"] == "new"


def test_deep_merge_adds_new_top_level_key() -> None:
    base: dict = {}
    _deep_merge(base, {"fresh": 42})
    assert base == {"fresh": 42}


def test_deep_merge_overwrites_dict_with_scalar_when_update_is_scalar() -> None:
    # If base has a dict but update has a scalar, the scalar wins.
    base = {"key": {"nested": 1}}
    _deep_merge(base, {"key": "flat"})
    assert base["key"] == "flat"


# ---------------------------------------------------------------------------
# save_config – error path
# ---------------------------------------------------------------------------


def test_save_config_returns_false_on_write_error(tmp_path, monkeypatch) -> None:
    # Point __file__ at a path whose parent cannot be written to.
    fake_module = tmp_path / "utils" / "config.py"
    fake_module.parent.mkdir()
    fake_module.write_text("")
    # Make the target config.json un-writable by replacing open with a raiser.
    import builtins

    real_open = builtins.open

    def _raising_open(path, mode="r", **kw):
        if "w" in str(mode):
            raise OSError("disk full")
        return real_open(path, mode, **kw)

    monkeypatch.setattr(config_module, "__file__", str(fake_module))
    monkeypatch.setattr(builtins, "open", _raising_open)

    result = config_module.save_config({"max_risk_per_trade": 0.01})
    assert result is False


# ---------------------------------------------------------------------------
# load_config – non-secret file keys are preserved after merge
# ---------------------------------------------------------------------------


def test_load_config_non_secret_file_keys_are_preserved(tmp_path, monkeypatch) -> None:
    fake_module = tmp_path / "utils" / "config.py"
    fake_module.parent.mkdir()
    fake_module.write_text("")
    (tmp_path / "config.json").write_text(
        json.dumps({"gemini_model": "gemini-1.5-pro", "max_risk_per_trade": 0.01})
    )
    monkeypatch.setattr(config_module, "__file__", str(fake_module))

    loaded = config_module.load_config()
    # Non-secret key from file must survive the merge.
    assert loaded["gemini_model"] == "gemini-1.5-pro"
    assert loaded["max_risk_per_trade"] == 0.01
