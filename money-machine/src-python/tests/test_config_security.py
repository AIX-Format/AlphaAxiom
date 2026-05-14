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
# Additional validate_config_update tests — boundary conditions
# ---------------------------------------------------------------------------


def test_validate_config_update_rejects_non_dict_input() -> None:
    """Non-dict inputs must always raise ValueError."""
    with pytest.raises(ValueError):
        validate_config_update("initial_balance=5000")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_config_update(["initial_balance", 5000])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_config_update(None)  # type: ignore[arg-type]


def test_validate_config_update_empty_dict_returns_empty() -> None:
    """An empty update is a no-op and should return an empty dict."""
    assert validate_config_update({}) == {}


def test_validate_config_update_rejects_bool_values() -> None:
    """Boolean values must be rejected even for otherwise valid keys,
    because `isinstance(True, int)` is True in Python and would pass
    a naive numeric check."""
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": True})
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": False})
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": True})


def test_validate_config_update_accepts_integer_values() -> None:
    """Integer literals must be accepted and coerced to float."""
    result = validate_config_update({"initial_balance": 5000})
    assert result == {"initial_balance": 5000.0}
    assert isinstance(result["initial_balance"], float)


def test_validate_config_update_initial_balance_inclusive_min() -> None:
    """initial_balance has an inclusive lower bound of 100.0."""
    assert validate_config_update({"initial_balance": 100.0}) == {
        "initial_balance": 100.0
    }
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 99.99})


def test_validate_config_update_initial_balance_inclusive_max() -> None:
    """initial_balance has an inclusive upper bound of 1_000_000.0."""
    assert validate_config_update({"initial_balance": 1_000_000.0}) == {
        "initial_balance": 1_000_000.0
    }
    with pytest.raises(ValueError):
        validate_config_update({"initial_balance": 1_000_001.0})


def test_validate_config_update_max_risk_exclusive_min() -> None:
    """max_risk_per_trade has an exclusive lower bound of 0.0 (must be > 0)."""
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.0})


def test_validate_config_update_max_risk_inclusive_max() -> None:
    """max_risk_per_trade accepts its maximum value of 0.1 exactly."""
    assert validate_config_update({"max_risk_per_trade": 0.1}) == {
        "max_risk_per_trade": 0.1
    }
    with pytest.raises(ValueError):
        validate_config_update({"max_risk_per_trade": 0.101})


def test_validate_config_update_max_daily_loss_inclusive_max() -> None:
    """max_daily_loss accepts its maximum value of 0.2 exactly."""
    assert validate_config_update({"max_daily_loss": 0.2}) == {
        "max_daily_loss": 0.2
    }
    with pytest.raises(ValueError):
        validate_config_update({"max_daily_loss": 0.21})


def test_validate_config_update_all_valid_keys_together() -> None:
    """All three supported keys can be updated in a single call."""
    result = validate_config_update(
        {
            "initial_balance": 20_000.0,
            "max_risk_per_trade": 0.05,
            "max_daily_loss": 0.10,
        }
    )
    assert result == {
        "initial_balance": 20_000.0,
        "max_risk_per_trade": 0.05,
        "max_daily_loss": 0.10,
    }
    # All values must be floats regardless of input type.
    for v in result.values():
        assert isinstance(v, float)


def test_validate_config_update_rejects_mix_of_valid_and_invalid_keys() -> None:
    """If any key is unsupported the whole update must be rejected.
    Partial application of a multi-key update would leave the config
    in an inconsistent state.
    """
    with pytest.raises(ValueError):
        validate_config_update(
            {"max_risk_per_trade": 0.01, "gemini_api_key": "secret"}
        )


# ---------------------------------------------------------------------------
# Additional config.py tests — _without_secrets with list values
# ---------------------------------------------------------------------------


def test_save_config_redacts_secrets_inside_lists(tmp_path, monkeypatch) -> None:
    """_without_secrets must recurse into list elements so a config
    that stores exchange credentials in a list is still sanitised."""
    fake_config_module_path = tmp_path / "utils" / "config.py"
    fake_config_module_path.parent.mkdir()
    fake_config_module_path.write_text("")
    monkeypatch.setattr(config_module, "__file__", str(fake_config_module_path))

    config_module.save_config(
        {
            "exchanges": [
                {"name": "binance", "api_key": "key-a", "secret": "sec-a"},
                {"name": "kraken", "api_key": "key-b", "secret": "sec-b"},
            ],
        }
    )

    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted == {
        "exchanges": [
            {"name": "binance"},
            {"name": "kraken"},
        ]
    }


def test_save_config_preserves_non_secret_data(tmp_path, monkeypatch) -> None:
    """Non-secret fields must survive the redaction pass unchanged."""
    fake_config_module_path = tmp_path / "utils" / "config.py"
    fake_config_module_path.parent.mkdir()
    fake_config_module_path.write_text("")
    monkeypatch.setattr(config_module, "__file__", str(fake_config_module_path))

    config_module.save_config(
        {
            "initial_balance": 5000.0,
            "max_risk_per_trade": 0.01,
            "exchange": {"name": "bybit", "sandbox": True},
        }
    )

    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted["initial_balance"] == 5000.0
    assert persisted["max_risk_per_trade"] == 0.01
    assert persisted["exchange"]["name"] == "bybit"
    assert persisted["exchange"]["sandbox"] is True


# ---------------------------------------------------------------------------
# TradingEngine.update_config integration with validate_config_update
# ---------------------------------------------------------------------------


def test_trading_engine_update_config_accepts_valid_values() -> None:
    """update_config() should persist a validated value into the engine config."""
    import asyncio

    from engine.trading_core import TradingEngine

    engine = TradingEngine({"initial_balance": 10_000.0, "max_risk_per_trade": 0.02})
    asyncio.run(engine.update_config({"max_risk_per_trade": 0.05}))
    assert engine.config["max_risk_per_trade"] == 0.05


def test_trading_engine_update_config_rejects_invalid_values() -> None:
    """update_config() must propagate ValueError for out-of-range values
    and leave the engine config unchanged."""
    import asyncio

    from engine.trading_core import TradingEngine

    engine = TradingEngine({"initial_balance": 10_000.0, "max_risk_per_trade": 0.02})
    with pytest.raises(ValueError):
        asyncio.run(engine.update_config({"max_risk_per_trade": 0.99}))
    # Config must remain unchanged.
    assert engine.config["max_risk_per_trade"] == 0.02


def test_trading_engine_update_config_rejects_unknown_keys() -> None:
    """update_config() must not allow arbitrary key injection."""
    import asyncio

    from engine.trading_core import TradingEngine

    engine = TradingEngine({"initial_balance": 10_000.0})
    with pytest.raises(ValueError):
        asyncio.run(engine.update_config({"exchange": {"api_key": "injected"}}))
    assert "exchange" not in engine.config or "api_key" not in engine.config.get("exchange", {})
