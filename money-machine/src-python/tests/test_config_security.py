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
    """
    Verifies that save_config persists a config.json with secret fields removed.
    
    Calls save_config with a nested configuration containing exchange.api_key, exchange.secret, and gemini_api_key and asserts the written config.json retains only non-secret fields (exchange.name and max_risk_per_trade).
    """
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
