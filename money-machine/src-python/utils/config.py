"""
Configuration loader for Money Machine
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

SECRET_KEYS = frozenset({"api_key", "secret", "gemini_api_key"})


def load_config() -> Dict[str, Any]:
    """
    Builds the application configuration from environment variables and an optional config.json file.
    
    If a config.json file is present, its values are merged into the environment-based defaults after removing secret-bearing keys; if the file cannot be read or parsed a warning is printed and the environment defaults are used unchanged.
    
    Returns:
        dict: Configuration dictionary with defaults from environment variables, updated by non-secret values from config.json when available.
    """
    
    config = {
        # Default values
        "initial_balance": float(os.environ.get("INITIAL_BALANCE", "10000")),
        "max_risk_per_trade": float(os.environ.get("MAX_RISK_PER_TRADE", "0.02")),  # 2%
        "max_daily_loss": float(os.environ.get("MAX_DAILY_LOSS", "0.05")),  # 5%
        
        # Exchange configuration
        "exchange": {
            "name": os.environ.get("EXCHANGE_NAME", "binance"),
            "api_key": os.environ.get("EXCHANGE_API_KEY", ""),
            "secret": os.environ.get("EXCHANGE_SECRET", ""),
            "sandbox": os.environ.get("EXCHANGE_SANDBOX", "true").lower() == "true",
        },
        
        # AI Provider (Gemini)
        "gemini_api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        
        # IPC
        "ipc_port": int(os.environ.get("TAURI_PORT", 19284)),
    }
    
    # Try to load from config file if exists
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                _deep_merge(config, _without_secrets(file_config))
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return config


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Recursively merge keys from `update` into `base`, mutating `base` in place.
    
    For keys present in both mappings whose values are dictionaries, their mappings are merged recursively; for all other keys the value from `update` replaces the value in `base`.
    
    Parameters:
        base (Dict[str, Any]): The target mapping to be updated; this object is modified in place.
        update (Dict[str, Any]): The source mapping whose keys and values are merged into `base`.
    """
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _without_secrets(value: Any) -> Any:
    """
    Produce a copy of `value` with any dictionary entries whose key (case-insensitive) is in `SECRET_KEYS` removed.
    
    Recursively processes dictionaries and lists; non-dict/list values are returned unchanged.
    
    Parameters:
        value (Any): The input structure (dict, list, or other) to cleanse of secret-bearing keys.
    
    Returns:
        Any: The cleaned value with secret keys removed, preserving the input's structure types.
    """
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in SECRET_KEYS:
                continue
            cleaned[key] = _without_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def save_config(config: Dict[str, Any]) -> bool:
    """
    Write the provided configuration to the project's config.json after removing secret keys.
    
    The file is written to the repository-level config.json (Path(__file__).parent.parent / "config.json"). Secret-bearing keys (case-insensitive matches against SECRET_KEYS) are removed from the data before it is persisted.
    
    Parameters:
        config (Dict[str, Any]): Configuration mapping to save; secret fields will be stripped before writing.
    
    Returns:
        bool: `True` if the file was written successfully, `False` otherwise.
    """
    config_path = Path(__file__).parent.parent / "config.json"
    
    try:
        with open(config_path, 'w') as f:
            json.dump(_without_secrets(config), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False
