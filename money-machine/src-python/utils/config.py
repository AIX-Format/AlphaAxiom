"""
Configuration loader for Money Machine
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and config file"""
    
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
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return config


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to config file"""
    config_path = Path(__file__).parent.parent / "config.json"
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False
