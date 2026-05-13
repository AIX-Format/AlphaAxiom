"""
Money Machine Trading Engine - Main Entry Point
Runs as subprocess from Rust backend, communicates via TCP/JSON
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.trading_core import TradingEngine
from engine.signal_generator import SignalGenerator
from engine.shadow_mode import ShadowDecision, ShadowModeMonitor
from skills.skill_executor import SkillExecutor
from utils.ipc_server import IPCServer
from utils.hot_reload import HotReloadManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MoneyMachineApp:
    def __init__(self):
        self.engine = None
        self.skill_executor = None
        self.signal_generator = None
        self.hot_reload = None
        self.ipc_server = None
        self.running = True
        self.config = None
        self.shadow_monitor = ShadowModeMonitor()
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Money Machine...")
        
        # Load configuration
        from utils.config import load_config
        self.config = load_config()
        
        # Initialize trading engine
        self.engine = TradingEngine(self.config)
        await self.engine.initialize()
        
        # Initialize skill executor
        self.skill_executor = SkillExecutor(
            engine=self.engine,
            api_key=self.config.get('gemini_api_key', '')
        )
        
        # Initialize signal generator (Phase 3: The Brain)
        self.signal_generator = SignalGenerator(
            api_key=self.config.get('gemini_api_key', '')
        )
        
        # Initialize hot-reload system
        self.hot_reload = HotReloadManager(self.skill_executor)
        self.hot_reload.setup()
        
        logger.info("✅ Initialization complete")
        logger.info(f"   - Trading Engine: {'Connected' if self.engine.is_connected() else 'Mock Mode'}")
        logger.info(f"   - Signal Generator: {'Gemini AI' if self.signal_generator.model else 'Rule-Based'}")
        logger.info(f"   - Skills Loaded: {len(self.skill_executor.loaded_skills)}")
    
    async def start(self):
        """Start the application"""
        port = int(os.environ.get('TAURI_PORT', 19284))
        
        # Start hot-reload system
        self.hot_reload.start()
        
        # Start IPC server (listens for Rust commands)
        self.ipc_server = IPCServer(self.handle_command, port=port)
        
        logger.info(f"🚀 IPC Server listening on port {port}")
        
        # Run IPC server
        await self.ipc_server.start()
    
    async def handle_command(self, command: str, payload: dict) -> dict:
        """Handle commands from Rust backend"""
        handlers = {
            "START_TRADING": self.cmd_start_trading,
            "STOP_TRADING": self.cmd_stop_trading,
            "GET_PORTFOLIO": self.cmd_get_portfolio,
            "EXECUTE_SKILL": self.cmd_execute_skill,
            "UPDATE_CONFIG": self.cmd_update_config,
            "GET_STATUS": self.cmd_get_status,
            "PING": self.cmd_ping,
            # Phase 3: AI Commands
            "GENERATE_SIGNAL": self.cmd_generate_signal,
            "GET_LAST_SIGNAL": self.cmd_get_last_signal,
            "RELOAD_SKILLS": self.cmd_reload_skills,
            "GET_SHADOW_REPORT": self.cmd_get_shadow_report,
        }
        
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Unknown command: {command}"}
        
        try:
            result = await handler(payload)
            return {"result": result, "error": None}
        except Exception as e:
            logger.error(f"Command error: {e}")
            return {"error": str(e)}
    
    async def cmd_ping(self, payload: dict) -> dict:
        """Health check"""
        return {"status": "pong", "version": "0.2.0"}
    
    async def cmd_start_trading(self, payload: dict) -> dict:
        """Enable automated trading"""
        self.engine.trading_active = True
        logger.info("Trading STARTED")
        return {"status": "trading_active"}
    
    async def cmd_stop_trading(self, payload: dict) -> dict:
        """Disable automated trading"""
        self.engine.trading_active = False
        logger.info("Trading STOPPED")
        return {"status": "trading_stopped"}
    
    async def cmd_get_portfolio(self, payload: dict) -> dict:
        """Return current portfolio state"""
        return {
            "balance": self.engine.portfolio.get_balance(),
            "positions": self.engine.portfolio.get_positions(),
            "pnl": self.engine.portfolio.calculate_pnl(),
            "timestamp": self.engine.get_server_time()
        }
    
    async def cmd_execute_skill(self, payload: dict) -> dict:
        """Execute a specific skill"""
        skill_name = payload.get("skill")
        params = payload.get("params", {})
        
        return await self.skill_executor.execute_skill(skill_name, params)
    
    async def cmd_update_config(self, payload: dict) -> dict:
        """Update configuration"""
        await self.engine.update_config(payload)
        return {"status": "config_updated"}
    
    async def cmd_get_status(self, payload: dict) -> dict:
        """Get engine status"""
        return {
            "trading_active": self.engine.trading_active,
            "connected": self.engine.is_connected(),
            "skills_loaded": len(self.skill_executor.loaded_skills),
            "uptime_seconds": self.engine.get_uptime(),
            "ai_enabled": self.signal_generator.model is not None
        }
    
    # === Phase 3: AI Signal Commands ===
    
    async def cmd_generate_signal(self, payload: dict) -> dict:
        """Generate an AI trading signal for a symbol"""
        symbol = payload.get("symbol", "BTC/USDT")
        
        # Get market data
        market_data = await self.engine.get_market_data(symbol)
        
        # Generate signal using AI
        signal = await self.signal_generator.generate_signal(
            symbol=symbol,
            market_data=market_data,
            portfolio_balance=self.engine.portfolio.get_balance()
        )
        
        signal_dict = signal.to_dict()
        if payload.get("shadow_mode", True):
            self._record_shadow_decision(symbol, signal_dict)
        return signal_dict
    
    async def cmd_get_last_signal(self, payload: dict) -> dict:
        """Get the last generated signal for a symbol"""
        symbol = payload.get("symbol", "BTC/USDT")
        signal = self.signal_generator.get_last_signal(symbol)
        
        if signal:
            return signal.to_dict()
        return {"error": "No signal found for symbol"}
    
    async def cmd_reload_skills(self, payload: dict) -> dict:
        """Manually trigger skill reload"""
        old_count = len(self.skill_executor.loaded_skills)
        self.skill_executor.reload_skills()
        new_count = len(self.skill_executor.loaded_skills)
        
        return {
            "status": "reloaded",
            "old_count": old_count,
            "new_count": new_count
        }

    async def cmd_get_shadow_report(self, payload: dict) -> dict:
        """Return shadow mode comparison dashboard metrics."""
        window_minutes = int(payload.get("window_minutes", 60))
        return self.shadow_monitor.summary(window_minutes=window_minutes)

    def _record_shadow_decision(self, symbol: str, signal: dict) -> None:
        """Store dry-run decision and compare against baseline strategy."""
        balance = float(self.engine.portfolio.get_balance())
        risk_pct = float(signal.get("amount") or 0.0)
        size = balance * risk_pct
        decision = ShadowDecision(
            timestamp=float(signal.get("timestamp", 0.0)) or self.engine.get_server_time(),
            symbol=symbol,
            entry=signal.get("entry_price"),
            size=size,
            risk=risk_pct,
            exit=signal.get("take_profit"),
            action=signal.get("action", "HOLD"),
            confidence=float(signal.get("confidence", 0.0)),
        )
        self.shadow_monitor.record_decision(decision, baseline=False)
        baseline_action = "HOLD" if decision.confidence < 0.55 else decision.action
        baseline = ShadowDecision(
            timestamp=decision.timestamp,
            symbol=symbol,
            entry=decision.entry,
            size=size * 0.9,
            risk=risk_pct * 0.9,
            exit=decision.exit,
            action=baseline_action,
            confidence=decision.confidence,
        )
        self.shadow_monitor.record_decision(baseline, baseline=True)


async def main():
    """Main entry point"""
    app = MoneyMachineApp()
    
    try:
        await app.initialize()
        await app.start()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        if app.hot_reload:
            app.hot_reload.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
