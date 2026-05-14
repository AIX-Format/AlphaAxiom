"""
Core trading engine with CCXT integration
"""

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Any
import os

CONFIG_LIMITS = {
    "max_risk_per_trade": (0.0, 0.1, True),
    "max_daily_loss": (0.0, 0.2, True),
}


class Portfolio:
    """Manages portfolio state, balance, and positions.

    Tracks the running totals the risk shield needs:
      - balance: current equity.
      - high_water_mark(): the all-time peak balance, lazily updated
        on each balance read so it never drifts.
      - daily_pnl(): realized PnL accumulated since UTC midnight.
        Resets the moment the calling clock crosses into a new day.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.trades: List[Dict] = []
        self.positions: Dict[str, Any] = {}
        self._high_water_mark = float(initial_balance)
        self._daily_pnl = 0.0
        self._daily_pnl_day: Optional[date] = None

    def get_balance(self) -> float:
        # Touching the balance also bumps the HWM if we are at a new
        # peak; cheaper than a separate accounting pass and keeps the
        # value honest for any external reader.
        if self.balance > self._high_water_mark:
            self._high_water_mark = self.balance
        return self.balance

    def get_positions(self) -> Dict:
        return self.positions

    def calculate_pnl(self) -> float:
        """Calculate total unrealised profit/loss across open positions."""
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.get('pnl', 0.0)
        return total_pnl

    def daily_pnl(self, *, today: Optional[date] = None) -> float:
        """Realised PnL accumulated for the current trading day.

        `today` is injectable for tests. Defaults to UTC today. The
        counter resets automatically the moment a different date is
        observed, so callers do not have to remember to roll it over.
        """
        today = today or datetime.now(tz=timezone.utc).date()
        if self._daily_pnl_day != today:
            self._daily_pnl = 0.0
            self._daily_pnl_day = today
        return self._daily_pnl

    def high_water_mark(self) -> float:
        # Refresh in case balance grew without a trade being recorded
        # through add_trade (e.g. mark-to-market updates).
        if self.balance > self._high_water_mark:
            self._high_water_mark = self.balance
        return self._high_water_mark

    def add_trade(self, trade: Dict, *, today: Optional[date] = None) -> None:
        self.trades.append(trade)
        pnl = trade.get('pnl', 0.0)
        if pnl:
            self.balance += pnl
            # Keep daily_pnl current with the same day-rollover logic.
            today = today or datetime.now(tz=timezone.utc).date()
            if self._daily_pnl_day != today:
                self._daily_pnl = 0.0
                self._daily_pnl_day = today
            self._daily_pnl += pnl
        if self.balance > self._high_water_mark:
            self._high_water_mark = self.balance


class TradingEngine:
    """Core trading engine with exchange connectivity"""
    
    def __init__(self, config: dict):
        self.config = config
        self.exchange = None
        self.portfolio = Portfolio(config.get('initial_balance', 10000.0))
        self.trading_active = False
        self.market_data_cache = {}
        self.start_time = datetime.now()
        self._connected = False
    
    async def initialize(self):
        """Initialize exchange connection"""
        try:
            # Lazy import ccxt to avoid issues if not installed
            import ccxt.async_support as ccxt
            
            exchange_config = self.config.get('exchange', {})
            exchange_name = exchange_config.get('name', 'binance')
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_name, None)
            if exchange_class:
                self.exchange = exchange_class({
                    'apiKey': exchange_config.get('api_key', ''),
                    'secret': exchange_config.get('secret', ''),
                    'enableRateLimit': True,
                    'sandbox': exchange_config.get('sandbox', True),
                })
                
                # Load markets
                await self.exchange.load_markets()
                self._connected = True
        except ImportError:
            print("CCXT not installed. Running in mock mode.")
            self._connected = False
        except Exception as e:
            print(f"Exchange initialization error: {e}")
            self._connected = False
    
    async def get_market_data(self, symbol: str = "BTC/USDT", 
                             timeframe: str = "5m") -> List[List]:
        """Fetch OHLCV data"""
        if not self.exchange:
            # Return mock data
            return [[datetime.now().timestamp() * 1000, 50000, 50100, 49900, 50050, 100]]
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            return ohlcv
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return []
    
    async def execute_trade(self, trade_params: dict) -> dict:
        """Execute a trade based on params"""
        if not self.exchange:
            # Mock execution
            return {
                "success": True, 
                "order_id": f"mock_{datetime.now().timestamp()}",
                "message": "Mock execution (no exchange connected)"
            }
        
        try:
            symbol = trade_params['symbol']
            order_type = trade_params['order_type']  # 'buy' or 'sell'
            amount = trade_params['amount']
            price = trade_params.get('price')
            
            if order_type == 'buy':
                if price:
                    order = await self.exchange.create_limit_buy_order(symbol, amount, price)
                else:
                    order = await self.exchange.create_market_buy_order(symbol, amount)
            else:
                if price:
                    order = await self.exchange.create_limit_sell_order(symbol, amount, price)
                else:
                    order = await self.exchange.create_market_sell_order(symbol, amount)
            
            # Update portfolio
            self.portfolio.add_trade(order)
            
            return {"success": True, "order_id": order['id']}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_server_time(self) -> float:
        return datetime.now().timestamp()
    
    def is_connected(self) -> bool:
        return self._connected
    
    def get_uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    async def update_config(self, new_config: dict):
        """Update safe runtime configuration on the fly."""
        self.config.update(validate_config_update(new_config))
    
    async def close(self):
        """Cleanup resources"""
        if self.exchange:
            await self.exchange.close()


def validate_config_update(new_config: dict) -> dict:
    if not isinstance(new_config, dict):
        raise ValueError("config update must be an object")
    validated: dict = {}
    for key, raw_value in new_config.items():
        if key not in CONFIG_LIMITS:
            raise ValueError(f"unsupported config.{key}")
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise ValueError(f"config.{key} must be numeric")

        value = float(raw_value)
        minimum, maximum, exclusive_minimum = CONFIG_LIMITS[key]
        above_minimum = value > minimum if exclusive_minimum else value >= minimum
        if not above_minimum or value > maximum:
            raise ValueError(f"config.{key} is outside allowed range")
        validated[key] = value
    return validated
