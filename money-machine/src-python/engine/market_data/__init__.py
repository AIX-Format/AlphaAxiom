"""
Historical and live market data ingestion.

Two layers:

- `MarketDataClient` (Protocol): pluggable async source of OHLCV
  candles. Production uses `BinancePublicClient` which calls the
  Binance public klines REST endpoint (no API key required, rate
  limited but generous for backtest seeding).
- `MarketDataService`: caches fetched candles to a local Parquet
  store (CSV fallback if pyarrow is missing) and serves them back
  as pandas DataFrames with the canonical OHLCV columns the
  indicators and backtest modules expect.

The split keeps the cache logic free of HTTP concerns and lets us
swap in a CCXT-backed client, a Tinybird-backed client, or a stub
client in tests without rewriting the cache.
"""

from .binance_client import BinancePublicClient
from .protocols import Candle, MarketDataClient
from .service import MarketDataService

__all__ = [
    "BinancePublicClient",
    "Candle",
    "MarketDataClient",
    "MarketDataService",
]
