# scripts/

Operator-facing CLIs that consume the trading engine through its
public primitives. Nothing here touches `engine/adapters/` or holds
secrets; this directory is research-only.

## `research.py`, zero-key crypto research CLI

Fetches public OHLCV from Binance (no API key required), caches it
locally, runs every requested strategy plus a buy-and-hold baseline
against the same data, and writes a comparison report to disk.

### Quick start

From `money-machine/src-python/`:

```bash
python -m scripts.research \
    --symbol BTC/USDT \
    --interval 1h \
    --since 2024-01-01 \
    --until 2024-12-31
```

That command will:

1. Fetch hourly BTC/USDT klines for the whole of 2024 from Binance's
   public REST endpoint, no key required (`engine.market_data.binance_client`).
2. Cache the candles under `.cache/ohlcv/BTCUSDT_1h.csv` so re-runs
   are free (`engine.market_data.service`).
3. Run `momentum`, `mean_reversion`, `breakout`, and `buy_and_hold`
   through the same `engine.backtest.Backtest` pipeline used by the
   rest of the system.
4. Write a comparison table to `.research/BTC_USDT_1h_2024-01-01_2024-12-31/`
   containing:
     - `report.json`, machine-readable metrics for every strategy
     - `report.md`, the same table in markdown
     - `equity_<strategy>.csv`, per-strategy equity curve
     - `equity.png`, equity-curve plot if `matplotlib` is installed

### Why this exists

`main.py` is the IPC server for the Tauri frontend, not a research
tool. Before this script there was no command-line entry point for
"compare these strategies on this symbol for this window". This
script fills that gap without touching the engine.

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--symbol` | `BTC/USDT` | Use the unified ccxt-style format with the slash |
| `--interval` | `1h` | Any value in `engine.market_data.protocols.INTERVAL_MS` |
| `--since` | `2024-01-01` | UTC inclusive start, YYYY-MM-DD |
| `--until` | `2024-12-31` | UTC exclusive end, YYYY-MM-DD |
| `--equity` | `10_000.0` | Initial equity per strategy |
| `--strategies` | `momentum,mean_reversion,breakout,buy_and_hold` | Comma-separated, see registry below |
| `--cache-dir` | `.cache/ohlcv` | Per-pair CSV cache directory |
| `--out-dir` | `.research` | Top-level report directory |
| `--no-plot` | `False` | Skip the equity plot even when matplotlib is installed |
| `--base-url` | `https://data-api.binance.vision` | Binance host. The default is the public market-data subdomain, which is the most geo-permissive option. Use `https://api.binance.com` for the canonical endpoint. |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Strategy registry

The CLI accepts these names in `--strategies`:

| Name | Class | Tier |
|------|-------|------|
| `momentum` | `engine.strategies.momentum.MomentumStrategy` | Active |
| `mean_reversion` | `engine.strategies.mean_reversion.MeanReversionStrategy` | Active |
| `breakout` | `engine.strategies.breakout.BreakoutStrategy` | Active |
| `buy_and_hold` | `scripts.research.BuyAndHoldStrategy` | Baseline |

`buy_and_hold` is implemented inside the script as a regular
`Strategy` subclass so it runs through the same commission and
slippage models as the active strategies. That makes the comparison
fair.

### What gets measured

Every strategy reports the standard engine metrics from
`engine.backtest.BacktestMetrics` (total return, Sharpe, max
drawdown, win rate, trade counts) plus the extended metrics from
`scripts.research_metrics.ExtendedMetrics`:

| Metric | Definition |
|--------|------------|
| Sortino | Mean return over downside-only std, annualised |
| Calmar | Total return divided by max drawdown |
| Profit factor | Gross wins over gross losses (`inf` if no losers) |
| Expectancy | Average pnl per trade in dollars |
| Exposure % | Share of bars with an open position |
| Best/worst trade | Largest and smallest single-trade pnl |

### Privacy and cost guarantees

- **Zero API keys.** The Binance public klines endpoint is anonymous.
- **Zero subscription cost.** No paid data plan required.
- **No outbound credentials.** This script never reads, requires, or
  transmits any secret from `engine/wallet/`, `keyring`, or `.env`.
- **No execution.** This script never imports `engine/adapters/*`
  and has no path to placing an order.

### Rate-limit hygiene

Binance's anonymous limit is roughly 1200 requests per minute per
IP. A six-month hourly backfill is well under 10 requests. The
`MarketDataService` cache makes subsequent runs fetch nothing. If
you are scripting many parallel runs against different pairs,
serialise them or expect HTTP 429 from the venue.

### Geo-restrictions

Binance applies geo-restrictions to `api.binance.com` for residents
of several jurisdictions, returning HTTP 451 ("Unavailable For
Legal Reasons"). To stay zero-key and zero-account, the CLI
defaults to `https://data-api.binance.vision`, Binance's public
market-data subdomain, which serves the same `/api/v3/klines`
endpoint with broader regional availability. Override with
`--base-url https://api.binance.com` if you have an unrestricted
network path; both hosts return identical schemas so the rest of
the engine does not care.

If you see HTTP 451 with `data-api.binance.vision`, your IP is on
the restricted list. A VPN to an unrestricted region or a different
public exchange (Kraken, Bybit) is the fix; both are out of scope
for this script, which is Binance-only by design to match the
engine's `BinancePublicClient`.
