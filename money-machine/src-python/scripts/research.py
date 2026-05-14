#!/usr/bin/env python3
"""
Zero-key crypto research CLI.

Fetches public OHLCV from Binance (no API key required, see
`engine.market_data.binance_client`), caches it locally via
`MarketDataService`, runs the three built-in strategies plus a
buy-and-hold baseline against the same data, and writes a comparison
report to disk.

Usage:

    python -m scripts.research \\
        --symbol BTC/USDT \\
        --interval 1h \\
        --since 2024-01-01 \\
        --until 2024-12-31 \\
        --out-dir .research

Output:

    <out-dir>/<symbol>_<interval>_<since>_<until>/
        report.json           machine-readable metrics for every strategy
        report.md             human-readable table
        equity_<strat>.csv    equity curve per strategy
        equity.png            (only if matplotlib is installed)

This script never touches `engine.adapters`, never holds credentials,
and never executes against a live venue. It is research-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pandas as pd

# Make `engine.*` importable when this file is run via
# `python -m scripts.research` from `money-machine/src-python/`.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from engine.backtest import (  # noqa: E402
    AtrSlippage,
    Backtest,
    BacktestConfig,
    BINANCE_SPOT_TAKER,
    BacktestResult,
)
from engine.market_data.binance_client import (  # noqa: E402
    BinancePublicClient,
    BinancePublicConfig,
)
from engine.market_data.service import MarketDataService  # noqa: E402
from engine.strategies.base import Strategy, TradingSignal  # noqa: E402
from engine.strategies.breakout import BreakoutStrategy  # noqa: E402
from engine.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402
from engine.strategies.momentum import MomentumStrategy  # noqa: E402

from scripts.research_metrics import (  # noqa: E402
    ExtendedMetrics,
    compute_extended_metrics,
)


logger = logging.getLogger("scripts.research")


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------


class BuyAndHoldStrategy(Strategy):
    """Single-shot BUY on the first eligible bar; never flips.

    Implemented as a regular strategy so the backtest engine runs it
    under the same slippage and commission rules as the active
    strategies. The position is force-closed at end_of_data, which
    matches what a real holder would have realised had they sold on
    the final bar.
    """

    name = "buy-and-hold"

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self._fired = False

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        # First call past warm-up wins.
        if self._fired or df is None or df.empty:
            return self._hold("buy-and-hold already entered")
        last_close = float(df["close"].iloc[-1])
        self._fired = True
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=1.0,
            strategy=self.name,
            entry_price=last_close,
            stop_loss=None,
            take_profit=None,
            reasoning="buy-and-hold baseline",
        )


# ---------------------------------------------------------------------------
# HTTP transport: aiohttp with stdlib fallback
# ---------------------------------------------------------------------------


def _build_http_fetch() -> Callable[[str, Dict[str, Any]], Awaitable[bytes]]:
    """Return an async fetcher that performs HTTP GET.

    Prefers aiohttp (already a hard dependency of the engine per
    requirements.txt), but falls back to a stdlib urllib path so the
    CLI also runs in minimal environments where the dependency tree
    has been pruned.
    """
    try:
        import aiohttp  # type: ignore

        async def _aiohttp_fetch(
            url: str, params: Dict[str, Any]
        ) -> bytes:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.read()

        return _aiohttp_fetch
    except ImportError:
        pass

    import urllib.parse
    import urllib.request

    async def _stdlib_fetch(url: str, params: Dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if query else url

        def _do_request() -> bytes:
            req = urllib.request.Request(
                full_url, headers={"User-Agent": "alphaaxiom-research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return resp.read()

        return await asyncio.get_running_loop().run_in_executor(None, _do_request)

    return _stdlib_fetch


# ---------------------------------------------------------------------------
# Strategy roster
# ---------------------------------------------------------------------------


def _build_strategies(symbol: str, names: List[str]) -> List[Strategy]:
    """Materialise the requested strategies with default parameters.

    `names` is the comma-separated list from --strategies. Unknown
    names are surfaced as a parser-level error rather than silently
    skipped.
    """
    registry: Dict[str, Callable[[str], Strategy]] = {
        "momentum": lambda s: MomentumStrategy(s),
        "mean_reversion": lambda s: MeanReversionStrategy(s),
        "breakout": lambda s: BreakoutStrategy(s),
        "buy_and_hold": lambda s: BuyAndHoldStrategy(s),
    }
    out: List[Strategy] = []
    for n in names:
        key = n.strip().lower()
        if key not in registry:
            raise SystemExit(
                f"unknown strategy {key!r}. Known: {sorted(registry)}"
            )
        out.append(registry[key](symbol))
    return out


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


@dataclass
class StrategyReport:
    name: str
    metrics: Dict[str, Any]            # BacktestMetrics.to_dict()
    extended: Dict[str, Any]           # ExtendedMetrics.to_dict()
    num_trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "extended": self.extended,
            "num_trades": self.num_trades,
        }


def _render_markdown_table(reports: List[StrategyReport]) -> str:
    """Produce a human-readable markdown table of the comparison."""
    header = (
        "| Strategy | Total Return | Sharpe | Sortino | Calmar | "
        "Max DD | Win Rate | Profit Factor | Trades | Exposure |"
    )
    sep = (
        "|----------|--------------|--------|---------|--------|"
        "--------|----------|---------------|--------|----------|"
    )
    rows = [header, sep]
    for r in reports:
        m = r.metrics
        e = r.extended
        pf = e["profit_factor"]
        pf_str = "inf" if math.isinf(pf) else f"{pf:.2f}"
        rows.append(
            f"| {r.name} "
            f"| {m['total_return']*100:7.2f}% "
            f"| {m['sharpe']:6.2f} "
            f"| {e['sortino']:7.2f} "
            f"| {e['calmar']:6.2f} "
            f"| {m['max_drawdown']*100:6.2f}% "
            f"| {m['win_rate']*100:6.2f}% "
            f"| {pf_str:>9s} "
            f"| {r.num_trades:6d} "
            f"| {e['exposure_pct']*100:6.2f}% |"
        )
    return "\n".join(rows)


def _maybe_plot(
    out_dir: Path,
    equity_curves: Dict[str, pd.Series],
) -> Optional[Path]:
    """Render an equity-curve plot if matplotlib is importable.

    Falls back silently when matplotlib is not installed; the CSV and
    JSON outputs alone are sufficient and the research CLI has no
    hard plotting dependency.
    """
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # headless safe default
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.info(
            "matplotlib not installed; skipping equity plot. "
            "Install matplotlib to enable rendering."
        )
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("equity")
    ax.set_title("Equity curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "equity.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_iso_date(s: str) -> int:
    """Parse YYYY-MM-DD into UTC epoch ms (00:00 of that date)."""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(f"--since/--until must be YYYY-MM-DD ({exc})")
    return int(dt.timestamp() * 1000)


async def _run_async(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    safe_symbol = args.symbol.replace("/", "_").replace(":", "_")
    run_dir = (
        out_root
        / f"{safe_symbol}_{args.interval}_{args.since}_{args.until}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_ms = _parse_iso_date(args.since)
    end_ms = _parse_iso_date(args.until)
    if end_ms <= start_ms:
        raise SystemExit("--until must be after --since")

    client = BinancePublicClient(
        http_fetch=_build_http_fetch(),
        config=BinancePublicConfig(base_url=args.base_url),
    )
    service = MarketDataService(client=client, cache_dir=cache_dir)
    logger.info(
        "fetching %s %s [%s, %s) into %s",
        args.symbol, args.interval, args.since, args.until, cache_dir,
    )
    df = await service.get_ohlcv(
        args.symbol, args.interval, start_ms, end_ms
    )
    if df.empty:
        raise SystemExit(
            "no candles returned. Check symbol spelling and date range; "
            "Binance public endpoint blocks some regions and unknown symbols."
        )
    logger.info("loaded %d candles spanning %s -> %s", len(df), df.index[0], df.index[-1])

    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = _build_strategies(args.symbol, strategy_names)

    reports: List[StrategyReport] = []
    equity_curves: Dict[str, pd.Series] = {}
    for strat in strategies:
        logger.info("running backtest for %s", strat.name)
        bt = Backtest(
            strategy=strat,
            commission=BINANCE_SPOT_TAKER,
            slippage=AtrSlippage(multiplier=0.1, floor_bps=1.0),
            config=BacktestConfig(initial_equity=args.equity),
        )
        result: BacktestResult = bt.run(df)
        extended = compute_extended_metrics(
            result.equity_curve,
            result.trades,
            initial_equity=args.equity,
        )
        reports.append(
            StrategyReport(
                name=strat.name,
                metrics=result.metrics.to_dict(),
                extended=extended.to_dict(),
                num_trades=len(result.trades),
            )
        )
        equity_curves[strat.name] = result.equity_curve
        # Persist per-strategy equity curve for downstream tools.
        result.equity_curve.to_csv(run_dir / f"equity_{strat.name}.csv")

    # JSON output, with infinities serialised as the literal string
    # "inf" rather than the JSON-incompatible float. Downstream
    # consumers can pattern-match on it.
    def _json_default(o: Any) -> Any:
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    json_path = run_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "interval": args.interval,
                "since": args.since,
                "until": args.until,
                "initial_equity": args.equity,
                "candle_count": len(df),
                "first_bar": df.index[0].isoformat(),
                "last_bar": df.index[-1].isoformat(),
                "reports": [r.to_dict() for r in reports],
            },
            indent=2,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    md_path = run_dir / "report.md"
    md_path.write_text(
        f"# Research report: {args.symbol} {args.interval}\n\n"
        f"- Window: {args.since} -> {args.until}\n"
        f"- Candles: {len(df)}\n"
        f"- Initial equity: ${args.equity:,.2f}\n\n"
        + _render_markdown_table(reports)
        + "\n",
        encoding="utf-8",
    )

    if not args.no_plot:
        plot_path = _maybe_plot(run_dir, equity_curves)
        if plot_path is not None:
            logger.info("wrote plot to %s", plot_path)

    # Echo the table to stdout for shell-pipeline use.
    print(_render_markdown_table(reports))
    print(f"\nFull report: {run_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-key crypto research CLI for AlphaAxiom.",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="e.g. BTC/USDT")
    parser.add_argument(
        "--interval",
        default="1h",
        help="kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, ...)",
    )
    parser.add_argument(
        "--since",
        default="2024-01-01",
        help="UTC inclusive start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--until",
        default="2024-12-31",
        help="UTC exclusive end date YYYY-MM-DD",
    )
    parser.add_argument(
        "--equity", type=float, default=10_000.0, help="initial equity"
    )
    parser.add_argument(
        "--strategies",
        default="momentum,mean_reversion,breakout,buy_and_hold",
        help="comma-separated strategy names",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/ohlcv",
        help="OHLCV CSV cache directory",
    )
    parser.add_argument(
        "--out-dir",
        default=".research",
        help="output directory for reports",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the equity-curve plot even when matplotlib is available",
    )
    parser.add_argument(
        "--base-url",
        default="https://data-api.binance.vision",
        help=(
            "Binance REST host. Default is `data-api.binance.vision`, "
            "Binance's public market-data subdomain, which is the most "
            "geo-permissive option. Use `https://api.binance.com` if "
            "you need the canonical endpoint."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
