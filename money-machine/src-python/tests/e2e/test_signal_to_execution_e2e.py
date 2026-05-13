from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

SRC_PYTHON = Path(__file__).resolve().parents[2]
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.exceptions import InvalidSignature  # noqa: E402

from engine.adapters import (  # noqa: E402
    HttpResponse,
    MT5Adapter,
    MT5Config,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperAdapter,
    canonical_payload,
    inline_signer,
)
from engine.risk_shield import RejectionReason, RiskConfig, RiskShield  # noqa: E402
from engine.signal_pipeline import PipelineConfig, SignalPipeline  # noqa: E402
from engine.strategies.base import Strategy, TradingSignal  # noqa: E402


TEST_SEED = b"b" * 32


def _run(coro):
    return asyncio.run(coro)


def ohlcv_fixture() -> pd.DataFrame:
    rows = [
        [1717200000000, 68000.0, 68450.0, 67750.0, 68300.0, 120.0],
        [1717203600000, 68300.0, 68800.0, 68220.0, 68720.0, 118.0],
        [1717207200000, 68720.0, 68950.0, 68510.0, 68840.0, 125.0],
    ]
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class FixedBuyStrategy(Strategy):
    name = "e2e-fixed-buy"

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        super().__init__(symbol)

    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        close = float(df["close"].iloc[-1])
        return TradingSignal(
            symbol=self.symbol,
            action="BUY",
            confidence=0.9,
            strategy=self.name,
            entry_price=close,
            stop_loss=close * 0.98,
            take_profit=close * 1.03,
            reasoning="deterministic e2e fixture",
        )


class _FakeMt5Http:
    def __init__(self) -> None:
        self.calls = []

    async def __call__(self, method: str, url: str, headers: Dict[str, str], body: bytes) -> HttpResponse:
        self.calls.append((method, url, dict(headers), body))
        return HttpResponse(status=202, body=b'{"venue_order_id":"relay-1"}', headers={})


def _public_key():
    return Ed25519PrivateKey.from_private_bytes(TEST_SEED).public_key()


def _request_from_result(result, client_order_id: str) -> OrderRequest:
    side = OrderSide.BUY if result.signal.action == "BUY" else OrderSide.SELL
    return OrderRequest(
        client_order_id=client_order_id,
        symbol=result.signal.symbol,
        side=side,
        order_type=OrderType.MARKET,
        notional=result.position_size,
        stop_loss=result.signal.stop_loss,
        take_profit=result.signal.take_profit,
        strategy=result.signal.strategy,
    )


def test_e2e_paper_adapter_path_with_idempotency_and_size_assertions() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=20_000.0)
        adapter.set_mark_price("BTC/USDT", 68_840.0)

        pipeline = SignalPipeline(
            strategy=FixedBuyStrategy(),
            risk_shield=RiskShield(),
            adapter=adapter,
            config=PipelineConfig(risk_per_trade_pct=0.003, fallback_stop_pct=0.02),
            execute_live=False,
        )

        result = await pipeline.run_once(ohlcv_fixture())
        assert result.risk_decision.approved is True
        assert result.position_size == pytest.approx(3000.0, rel=1e-4)

        req = _request_from_result(result, "paper-ord-1")
        first = await adapter.place_order(req)
        second = await adapter.place_order(req)

        assert first.status is OrderStatus.FILLED
        assert second == first
        assert (await adapter.get_positions())["BTC/USDT"] > 0

    _run(scenario())


def test_e2e_rejects_unsafe_trade_via_risk_shield() -> None:
    async def scenario() -> None:
        adapter = PaperAdapter(initial_balance=10_000.0)
        adapter.set_mark_price("BTC/USDT", 68_840.0)
        shield = RiskShield(config=RiskConfig(max_position_size_pct=0.05))
        pipeline = SignalPipeline(
            strategy=FixedBuyStrategy(),
            risk_shield=shield,
            adapter=adapter,
            config=PipelineConfig(risk_per_trade_pct=0.003, fallback_stop_pct=0.02),
            execute_live=False,
        )

        result = await pipeline.run_once(ohlcv_fixture())
        assert result.risk_decision.approved is False
        assert result.risk_decision.reason == RejectionReason.POSITION_SIZE_EXCEEDED

    _run(scenario())


def test_e2e_mt5_mock_signature_verification_and_no_network() -> None:
    async def scenario() -> None:
        http = _FakeMt5Http()
        adapter = MT5Adapter(
            http_client=http,
            signer=inline_signer(TEST_SEED),
            config=MT5Config(oracle_url="https://oracle.mock", max_retries=0),
            account_equity=20_000.0,
        )

        pipeline = SignalPipeline(
            strategy=FixedBuyStrategy(symbol="EURUSD"),
            risk_shield=RiskShield(),
            adapter=adapter,
            config=PipelineConfig(risk_per_trade_pct=0.003, fallback_stop_pct=0.02),
            execute_live=True,
        )

        result = await pipeline.run_once(ohlcv_fixture())

        assert result.executed is True
        assert result.execution["status"] == "PENDING"
        assert result.position_size == pytest.approx(3000.0, rel=1e-4)
        assert len(http.calls) == 1

        _, _, _, body = http.calls[0]
        envelope = json.loads(body)
        signature = bytes.fromhex(envelope["signature"])
        canonical = canonical_payload(envelope["payload"])
        _public_key().verify(signature, canonical)

        tampered = dict(envelope["payload"])
        tampered["symbol"] = "GBPUSD"
        with pytest.raises(InvalidSignature):
            _public_key().verify(signature, canonical_payload(tampered))

    _run(scenario())
