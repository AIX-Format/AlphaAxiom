from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from engine.adapters import EVMAdapter, MT5Adapter, OrderRequest, OrderSide, OrderStatus, OrderType, PaperAdapter
from engine.adapters.evm import EVMAdapterConfig, TxReceipt
from engine.adapters.mt5 import HttpResponse, MT5Config


def _run(coro):
    return asyncio.run(coro)


def _sample_request(client_id: str = "cid-1") -> OrderRequest:
    return OrderRequest(
        client_order_id=client_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=100.0,
    )


def test_paper_adapter_contract_place_order_returns_order_result_shape() -> None:
    adapter = PaperAdapter(initial_balance=1_000.0)
    adapter.set_mark_price("BTC/USDT", 50_000.0)
    result = _run(adapter.place_order(_sample_request()))
    assert result.client_order_id == "cid-1"
    assert result.status in {OrderStatus.FILLED, OrderStatus.PENDING, OrderStatus.REJECTED}


async def _ok_http(method: str, url: str, headers: dict[str, str], body: bytes) -> HttpResponse:
    return HttpResponse(status=200, body=b'{"ok":true,"id":"remote-1"}', headers={})


def test_mt5_adapter_contract_place_order_returns_order_result_shape() -> None:
    adapter = MT5Adapter(
        http_client=_ok_http,
        signer=lambda payload: b"x" * 64,
        config=MT5Config(oracle_url="https://example.com", max_retries=1),
        account_equity=1_000.0,
    )
    result = _run(adapter.place_order(_sample_request()))
    assert result.client_order_id == "cid-1"
    assert result.status in {OrderStatus.PENDING, OrderStatus.REJECTED, OrderStatus.FILLED}


async def _rpc(method: str, params: list):
    if method == "eth_getTransactionCount":
        return hex(7)
    if method == "eth_chainId":
        return hex(1)
    if method == "eth_gasPrice":
        return hex(20 * 10**9)
    if method == "eth_sendRawTransaction":
        return "0xabc"
    if method == "eth_getTransactionReceipt":
        return {
            "transactionHash": "0xabc",
            "status": hex(1),
            "blockNumber": hex(1),
            "gasUsed": hex(21000),
        }
    if method == "eth_call":
        return "0x"
    return "0x0"


def test_evm_adapter_contract_place_order_returns_order_result_shape() -> None:
    adapter = EVMAdapter(
        rpc=_rpc,
        private_key=b"\x01" * 32,
        account_address="0x1111111111111111111111111111111111111111",
        config=EVMAdapterConfig(confirm_poll_seconds=0.01, confirm_timeout_seconds=0.1),
    )
    req = OrderRequest(
        client_order_id="cid-1",
        symbol="WETH/USDC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=10.0,
        metadata={
            "from_address": "0x2222222222222222222222222222222222222222",
            "to_address": "0x3333333333333333333333333333333333333333",
            "from_decimals": 18,
            "to_decimals": 6,
            "expected_amount_out": 0.001,
        },
    )
    result = _run(adapter.place_order(req))
    assert result.client_order_id == "cid-1"
    assert result.status in {OrderStatus.FILLED, OrderStatus.PENDING, OrderStatus.REJECTED}
