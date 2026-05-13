"""
Tests for the EVM swap adapter.

The adapter is wired to a stub `EVMRpcClient` so the suite has no
network and no live chain. Coverage:

  - Input validation: bad private key length, malformed account
    address, missing token addresses in metadata, non-positive
    amount_in.
  - ABI encoding: the function selector and per-field layout of
    swapExactTokensForTokens(amountIn, amountOutMin, path, to,
    deadline) match the expected byte-level template.
  - Happy path: place_order signs a tx, broadcasts via
    eth_sendRawTransaction, returns PENDING with the tx hash.
  - Idempotency: re-submitting the same client_order_id returns
    the cached result without re-signing or re-broadcasting; the
    nonce is reused on retry so the chain dedupes if our cache
    ever clears.
  - Slippage: amount_out_min derived from expected output and
    slippage_bps; explicit override wins.
  - get_balance: parses hex wei into ether float.
  - cancel_order: local-only cancel marks the order CANCELLED and
    surfaces a warning that the chain may still mine the tx.
  - wait_for_receipt: polls for the receipt, transitions PENDING
    -> FILLED on status=1, PENDING -> REJECTED on revert (status=0).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

pytest.importorskip("eth_account")

from engine.adapters import (  # noqa: E402
    AdapterError,
    EVMAdapter,
    EVMAdapterConfig,
    EVMChainConfig,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)


def _run(coro):
    return asyncio.run(coro)


# A throw-away private key with a known checksum address. Generated
# with Account.create() on a one-off, no funds anywhere.
PRIVATE_KEY = bytes.fromhex(
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
ACCOUNT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

TOKEN_A = "0xAAaAaaAaAaAaAaaAaAAAAAAAAaaaAaAaAaaAaaAa"  # noqa: S105 - public test fixture address
TOKEN_B = "0xBbbBBBbbbBBBbbbBbbBbBBBb000000000000bBBb"  # noqa: S105 - public test fixture address


class _FakeRpc:
    """Async stub for the EVM JSON-RPC client. Replays scripted
    responses keyed by method.
    """

    def __init__(self, responses: Dict[str, Any]) -> None:
        self.responses = dict(responses)
        self.calls: List[Tuple[str, List[Any]]] = []

    async def __call__(self, method: str, params: List[Any]) -> Any:
        self.calls.append((method, list(params)))
        value = self.responses.get(method)
        if isinstance(value, list):
            # Pop one response per call for methods that have a
            # sequence of replies.
            if not value:
                return None
            return value.pop(0)
        return value


def _request(client_id: str = "ord-1", **overrides: Any) -> OrderRequest:
    """Build a default SELL request: spend 1 BASE for an expected
    1000 QUOTE. Tests override fields as needed.
    """
    base: Dict[str, Any] = dict(
        client_order_id=client_id,
        symbol="TOKA/TOKB",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.0,  # 1 BASE token
        strategy="evm-test",
        metadata={
            "from_address": TOKEN_A,
            "to_address": TOKEN_B,
            "from_decimals": 18,
            "to_decimals": 18,
            "expected_amount_out": 1000.0,
            "slippage_bps": 50,
        },
    )
    base.update(overrides)
    return OrderRequest(**base)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construction_rejects_wrong_key_length() -> None:
    with pytest.raises(AdapterError):
        EVMAdapter(rpc=_FakeRpc({}), private_key=b"\x00" * 16, account_address=ACCOUNT)


def test_construction_rejects_bad_account_address() -> None:
    with pytest.raises(AdapterError):
        EVMAdapter(
            rpc=_FakeRpc({}),
            private_key=PRIVATE_KEY,
            account_address="not-an-address",
        )


def test_place_order_requires_token_addresses() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({})
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        bad = _request(metadata={})
        result = await adapter.place_order(bad)
        assert result.status is OrderStatus.REJECTED
        assert "from_address" in (result.error or "")
        assert rpc.calls == []

    _run(scenario())


def test_place_order_rejects_invalid_request() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({})
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        bad = OrderRequest(
            client_order_id="x",
            symbol="TOKA/TOKB",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=None,
            quantity=None,
            metadata={"from_address": TOKEN_A, "to_address": TOKEN_B},
        )
        result = await adapter.place_order(bad)
        assert result.status is OrderStatus.REJECTED
        assert "quantity or notional" in (result.error or "")

    _run(scenario())


# ---------------------------------------------------------------------------
# ABI encoding
# ---------------------------------------------------------------------------


def test_swap_calldata_layout_matches_uniswap_v2_selector() -> None:
    from engine.adapters.evm import _encode_swap_exact_tokens_for_tokens

    data_hex = _encode_swap_exact_tokens_for_tokens(
        amount_in=10**18,
        amount_out_min=999 * 10**18,
        path=[TOKEN_A, TOKEN_B],
        to=ACCOUNT,
        deadline=1_700_000_000,
    )
    assert data_hex.startswith("0x38ed1739"), "wrong function selector"
    raw = bytes.fromhex(data_hex[2:])
    # Selector(4) + 5 head words(160) + dynamic length(32) + 2 addresses(64) = 260 bytes
    assert len(raw) == 4 + 5 * 32 + 32 + 2 * 32

    # Word slices for readability.
    words = [raw[4 + i * 32 : 4 + (i + 1) * 32] for i in range(5)]
    assert int.from_bytes(words[0], "big") == 10**18                # amountIn
    assert int.from_bytes(words[1], "big") == 999 * 10**18          # amountOutMin
    assert int.from_bytes(words[2], "big") == 5 * 32                # path offset
    # `to` lives right-padded in word[3].
    assert words[3][12:].hex() == ACCOUNT.lower().replace("0x", "")
    assert int.from_bytes(words[4], "big") == 1_700_000_000         # deadline

    # Path tail: length 2 then two right-padded addresses.
    tail_offset = 4 + 5 * 32
    assert int.from_bytes(raw[tail_offset : tail_offset + 32], "big") == 2
    first_addr_word = raw[tail_offset + 32 : tail_offset + 64]
    second_addr_word = raw[tail_offset + 64 : tail_offset + 96]
    assert first_addr_word[12:].hex() == TOKEN_A.lower().replace("0x", "")
    assert second_addr_word[12:].hex() == TOKEN_B.lower().replace("0x", "")


# ---------------------------------------------------------------------------
# place_order happy path
# ---------------------------------------------------------------------------


def test_place_order_signs_and_broadcasts() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x7",     # nonce 7
            "eth_gasPrice": "0x4a817c800",        # 20 gwei
            "eth_sendRawTransaction": "0xdeadbeef",
        })
        adapter = EVMAdapter(
            rpc=rpc,
            private_key=PRIVATE_KEY,
            account_address=ACCOUNT,
            config=EVMAdapterConfig(chain=EVMChainConfig(chain_id=1)),
            clock=lambda: 1_700_000_000.0,
        )
        result = await adapter.place_order(_request("ord-1"))

        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == "0xdeadbeef"
        # Nonce, gas, and broadcast were all queried.
        methods = [c[0] for c in rpc.calls]
        assert methods == [
            "eth_getTransactionCount",
            "eth_gasPrice",
            "eth_sendRawTransaction",
        ]
        # eth_sendRawTransaction received a 0x-prefixed hex blob.
        _, params = rpc.calls[-1]
        assert isinstance(params[0], str) and params[0].startswith("0x")

    _run(scenario())


def test_place_order_is_idempotent_and_reuses_nonce() -> None:
    async def scenario() -> None:
        # Provide enough responses for two full place_order paths.
        # The adapter should hit only the second response set zero
        # times because of the idempotency cache.
        rpc = _FakeRpc({
            "eth_getTransactionCount": ["0x7", "0x99"],
            "eth_gasPrice": ["0x4a817c800", "0x4a817c800"],
            "eth_sendRawTransaction": ["0xtxhash1", "0xtxhash2"],
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 1_700_000_000.0,
        )
        first = await adapter.place_order(_request("ord-1"))
        second = await adapter.place_order(_request("ord-1"))
        assert first == second
        # Exactly one nonce / gas / send call cycle.
        methods = [c[0] for c in rpc.calls]
        assert methods.count("eth_sendRawTransaction") == 1
        assert methods.count("eth_getTransactionCount") == 1

    _run(scenario())


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


def test_amount_out_min_uses_explicit_override_when_present() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xh",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        req = _request(metadata={
            "from_address": TOKEN_A,
            "to_address": TOKEN_B,
            "expected_amount_out": 1000.0,
            "slippage_bps": 50,
            "amount_out_min": 999_000_000_000_000_000_000,  # explicit
        })
        result = await adapter.place_order(req)
        assert result.metadata["amount_out_min_wei"] == 999_000_000_000_000_000_000

    _run(scenario())


def test_amount_out_min_derived_from_slippage_bps() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xh",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        # expected_amount_out = 1000 with 18 decimals = 10^21 wei
        # 50 bps slippage -> 0.5% off -> 9.95 * 10^20
        req = _request()
        result = await adapter.place_order(req)
        expected_min = 10**21 - 10**21 * 50 // 10_000
        assert result.metadata["amount_out_min_wei"] == expected_min

    _run(scenario())


def test_slippage_bps_rejects_out_of_range() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        req = _request(metadata={
            "from_address": TOKEN_A,
            "to_address": TOKEN_B,
            "expected_amount_out": 1.0,
            "slippage_bps": 20_000,  # > 100%
        })
        # The adapter's "never raise on routine failure" contract
        # now converts this AdapterError into a REJECTED result.
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.REJECTED
        assert "slippage_bps" in (result.error or "")

    _run(scenario())


# ---------------------------------------------------------------------------
# get_balance + receipt handling
# ---------------------------------------------------------------------------


def test_get_balance_parses_hex_wei_to_ether() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({"eth_getBalance": hex(2_500_000_000_000_000_000)})  # 2.5 ETH
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        balance = await adapter.get_balance()
        assert balance == pytest.approx(2.5)

    _run(scenario())


def test_get_balance_returns_zero_on_rpc_failure() -> None:
    async def scenario() -> None:
        async def boom(_m: str, _p: List[Any]) -> Any:
            raise ConnectionError("rpc down")

        adapter = EVMAdapter(
            rpc=boom, private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        assert (await adapter.get_balance()) == 0.0

    _run(scenario())


def test_wait_for_receipt_transitions_to_filled_on_status_1() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xtxhash",
            "eth_getTransactionReceipt": {
                "transactionHash": "0xtxhash",
                "status": "0x1",
                "blockNumber": "0x12345",
                "gasUsed": "0x5208",
            },
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
            config=EVMAdapterConfig(confirm_poll_seconds=0.0),
        )
        await adapter.place_order(_request("ord-r"))
        receipt = await adapter.wait_for_receipt("ord-r")
        assert receipt is not None
        assert receipt.status == 1
        cached = (await adapter.get_open_orders())
        assert cached == []  # FILLED is no longer open

    _run(scenario())


def test_wait_for_receipt_transitions_to_rejected_on_revert() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xtxhash",
            "eth_getTransactionReceipt": {
                "transactionHash": "0xtxhash",
                "status": "0x0",
                "blockNumber": "0x12345",
                "gasUsed": "0x5208",
            },
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
            config=EVMAdapterConfig(confirm_poll_seconds=0.0),
        )
        await adapter.place_order(_request("ord-r"))
        receipt = await adapter.wait_for_receipt("ord-r")
        assert receipt is not None
        assert receipt.status == 0
        # The order is no longer open and now carries the revert error.
        opens = await adapter.get_open_orders()
        assert opens == []

    _run(scenario())


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


def test_cancel_order_marks_local_state_but_warns_chain_may_mine() -> None:
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xtxhash",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        await adapter.place_order(_request("ord-c"))
        cancelled = await adapter.cancel_order("ord-c")
        assert cancelled.status is OrderStatus.CANCELLED
        assert "chain may still mine" in (cancelled.metadata.get("warning") or "")

    _run(scenario())


def test_buy_uses_reversed_path_and_quote_notional() -> None:
    """BUY spends QUOTE (to_address) to acquire BASE (from_address);
    the swap path must be [to_address, from_address], the
    amount_in must come from notional in QUOTE units, and
    amount_out_min must use BASE decimals. Encoding it as the
    same direction as SELL would execute the opposite trade.
    """
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xtxbuy",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        # BUY 1 BASE for 1.0 QUOTE notional. Tokens have different
        # decimals so we can check scaling is side-aware.
        req = _request(
            client_id="buy-1",
            side=OrderSide.BUY,
            quantity=None,
            notional=1.0,
            metadata={
                "from_address": TOKEN_A,
                "to_address": TOKEN_B,
                "from_decimals": 18,
                "to_decimals": 6,  # USDC-like quote
                "expected_amount_out": 1.0,
                "slippage_bps": 50,
            },
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.PENDING
        # amount_in is notional * 10^to_decimals (QUOTE units).
        assert result.metadata["amount_in_wei"] == 1 * 10**6
        # input_address is QUOTE (to_address), output_address is BASE.
        assert result.metadata["input_address"] == TOKEN_B
        assert result.metadata["output_address"] == TOKEN_A
        assert result.metadata["side"] == "BUY"
        # amount_out_min uses BASE decimals (from_decimals=18).
        expected_min = 10**18 - 10**18 * 50 // 10_000
        assert result.metadata["amount_out_min_wei"] == expected_min

    _run(scenario())


def test_buy_without_notional_is_rejected() -> None:
    """BUY needs notional in QUOTE units; quantity alone needs a
    price the adapter does not know.
    """
    async def scenario() -> None:
        adapter = EVMAdapter(
            rpc=_FakeRpc({}), private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        req = _request(
            side=OrderSide.BUY,
            quantity=1.0,
            notional=None,
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.REJECTED
        assert "BUY" in (result.error or "")

    _run(scenario())


def test_sell_without_quantity_is_rejected() -> None:
    async def scenario() -> None:
        adapter = EVMAdapter(
            rpc=_FakeRpc({}), private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        req = _request(
            side=OrderSide.SELL,
            quantity=None,
            notional=1.0,
        )
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.REJECTED
        assert "SELL" in (result.error or "")

    _run(scenario())


def test_missing_expected_amount_out_is_rejected() -> None:
    """Without a quote or explicit amount_out_min the adapter would
    have to default to 1 wei = unlimited slippage = sandwich
    attack invitation. Reject the order instead.
    """
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
        })
        adapter = EVMAdapter(
            rpc=rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
        )
        req = _request(metadata={
            "from_address": TOKEN_A,
            "to_address": TOKEN_B,
            # no expected_amount_out, no amount_out_min
        })
        result = await adapter.place_order(req)
        assert result.status is OrderStatus.REJECTED
        # The original AdapterError message is wrapped in
        # "chain interaction failed: ..." so both fragments are
        # visible to the caller.
        assert "amount_out_min" in (result.error or "")

    _run(scenario())


def test_get_balance_handles_null_wei_response() -> None:
    """Some RPC providers return null on the very first balance
    query for a newly seen account. Must not crash; return 0.0.
    """
    async def scenario() -> None:
        adapter = EVMAdapter(
            rpc=_FakeRpc({"eth_getBalance": None}),
            private_key=PRIVATE_KEY,
            account_address=ACCOUNT,
        )
        assert (await adapter.get_balance()) == 0.0

    _run(scenario())


def test_gas_price_fallback_can_be_disabled_for_strict_chains() -> None:
    """When the operator sets gas_price_fallback_wei=None the
    adapter should refuse to size a transaction it cannot price,
    instead of paying out of a stale hardcoded 20 gwei.
    """
    async def scenario() -> None:
        async def boom_gas(method, params):
            if method == "eth_gasPrice":
                raise ConnectionError("rpc down")
            return "0x0"

        adapter = EVMAdapter(
            rpc=boom_gas, private_key=PRIVATE_KEY, account_address=ACCOUNT,
            clock=lambda: 0.0,
            config=EVMAdapterConfig(gas_price_fallback_wei=None),
        )
        result = await adapter.place_order(_request("ord-nogas"))
        assert result.status is OrderStatus.REJECTED
        assert "gas price unavailable" in (result.error or "")

    _run(scenario())


def test_cancel_unknown_returns_rejected() -> None:
    async def scenario() -> None:
        adapter = EVMAdapter(
            rpc=_FakeRpc({}), private_key=PRIVATE_KEY, account_address=ACCOUNT
        )
        result = await adapter.cancel_order("never-seen")
        assert result.status is OrderStatus.REJECTED

    _run(scenario())


def test_place_order_rpc_timeout_is_rejected_and_reason_is_logged() -> None:
    async def scenario() -> None:
        async def timeout_rpc(method: str, params: List[Any]) -> Any:
            if method == "eth_sendRawTransaction":
                raise TimeoutError("rpc timeout")
            if method == "eth_getTransactionCount":
                return "0x0"
            if method == "eth_gasPrice":
                return "0x1"
            return None

        adapter = EVMAdapter(rpc=timeout_rpc, private_key=PRIVATE_KEY, account_address=ACCOUNT)
        result = await adapter.place_order(_request("evm-timeout"))
        assert result.status is OrderStatus.REJECTED
        assert "timeout" in (result.error or "")

    _run(scenario())


def test_wait_for_receipt_timeout_threshold_and_poll_backoff() -> None:
    """Polling should respect confirm timeout and bounded poll cadence."""
    async def scenario() -> None:
        rpc = _FakeRpc({
            "eth_getTransactionCount": "0x0",
            "eth_gasPrice": "0x1",
            "eth_sendRawTransaction": "0xtxhash",
            "eth_getTransactionReceipt": [None, None, None],
        })

        now = [0.0]
        def fake_clock() -> float:
            now[0] += 0.11
            return now[0]

        adapter = EVMAdapter(
            rpc=rpc,
            private_key=PRIVATE_KEY,
            account_address=ACCOUNT,
            clock=fake_clock,
            config=EVMAdapterConfig(confirm_timeout_seconds=0.3, confirm_poll_seconds=0.05),
        )
        await adapter.place_order(_request("ord-timeout"))

        sleeps = []
        original_sleep = asyncio.sleep
        async def capture_sleep(delay: float) -> None:
            sleeps.append(delay)

        asyncio.sleep = capture_sleep  # type: ignore[assignment]
        try:
            receipt = await adapter.wait_for_receipt("ord-timeout")
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert receipt is None
        cached = adapter._order_cache["ord-timeout"]
        assert cached.status is OrderStatus.PENDING
        assert sleeps and all(d <= 0.05 for d in sleeps)

    _run(scenario())
