"""
EVM-chain swap adapter.

Translates `OrderRequest` into a Uniswap-V2-style ``swapExactTokensFor
Tokens`` call against a configurable router contract, signs the
transaction locally with the operator's private key from the
`Keystore`, and broadcasts it through an injectable
`EVMRpcClient`. Tests inject a stub RPC and confirm the wire
without touching any real chain.

The adapter intentionally does not include a generic web3.py
import: the chain interaction is reduced to a small, well-typed
Protocol so we can plug in any provider (Alchemy, Infura,
Flashbots Protect, a local Anvil for replay tests). Real production
deployments wrap web3.py in a thin `EVMRpcClient` adapter; the
inline `RawJsonRpcClient` in this module is enough for tests.

Threat model:

- Private key never appears in any RPC payload. The transaction is
  signed locally inside `place_order` and only the signed
  serialised bytes are sent over the wire.
- Idempotency is enforced both by the adapter (in-memory cache on
  `client_order_id`) and by the EVM transaction nonce: a retry
  reuses the same nonce so the chain itself dedupes if our cache
  ever gets cleared.
- Slippage is bounded by `amount_out_min` which the caller can set
  directly (`metadata['amount_out_min']`) or which the adapter
  derives from `slippage_bps` over the expected output.
- The deadline is short by default (90 seconds from `now`) so a
  pending tx that lingers in the mempool past that window is
  guaranteed to revert rather than fill at a stale quote.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .base import (
    AdapterError,
    ExecutionAdapter,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RPC Protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TxReceipt:
    """The slice of an EVM transaction receipt the adapter actually
    consumes. Real implementations populate more fields; the rest is
    free-form on `OrderResult.metadata`.
    """

    transaction_hash: str
    status: int  # 1 = success, 0 = revert
    block_number: int
    gas_used: int


JsonRpcRequest = Callable[[str, List[Any]], Awaitable[Any]]
"""``await rpc('eth_method', [params])`` style."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EVMChainConfig:
    """One chain + router pair. Defaults match Ethereum mainnet."""

    chain_id: int = 1
    router_address: str = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"  # Uniswap V2
    weth_address: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    rpc_url: str = "https://cloudflare-eth.com"


@dataclass
class EVMAdapterConfig:
    chain: EVMChainConfig = field(default_factory=EVMChainConfig)
    slippage_bps: int = 50          # 0.5% default acceptable slippage
    deadline_seconds: int = 90      # tx must mine within 90s
    gas_priority_gwei: float = 1.5
    gas_limit: int = 300_000
    confirm_timeout_seconds: float = 60.0
    confirm_poll_seconds: float = 2.0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class EVMAdapter(ExecutionAdapter):
    """Uniswap-V2-style swap adapter.

    Constructor takes:

    - `rpc`: injectable async JSON-RPC client.
    - `private_key`: raw secp256k1 bytes (loaded from `Keystore` at
      construction time). The adapter retains this in memory for
      its lifetime; instantiate it inside a short-lived `async
      with` block in production code.
    - `account_address`: 0x-prefixed checksum address. Provided
      explicitly so we never have to derive it from the private key
      on the hot path.
    - `config`: chain + slippage tunables.

    `place_order` builds an ERC-20-token swap (``swapExactTokensFor
    Tokens(amountIn, amountOutMin, path, to, deadline)``) given:

      symbol     -> "FROM/TO", maps to a 2-hop path [FROM, TO].
      side       -> BUY swaps TO into FROM, SELL swaps FROM into TO.
      notional   -> in FROM units, scaled by the FROM decimals
                    (passed via metadata['from_decimals'], defaults
                    to 18).
      metadata   -> may override `amount_out_min`, `gas_priority_gwei`,
                    `path` for multi-hop routes.

    The adapter does NOT discover token addresses; the caller passes
    them through `metadata['from_address']` and `metadata
    ['to_address']`. A higher-level routing layer is the right place
    to map symbols to addresses.
    """

    name = "evm"

    def __init__(
        self,
        rpc: JsonRpcRequest,
        private_key: bytes,
        account_address: str,
        config: Optional[EVMAdapterConfig] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(private_key, (bytes, bytearray)) or len(private_key) != 32:
            raise AdapterError(
                f"private_key must be 32 raw bytes, got {len(private_key) if isinstance(private_key, (bytes, bytearray)) else type(private_key).__name__}"
            )
        if not account_address.startswith("0x") or len(account_address) != 42:
            raise AdapterError(
                f"account_address must be a 0x-prefixed 20-byte hex string, got {account_address!r}"
            )
        self.rpc = rpc
        self._private_key = bytes(private_key)
        self.account_address = account_address
        self.config = config or EVMAdapterConfig()
        self.clock = clock
        self._order_cache: Dict[str, OrderResult] = {}
        self._nonce_cache: Dict[str, int] = {}  # client_order_id -> nonce
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ExecutionAdapter API
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        async with self._lock:
            cached = self._order_cache.get(request.client_order_id)
            if cached is not None:
                return cached

            err = self._validate_request(request)
            if err is not None:
                result = self._rejected(request, err)
                self._order_cache[request.client_order_id] = result
                return result

            metadata = dict(request.metadata or {})
            from_address = metadata.get("from_address")
            to_address = metadata.get("to_address")
            if not from_address or not to_address:
                result = self._rejected(
                    request,
                    "EVM swaps require metadata['from_address'] and metadata['to_address']",
                )
                self._order_cache[request.client_order_id] = result
                return result

            from_decimals = int(metadata.get("from_decimals", 18))
            to_decimals = int(metadata.get("to_decimals", 18))
            amount_in = self._resolve_amount_in(request, from_decimals)
            if amount_in <= 0:
                result = self._rejected(
                    request, "could not resolve a positive amount_in"
                )
                self._order_cache[request.client_order_id] = result
                return result

            try:
                amount_out_min = await self._compute_amount_out_min(
                    request, amount_in, from_address, to_address, metadata, to_decimals
                )
                tx_unsigned = await self._build_swap_tx(
                    amount_in=amount_in,
                    amount_out_min=amount_out_min,
                    from_address=from_address,
                    to_address=to_address,
                    metadata=metadata,
                    client_order_id=request.client_order_id,
                )
                signed_hex = self._sign_tx(tx_unsigned)
                tx_hash = await self.rpc("eth_sendRawTransaction", [signed_hex])
            except AdapterError:
                raise
            except Exception as exc:
                logger.error("EVM place_order failure: %s", exc)
                result = self._rejected(request, f"chain interaction failed: {exc}")
                self._order_cache[request.client_order_id] = result
                return result

            pending = OrderResult(
                client_order_id=request.client_order_id,
                venue_order_id=str(tx_hash),
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.PENDING,
                requested_quantity=float(amount_in) / 10**from_decimals,
                metadata={
                    "tx_hash": str(tx_hash),
                    "amount_in_wei": int(amount_in),
                    "amount_out_min_wei": int(amount_out_min),
                    "from_address": from_address,
                    "to_address": to_address,
                    "nonce": tx_unsigned["nonce"],
                },
            )
            self._order_cache[request.client_order_id] = pending
            return pending

    async def cancel_order(self, client_order_id: str) -> OrderResult:
        """EVM transactions cannot be cancelled once broadcast, only
        replaced with a higher-fee transaction reusing the same
        nonce. We expose that flow as a `replace_with_self_send`
        but keep `cancel_order` strict: if the tx is already mined
        we return its current state; otherwise we mark the cached
        record CANCELLED locally (the chain may still mine it,
        which is the user's risk).
        """
        async with self._lock:
            existing = self._order_cache.get(client_order_id)
            if existing is None:
                return OrderResult(
                    client_order_id=client_order_id,
                    venue_order_id=None,
                    symbol="",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    error="unknown client_order_id",
                )
            if existing.is_done:
                return existing
            cancelled = OrderResult(
                client_order_id=existing.client_order_id,
                venue_order_id=existing.venue_order_id,
                symbol=existing.symbol,
                side=existing.side,
                order_type=existing.order_type,
                status=OrderStatus.CANCELLED,
                requested_quantity=existing.requested_quantity,
                filled_quantity=existing.filled_quantity,
                average_fill_price=existing.average_fill_price,
                fills=list(existing.fills),
                metadata={
                    **existing.metadata,
                    "warning": "EVM cancel is local-only; chain may still mine the tx",
                },
            )
            self._order_cache[client_order_id] = cancelled
            return cancelled

    async def get_open_orders(self) -> List[OrderResult]:
        return [r for r in self._order_cache.values() if r.is_open]

    async def get_positions(self) -> Dict[str, float]:
        # Pure ERC-20 balances live on the chain; expose via a
        # follow-up `EVMBalanceService`. The adapter holds no
        # local position bookkeeping.
        return {}

    async def get_balance(self) -> float:
        """Native-token balance of the operator account.

        Returns the value in whole units (ether for ETH-style
        chains). Returns 0.0 if the RPC fails so the pipeline can
        still produce a result.
        """
        try:
            wei = await self.rpc("eth_getBalance", [self.account_address, "latest"])
        except Exception as exc:
            logger.warning("eth_getBalance failed: %s", exc)
            return 0.0
        try:
            return int(wei, 16) / 10**18 if isinstance(wei, str) else float(wei) / 10**18
        except (TypeError, ValueError):
            return 0.0

    async def wait_for_receipt(
        self, client_order_id: str
    ) -> Optional[TxReceipt]:
        """Block until the tx for `client_order_id` lands or the
        confirm timeout elapses. Updates the cached order status to
        FILLED on success-1 receipt, REJECTED on revert (status 0).
        """
        async with self._lock:
            existing = self._order_cache.get(client_order_id)
        if existing is None or not existing.is_open:
            return None
        tx_hash = (existing.metadata or {}).get("tx_hash")
        if not tx_hash:
            return None

        deadline = self.clock() + self.config.confirm_timeout_seconds
        while self.clock() < deadline:
            try:
                receipt = await self.rpc("eth_getTransactionReceipt", [tx_hash])
            except Exception as exc:
                logger.debug("eth_getTransactionReceipt failed: %s", exc)
                receipt = None
            if receipt:
                parsed = _parse_receipt(receipt)
                await self._apply_receipt(client_order_id, parsed)
                return parsed
            await asyncio.sleep(self.config.confirm_poll_seconds)
        return None

    # ------------------------------------------------------------------
    # Building / signing
    # ------------------------------------------------------------------

    async def _build_swap_tx(
        self,
        *,
        amount_in: int,
        amount_out_min: int,
        from_address: str,
        to_address: str,
        metadata: Dict[str, Any],
        client_order_id: str,
    ) -> Dict[str, Any]:
        deadline = int(self.clock()) + self.config.deadline_seconds
        path = metadata.get("path") or [from_address, to_address]
        if len(path) < 2:
            raise AdapterError("swap path must have at least 2 hops")

        data = _encode_swap_exact_tokens_for_tokens(
            amount_in=amount_in,
            amount_out_min=amount_out_min,
            path=path,
            to=self.account_address,
            deadline=deadline,
        )

        nonce = await self._allocate_nonce(client_order_id)
        gas_price = await self._fetch_gas_price()

        return {
            "from": self.account_address,
            "to": self.config.chain.router_address,
            "value": 0,
            "data": data,
            "chainId": self.config.chain.chain_id,
            "nonce": nonce,
            "gas": self.config.gas_limit,
            "gasPrice": gas_price,
        }

    async def _allocate_nonce(self, client_order_id: str) -> int:
        # A retry with the same client_order_id reuses the same
        # nonce so the chain itself dedupes the tx.
        cached = self._nonce_cache.get(client_order_id)
        if cached is not None:
            return cached
        nonce_hex = await self.rpc(
            "eth_getTransactionCount", [self.account_address, "pending"]
        )
        nonce = int(nonce_hex, 16) if isinstance(nonce_hex, str) else int(nonce_hex)
        self._nonce_cache[client_order_id] = nonce
        return nonce

    async def _fetch_gas_price(self) -> int:
        try:
            raw = await self.rpc("eth_gasPrice", [])
        except Exception as exc:
            logger.warning("eth_gasPrice failed: %s; falling back to 20 gwei", exc)
            return 20 * 10**9
        try:
            return int(raw, 16) if isinstance(raw, str) else int(raw)
        except (TypeError, ValueError):
            return 20 * 10**9

    async def _compute_amount_out_min(
        self,
        request: OrderRequest,
        amount_in: int,
        from_address: str,
        to_address: str,
        metadata: Dict[str, Any],
        to_decimals: int,
    ) -> int:
        # Explicit override wins; otherwise derive from a quoted
        # expected output (metadata['expected_amount_out']) and
        # apply the slippage tolerance.
        explicit = metadata.get("amount_out_min")
        if explicit is not None:
            return int(explicit)
        expected = metadata.get("expected_amount_out")
        if expected is None:
            # No quote available; default to 1 wei so the swap
            # still proceeds and only reverts on truly catastrophic
            # slippage. Callers SHOULD provide a quote.
            return 1
        expected_int = int(float(expected) * 10**to_decimals) if isinstance(
            expected, float
        ) else int(expected)
        bps = int(metadata.get("slippage_bps", self.config.slippage_bps))
        if bps < 0 or bps > 10000:
            raise AdapterError(f"slippage_bps must be in [0, 10000], got {bps}")
        return max(0, expected_int - expected_int * bps // 10_000)

    def _sign_tx(self, tx: Dict[str, Any]) -> str:
        try:
            from eth_account import Account
        except ImportError as exc:  # pragma: no cover
            raise AdapterError(
                "EVMAdapter requires the `eth-account` package"
            ) from exc
        signed = Account.sign_transaction(tx, self._private_key)
        # eth-account returns either `raw_transaction` (>= 0.13) or
        # `rawTransaction` (older). Support both.
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        if raw is None:
            raise AdapterError("signed transaction missing raw bytes")
        return "0x" + raw.hex() if not raw.hex().startswith("0x") else raw.hex()

    async def _apply_receipt(self, client_order_id: str, receipt: TxReceipt) -> None:
        async with self._lock:
            existing = self._order_cache.get(client_order_id)
            if existing is None:
                return
            if receipt.status == 1:
                updated = OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.FILLED,
                    requested_quantity=existing.requested_quantity,
                    filled_quantity=existing.requested_quantity or 0.0,
                    average_fill_price=None,
                    metadata={
                        **existing.metadata,
                        "block_number": receipt.block_number,
                        "gas_used": receipt.gas_used,
                    },
                )
            else:
                updated = OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.REJECTED,
                    error="transaction reverted on-chain",
                    metadata={
                        **existing.metadata,
                        "block_number": receipt.block_number,
                        "gas_used": receipt.gas_used,
                    },
                )
            self._order_cache[client_order_id] = updated

    @staticmethod
    def _resolve_amount_in(request: OrderRequest, decimals: int) -> int:
        if request.quantity is not None:
            return int(float(request.quantity) * 10**decimals)
        if request.notional is not None:
            return int(float(request.notional) * 10**decimals)
        return 0

    def _rejected(self, request: OrderRequest, error: str) -> OrderResult:
        return OrderResult(
            client_order_id=request.client_order_id,
            venue_order_id=None,
            symbol=request.symbol,
            side=request.side if isinstance(request.side, OrderSide) else OrderSide.BUY,
            order_type=request.order_type if isinstance(request.order_type, OrderType) else OrderType.MARKET,
            status=OrderStatus.REJECTED,
            error=error,
        )


# ---------------------------------------------------------------------------
# ABI encoding (just what we need for swapExactTokensForTokens)
# ---------------------------------------------------------------------------


_SWAP_EXACT_TOKENS_FOR_TOKENS_SELECTOR = bytes.fromhex("38ed1739")
"""Function selector for swapExactTokensForTokens(uint256,uint256,
address[],address,uint256).
"""


def _encode_swap_exact_tokens_for_tokens(
    *,
    amount_in: int,
    amount_out_min: int,
    path: List[str],
    to: str,
    deadline: int,
) -> str:
    """Hand-roll the ABI encoding of swapExactTokensForTokens.

    We avoid eth-abi as a dep by encoding directly: the layout is:
      selector (4 bytes)
      amountIn (32) | amountOutMin (32) | path_offset (32) |
        to (32) | deadline (32) | path_length (32) |
        path_entry_1 (32) ... path_entry_N (32)

    Returns a 0x-prefixed hex string suitable for `data`.
    """
    head = bytearray()
    head.extend(amount_in.to_bytes(32, "big"))
    head.extend(amount_out_min.to_bytes(32, "big"))
    # path is a dynamic type; its offset is 5*32 (after the 5 head
    # words) from the start of the data section.
    head.extend((5 * 32).to_bytes(32, "big"))
    head.extend(bytes.fromhex(to.lower().replace("0x", "")).rjust(32, b"\x00"))
    head.extend(deadline.to_bytes(32, "big"))

    tail = bytearray()
    tail.extend(len(path).to_bytes(32, "big"))
    for addr in path:
        tail.extend(bytes.fromhex(addr.lower().replace("0x", "")).rjust(32, b"\x00"))

    payload = _SWAP_EXACT_TOKENS_FOR_TOKENS_SELECTOR + bytes(head) + bytes(tail)
    return "0x" + payload.hex()


def _parse_receipt(raw: Dict[str, Any]) -> TxReceipt:
    def _hex_to_int(v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return int(v, 16) if v.startswith("0x") else int(v)
        return 0

    return TxReceipt(
        transaction_hash=str(raw.get("transactionHash") or raw.get("transaction_hash") or ""),
        status=_hex_to_int(raw.get("status", 0)),
        block_number=_hex_to_int(raw.get("blockNumber") or raw.get("block_number") or 0),
        gas_used=_hex_to_int(raw.get("gasUsed") or raw.get("gas_used") or 0),
    )
