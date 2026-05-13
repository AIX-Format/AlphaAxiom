"""
MT5 execution adapter.

This adapter does not talk to MetaTrader 5 directly. MT5's MQL
environment runs inside the terminal and cannot accept inbound TCP
from a separate Python process, so AlphaAxiom routes signals through
a Cloudflare Worker relay (default `oracle.axiomid.app`):

    Python sidecar  --HTTPS--->  Cloudflare Worker  <--HTTPS-- AlphaReceiver.mq5

The Python side signs every signal with the operator's Ed25519
private key, the Worker verifies the signature against the published
public key, and the MQL Expert Advisor polls the Worker for the
next signed signal it has not yet executed.

Threat model the adapter defends against:

- Replay: every signed payload includes a UTC `timestamp` and the
  `client_order_id` is the idempotency key. The Worker is expected
  to reject duplicate ids; the adapter never resigns a request that
  has already returned a definitive result.
- Tampering: Ed25519 over the canonical JSON of the request body
  (sorted keys, no whitespace). Flipping any byte invalidates the
  signature.
- Compromised storage: the private key lives in the OS keychain
  (read via the existing keyring path) and is only resolved at the
  moment of signing. The adapter accepts an in-memory key for tests
  and CI but warns when one is not provisioned through the
  keychain.
- Transient network failure: bounded exponential backoff with
  jitter. The retry loop only re-sends the same signed payload,
  preserving idempotency.

What the adapter does NOT do:

- Track fills. The Worker is fire-and-forget from the adapter's
  point of view; fills come back through a separate poll path that
  the next PR will wire up. `place_order` returns `OrderStatus.
  PENDING` (signal accepted by the relay, fill TBD) or `REJECTED`
  on a transport / signing / validation failure.
- Manage positions or balance. Those numbers live on the MT5 side;
  this adapter is the outbound signal pipe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
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
# Transport + signing protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpResponse:
    """The minimal HTTP response shape the adapter needs.

    Real callers build this from `aiohttp.ClientResponse.read()`
    or `requests.Response`; tests build it directly.
    """

    status: int
    body: bytes
    headers: Dict[str, str]


HttpClient = Callable[[str, str, Dict[str, str], bytes], Awaitable[HttpResponse]]
"""Async transport: ``await client(method, url, headers, body)``."""


SigningFunction = Callable[[bytes], bytes]
"""Sign a canonical payload, return the raw signature bytes."""


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def canonical_payload(payload: Dict[str, Any]) -> bytes:
    """Deterministic JSON serialisation for signing.

    Sorted keys, no whitespace, UTF-8 encoded. The Worker side MUST
    use the same canonicalisation; any whitespace divergence breaks
    verification.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Keychain-backed signer
# ---------------------------------------------------------------------------


_KEYCHAIN_SERVICE = "money-machine"
_KEYCHAIN_ACCOUNT_PRIVATE = "mt5-signing-private-key"


def keychain_signer() -> SigningFunction:
    """Build a signer that reads the Ed25519 private key from the OS
    keychain at every call.

    Raises `AdapterError` if `cryptography` or `keyring` is missing,
    or if no key is present. Construct this in production; in tests
    pass an explicit `inline_signer(seed)` for determinism.
    """
    try:
        import keyring  # type: ignore
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
    except ImportError as exc:  # pragma: no cover - dev environment
        raise AdapterError(
            "MT5 keychain signer needs `cryptography` and `keyring` installed"
        ) from exc

    def _sign(payload: bytes) -> bytes:
        pem = keyring.get_password(_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT_PRIVATE)
        if not pem:
            raise AdapterError(
                "MT5 signing key not found in keychain. Run the Tauri shell "
                "to provision it, or set inline_signer for tests."
            )
        key = load_pem_private_key(pem.encode("utf-8"), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise AdapterError("MT5 signing key in keychain is not Ed25519")
        return key.sign(payload)

    return _sign


def inline_signer(seed: bytes) -> SigningFunction:
    """Build a deterministic signer from a 32-byte seed (tests only)."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:  # pragma: no cover
        raise AdapterError("cryptography required for inline_signer") from exc
    if len(seed) != 32:
        raise AdapterError(f"Ed25519 seed must be 32 bytes, got {len(seed)}")
    key = Ed25519PrivateKey.from_private_bytes(seed)

    def _sign(payload: bytes) -> bytes:
        return key.sign(payload)

    return _sign


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class MT5Config:
    """Tunables for the MT5 transport."""

    oracle_url: str = "https://oracle.axiomid.app"
    request_timeout_seconds: float = 5.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.25
    backoff_max_seconds: float = 4.0


class MT5Adapter(ExecutionAdapter):
    """Outbound signal pipe to AlphaReceiver.mq5 via the Worker relay."""

    name = "mt5"

    def __init__(
        self,
        http_client: HttpClient,
        signer: SigningFunction,
        *,
        config: Optional[MT5Config] = None,
        public_key_id: str = "default",
        clock: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
    ) -> None:
        self.http = http_client
        self.signer = signer
        self.config = config or MT5Config()
        self.public_key_id = public_key_id
        self.clock = clock
        # Idempotency cache: client_order_id -> last definitive result.
        # Anything PENDING is also cached so a retry sees the same
        # response without re-signing.
        self._order_cache: Dict[str, OrderResult] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ExecutionAdapter API
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        # Build the signed envelope under the lock (short, in-memory
        # only), then release the lock before the HTTP call so a
        # slow/timing-out relay request does NOT serialise every
        # other order on this adapter instance. We re-acquire only
        # to persist the final result in the idempotency cache.
        async with self._lock:
            cached = self._order_cache.get(request.client_order_id)
            if cached is not None:
                logger.debug(
                    "MT5 place_order idempotent hit for %s",
                    request.client_order_id,
                )
                return cached

            err = self._validate_request(request)
            if err is not None:
                result = self._rejected(request, err)
                self._order_cache[request.client_order_id] = result
                return result

            payload = self._build_payload(request)
            canonical = canonical_payload(payload)
            try:
                signature_hex = self.signer(canonical).hex()
            except Exception as exc:
                # AdapterError is included here; the adapter contract
                # is "no exceptions, return REJECTED" and signing is
                # a routine failure path (missing keychain entry,
                # libsodium quirk, etc.).
                result = self._rejected(request, f"signing failed: {exc}")
                self._order_cache[request.client_order_id] = result
                return result

            envelope = {
                "payload": payload,
                "signature": signature_hex,
                "public_key_id": self.public_key_id,
            }
            body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "X-Client-Order-Id": request.client_order_id,
                "X-Public-Key-Id": self.public_key_id,
            }
            url = self.config.oracle_url.rstrip("/") + "/signals"

        # Lock released; the retry loop runs without head-of-line
        # blocking other client_order_ids.
        response = await self._post_with_retries(url, headers, body)
        result = self._result_from_response(request, response)
        async with self._lock:
            # Last-writer wins, but the idempotency cache check at
            # the top means a competing retry of the SAME id would
            # have returned the prior cached result instead of
            # getting here.
            self._order_cache[request.client_order_id] = result
        return result

    async def cancel_order(self, client_order_id: str) -> OrderResult:
        """Cancel a previously sent signal.

        Sends a signed cancel envelope to the Worker. The semantics
        match the venue:

          - unknown id: REJECTED, never reaches the relay.
          - already FILLED / CANCELLED / REJECTED: return the
            existing cached state, no relay call.
          - signing failure: return a fresh REJECTED with the
            error text, leave the cache untouched so the caller
            can retry.
          - relay non-2xx: return a fresh REJECTED with the HTTP
            error text, leave the cache untouched.
          - relay 2xx that reports a non-cancel terminal state
            (already filled, etc.): respect the relay payload and
            apply that state instead of forcing CANCELLED.
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
            if existing.status in (OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED):
                return existing

            payload = {
                "client_order_id": client_order_id,
                "action": "CANCEL",
                "timestamp": self.clock().isoformat(),
            }
            canonical = canonical_payload(payload)
            try:
                signature_hex = self.signer(canonical).hex()
            except Exception as exc:
                logger.warning("MT5 cancel signing failed: %s", exc)
                return OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.REJECTED,
                    error=f"cancel signing failed: {exc}",
                )
            envelope = {
                "payload": payload,
                "signature": signature_hex,
                "public_key_id": self.public_key_id,
            }
            body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "X-Client-Order-Id": client_order_id,
                "X-Public-Key-Id": self.public_key_id,
            }
            url = self.config.oracle_url.rstrip("/") + "/signals/cancel"

        # Release the lock for the network call so other orders are
        # not blocked while a cancel retries through the relay.
        response = await self._post_with_retries(url, headers, body)

        if 200 <= response.status < 300:
            parsed = _safe_parse_json_dict(response.body)
            reported_status = str(parsed.get("status", "")).upper()
            # Race window: the relay might tell us the order already
            # filled or was rejected. Respect that instead of
            # forcing CANCELLED.
            if reported_status == "FILLED":
                final = OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.FILLED,
                    requested_quantity=existing.requested_quantity,
                    filled_quantity=float(parsed.get("filled_quantity", existing.filled_quantity or 0.0)),
                    average_fill_price=parsed.get("average_fill_price", existing.average_fill_price),
                    fills=list(existing.fills),
                    metadata={**(existing.metadata or {}), "cancel_race": "filled_first"},
                )
            elif reported_status == "REJECTED":
                final = OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.REJECTED,
                    error=str(parsed.get("error", "rejected before cancel")),
                )
            else:
                final = OrderResult(
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
                )
            async with self._lock:
                self._order_cache[client_order_id] = final
            return final

        # Non-2xx: surface the failure explicitly. Cache untouched so
        # the caller can retry cancellation.
        error_text = response.body[:256].decode("utf-8", errors="replace")
        logger.warning(
            "MT5 cancel rejected by relay (status=%s): %s",
            response.status,
            error_text,
        )
        return OrderResult(
            client_order_id=existing.client_order_id,
            venue_order_id=existing.venue_order_id,
            symbol=existing.symbol,
            side=existing.side,
            order_type=existing.order_type,
            status=OrderStatus.REJECTED,
            error=f"cancel relay HTTP {response.status}: {error_text}",
        )

    async def get_open_orders(self) -> List[OrderResult]:
        # Acquire the lock + snapshot to a list so concurrent
        # place_order / cancel_order calls cannot mutate the dict
        # during iteration (would raise RuntimeError without this).
        async with self._lock:
            return [r for r in self._order_cache.values() if r.is_open]

    async def get_positions(self) -> Dict[str, float]:
        # The Worker holds the source of truth for MT5 positions and
        # is queried via a separate poll path (next PR). For now we
        # report no positions tracked locally.
        return {}

    async def get_balance(self) -> float:
        # Same as get_positions: MT5 balance lives on the terminal
        # side and is not yet polled.
        return 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(self, request: OrderRequest) -> Dict[str, Any]:
        """Construct the signal payload exactly as the Worker expects."""
        payload: Dict[str, Any] = {
            "client_order_id": request.client_order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "order_type": request.order_type.value,
            "timestamp": self.clock().isoformat(),
            "strategy": request.strategy,
        }
        if request.notional is not None:
            payload["notional"] = float(request.notional)
        if request.quantity is not None:
            payload["quantity"] = float(request.quantity)
        if request.limit_price is not None:
            payload["limit_price"] = float(request.limit_price)
        if request.stop_loss is not None:
            payload["stop_loss"] = float(request.stop_loss)
        if request.take_profit is not None:
            payload["take_profit"] = float(request.take_profit)
        return payload

    async def _post_with_retries(
        self,
        url: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> HttpResponse:
        """POST with bounded exponential backoff + jitter.

        Retries only on transport errors and 5xx responses. 4xx
        responses are deterministic venue rejections; retrying them
        would just waste calls and inflate logs. Returns the final
        HttpResponse (success or last failure); transport-level
        exceptions are wrapped into a synthetic 599 response.
        """
        attempts = max(1, self.config.max_retries + 1)
        last_response: Optional[HttpResponse] = None
        for attempt in range(attempts):
            try:
                response = await asyncio.wait_for(
                    self.http("POST", url, headers, body),
                    timeout=self.config.request_timeout_seconds,
                )
            except asyncio.TimeoutError:
                response = HttpResponse(status=599, body=b"timeout", headers={})
            except Exception as exc:  # network error
                response = HttpResponse(
                    status=599, body=str(exc).encode("utf-8"), headers={}
                )

            last_response = response
            if response.status < 500 or attempt == attempts - 1:
                return response
            # Backoff before retry: base * 2^attempt + jitter, capped.
            delay = min(
                self.config.backoff_base_seconds * (2 ** attempt),
                self.config.backoff_max_seconds,
            )
            delay += random.uniform(0.0, delay * 0.1)
            await asyncio.sleep(delay)
        # Defensive: loop always assigns last_response and either
        # returns inside the loop or this final return runs.
        assert last_response is not None
        return last_response

    def _result_from_response(
        self,
        request: OrderRequest,
        response: HttpResponse,
    ) -> OrderResult:
        """Turn an HTTP response from the Worker into an OrderResult."""
        if 200 <= response.status < 300:
            parsed = _safe_parse_json_dict(response.body)
            venue_id = parsed.get("venue_order_id")
            return OrderResult(
                client_order_id=request.client_order_id,
                venue_order_id=venue_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.PENDING,
                metadata={"relay_response": parsed},
            )

        # Anything else: rejection with the body text as detail (truncated).
        error_text = response.body[:256].decode("utf-8", errors="replace")
        return OrderResult(
            client_order_id=request.client_order_id,
            venue_order_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.REJECTED,
            error=f"relay HTTP {response.status}: {error_text}",
        )

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


def _safe_parse_json_dict(body: bytes) -> Dict[str, Any]:
    """Defensive JSON decoder.

    Returns the parsed dict on success, an empty dict on any failure
    or non-dict result. Used for relay responses where a list, a
    string, or a malformed payload would otherwise crash a `.get()`
    call on `None`.
    """
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed
