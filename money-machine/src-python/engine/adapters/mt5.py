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

    Sorted keys, no whitespace, UTF-8 encoded. `ensure_ascii=False`
    so non-ASCII characters (e.g. an Arabic strategy name, a
    Unicode comment) are emitted as raw UTF-8 bytes; that matches
    what JavaScript's `JSON.stringify` produces on the Worker side
    and is required for Ed25519 signature verification to succeed
    across languages. Python's default `ensure_ascii=True` would
    emit `\\uXXXX` escape sequences and break verification on the
    very first non-ASCII payload.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


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
        account_equity: Optional[float] = None,
        positions: Optional[Dict[str, float]] = None,
    ) -> None:
        self.http = http_client
        self.signer = signer
        self.config = config or MT5Config()
        self.public_key_id = public_key_id
        self.clock = clock
        # MT5 holds the source-of-truth balance on the terminal side;
        # there is no live poll path yet. We accept an externally
        # supplied equity at construction time and expose
        # `set_account_equity` so a future MT5 poll service (or
        # an operator-managed config) can keep the value current.
        # Without this the SignalPipeline now reads
        # `adapter.get_balance()` as the equity for risk sizing and
        # would short-circuit every order with equity == 0.
        self._account_equity = (
            float(account_equity) if account_equity is not None else 0.0
        )
        self._positions: Dict[str, float] = dict(positions or {})
        # Idempotency cache: client_order_id -> last definitive result.
        # Anything PENDING is also cached so a retry sees the same
        # response without re-signing.
        self._order_cache: Dict[str, OrderResult] = {}
        # In-flight placeholder maps: when a place_order or
        # cancel_order call is mid HTTP retry, its client_order_id
        # maps to a Future so a concurrent caller for the same id
        # awaits the original result instead of submitting a
        # duplicate signed envelope. Cleared once the result lands
        # in `_order_cache`.
        self._in_flight: Dict[str, "asyncio.Future[OrderResult]"] = {}
        self._in_flight_cancels: Dict[str, "asyncio.Future[OrderResult]"] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ExecutionAdapter API
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        # Build the signed envelope under the lock (short, in-memory
        # only), then release the lock before the HTTP call so a
        # slow/timing-out relay request does NOT serialise every
        # other order on this adapter instance. An in-flight Future
        # parked in `self._in_flight` makes concurrent calls for the
        # SAME client_order_id wait on the original result rather
        # than racing to sign and POST a duplicate signal.
        async with self._lock:
            cached = self._order_cache.get(request.client_order_id)
            if cached is not None:
                logger.debug(
                    "MT5 place_order idempotent hit for %s",
                    request.client_order_id,
                )
                return cached
            inflight = self._in_flight.get(request.client_order_id)
            if inflight is not None:
                logger.debug(
                    "MT5 place_order awaiting in-flight call for %s",
                    request.client_order_id,
                )
                # Release the lock before awaiting; the original
                # caller will set the Future's result when its HTTP
                # round trip completes.
                pass
            else:
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
                    # AdapterError is included here; the adapter
                    # contract is "no exceptions, return REJECTED"
                    # and signing is a routine failure path
                    # (missing keychain entry, libsodium quirk,
                    # transient HSM outage).
                    #
                    # Intentionally NOT writing this REJECTED to
                    # `_order_cache`: no request reached the relay,
                    # so the order has no exposure. Caching a
                    # signing failure would permanently short-circuit
                    # every later retry of the same client_order_id
                    # from cache and prevent the natural retry path
                    # for a transient HSM blip. The id is also kept
                    # out of `_in_flight` because we never created
                    # a Future for it (the placeholder is set after
                    # signing succeeds).
                    return self._rejected(request, f"signing failed: {exc}")

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

                # Mark this id as in-flight; concurrent callers see
                # the Future and await it instead of re-signing.
                future: "asyncio.Future[OrderResult]" = asyncio.get_event_loop().create_future()
                self._in_flight[request.client_order_id] = future

        if inflight is not None:
            return await inflight

        # Lock released for the network call; in-flight Future
        # parked under the lock above. The whole tail of the method
        # is wrapped in try/finally so any cancellation point
        # (during _post_with_retries, between it and the second
        # lock acquisition, or while we wait for the lock itself)
        # is followed by a deterministic cleanup: the in-flight
        # slot is cleared and the Future is resolved, so concurrent
        # callers cannot deadlock awaiting a Future that nothing
        # else will ever set.
        result: Optional[OrderResult] = None
        cache_result = False
        try:
            try:
                response = await self._post_with_retries(url, headers, body)
                result = self._result_from_response(request, response)
                # A real venue outcome lands in the durable cache.
                cache_result = True
            except BaseException as exc:
                # Synthetic abort result: NOT cached as a
                # definitive outcome (a fresh standalone retry
                # must be able to re-sign and re-POST), but still
                # used to unblock concurrent waiters via the
                # Future.
                result = self._rejected(
                    request,
                    f"place_order aborted: {type(exc).__name__}: {exc}",
                )
                cache_result = False
                raise
        finally:
            try:
                async with self._lock:
                    if cache_result and result is not None:
                        self._order_cache[request.client_order_id] = result
                    self._in_flight.pop(request.client_order_id, None)
            except BaseException:
                # Second cleanup-time cancellation: drop the
                # in-flight slot synchronously so the Future
                # resolution below still happens.
                self._in_flight.pop(request.client_order_id, None)
            if not future.done():
                future.set_result(
                    result
                    if result is not None
                    else self._rejected(request, "place_order aborted before result")
                )

        assert result is not None
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

        Concurrent cancels for the same client_order_id are
        serialised through an in-flight Future map, mirroring
        place_order's idempotency pattern. Without it, two parallel
        cancellations would both POST `/signals/cancel` and one
        would see "already cancelled" / REJECTED from the relay,
        violating idempotent cancel semantics.
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

            inflight_cancel = self._in_flight_cancels.get(client_order_id)
            if inflight_cancel is not None:
                # Another coroutine is already cancelling this id;
                # await its result outside the lock instead of
                # signing and POSTing a duplicate.
                cancel_future = inflight_cancel
            else:
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

                cancel_future = asyncio.get_event_loop().create_future()
                self._in_flight_cancels[client_order_id] = cancel_future

        if inflight_cancel is not None:
            # Wait for the original cancel to complete and return
            # the same OrderResult so the second caller observes
            # the same outcome as the first.
            return await cancel_future

        # Release the lock for the network call so other orders are
        # not blocked while a cancel retries through the relay. Any
        # exception from the retry loop must still resolve the
        # in-flight Future so concurrent callers do not wait forever.
        try:
            response = await self._post_with_retries(url, headers, body)
            final = self._build_cancel_result(existing, response)
        except BaseException as exc:
            # Includes asyncio.CancelledError (which is a
            # BaseException, not Exception). Resolve the future so
            # any waiting callers see a deterministic result and
            # clear the in-flight slot before re-raising.
            failed = OrderResult(
                client_order_id=existing.client_order_id,
                venue_order_id=existing.venue_order_id,
                symbol=existing.symbol,
                side=existing.side,
                order_type=existing.order_type,
                status=OrderStatus.REJECTED,
                error=f"cancel aborted: {type(exc).__name__}",
            )
            async with self._lock:
                self._in_flight_cancels.pop(client_order_id, None)
            if not cancel_future.done():
                cancel_future.set_result(failed)
            raise

        async with self._lock:
            # Only persist to the cache when the cancel reached a
            # terminal state on the venue side (2xx CANCELLED /
            # FILLED / REJECTED). Pure transport failures leave the
            # cache untouched so the caller can retry.
            if final.status in (OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED):
                # Only overwrite the cached order on actual venue
                # responses (2xx), not on transport-layer rejections
                # surfaced as "cancel relay HTTP ...".
                if 200 <= response.status < 300:
                    self._order_cache[client_order_id] = final
            self._in_flight_cancels.pop(client_order_id, None)
        if not cancel_future.done():
            cancel_future.set_result(final)
        return final

    def _build_cancel_result(
        self,
        existing: OrderResult,
        response: HttpResponse,
    ) -> OrderResult:
        """Translate a cancel HTTP response into an OrderResult.

        2xx with `status=FILLED` in the body returns the FILLED
        race state; 2xx with `status=REJECTED` returns REJECTED;
        any other 2xx body becomes CANCELLED. Non-2xx returns a
        REJECTED result that includes the HTTP status and body
        text so the caller can decide whether to retry.
        """
        if 200 <= response.status < 300:
            parsed = _safe_parse_json_dict(response.body)
            reported_status = str(parsed.get("status", "")).upper()
            if reported_status == "FILLED":
                # Guard against a relay that sends `null` or a
                # non-numeric `filled_quantity`; falling back to
                # `existing.filled_quantity` keeps the result
                # well-typed.
                fq_raw = parsed.get("filled_quantity")
                try:
                    fq = float(fq_raw) if fq_raw is not None else float(
                        existing.filled_quantity or 0.0
                    )
                except (TypeError, ValueError):
                    fq = float(existing.filled_quantity or 0.0)
                fp_raw = parsed.get("average_fill_price", existing.average_fill_price)
                try:
                    fp = float(fp_raw) if fp_raw is not None else existing.average_fill_price
                except (TypeError, ValueError):
                    fp = existing.average_fill_price
                return OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.FILLED,
                    requested_quantity=existing.requested_quantity,
                    filled_quantity=fq,
                    average_fill_price=fp,
                    fills=list(existing.fills),
                    metadata={**(existing.metadata or {}), "cancel_race": "filled_first"},
                )
            if reported_status == "REJECTED":
                return OrderResult(
                    client_order_id=existing.client_order_id,
                    venue_order_id=existing.venue_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_type=existing.order_type,
                    status=OrderStatus.REJECTED,
                    error=str(parsed.get("error", "rejected before cancel")),
                )
            return OrderResult(
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

        # Non-2xx: surface the failure explicitly. Cache untouched
        # so the caller can retry cancellation.
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
        """Last known MT5 position snapshot.

        Default-empty until the caller wires up `set_positions` from
        an MT5 poll service. Returning {} here is safe even with the
        new pipeline contract because the risk shield's
        max_concurrent_positions rule then operates on a known-zero
        snapshot; the operator must set positions for the rule to
        be enforced against real MT5 exposure.
        """
        async with self._lock:
            return dict(self._positions)

    async def get_balance(self) -> float:
        """Last known MT5 account equity.

        Returns the value supplied at construction or set via
        `set_account_equity`. The previous 0.0 hardcoded return
        blocked every order routed through the SignalPipeline
        because the pipeline now sources equity from
        `adapter.get_balance()`; with equity <= 0 the risk shield
        rejects the order as INVALID_PORTFOLIO_STATE before any
        relay POST happens.
        """
        async with self._lock:
            return float(self._account_equity)

    def set_account_equity(self, value: float) -> None:
        """Update the MT5 account equity used by the pipeline for
        sizing and risk checks. Reject NaN/inf and negative values
        explicitly; those would silently corrupt downstream math.
        """
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                f"account_equity must be numeric, got {value!r}"
            ) from exc
        import math

        if not math.isfinite(v) or v < 0:
            raise AdapterError(
                f"account_equity must be finite and >= 0, got {value!r}"
            )
        self._account_equity = v

    def set_positions(self, positions: Dict[str, float]) -> None:
        """Update the cached MT5 position snapshot."""
        if not isinstance(positions, dict):
            raise AdapterError(
                f"positions must be a dict, got {type(positions).__name__}"
            )
        self._positions = dict(positions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(self, request: OrderRequest) -> Dict[str, Any]:
        """Construct the signal payload exactly as the Worker expects.

        Forwards `request.metadata` under a nested `metadata` key so
        MT5-specific overrides (instrument code, account hint,
        routing flags, comments) survive the trip through the
        pipeline. We do NOT flatten them into top-level fields to
        avoid colliding with the canonical keys above; the Worker
        and the MQL EA agree on `metadata.*` as the namespace for
        anything venue-specific.
        """
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
        if request.metadata:
            # Only serialise JSON-friendly values. Caller is
            # responsible for keeping metadata small (the canonical
            # form is signed, so big blobs slow signing too).
            payload["metadata"] = _stringify_metadata(request.metadata)
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


def _stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter metadata to JSON-serialisable scalars and short
    collections. Anything that does not round-trip through
    `json.dumps` is stringified so signing cannot fail on weird
    types the caller stuffed into the dict.
    """
    out: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            key = str(key)
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)
    return out


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
