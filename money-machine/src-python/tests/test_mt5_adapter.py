"""
Tests for the MT5 execution adapter.

Coverage:

  - Canonical payload serialisation: sorted keys, no whitespace,
    UTF-8 bytes, idempotent across equal dicts in different
    insertion orders.
  - Signing + verification round trip using inline_signer and the
    matching Ed25519 public key. A flipped byte invalidates the
    signature, locking down the canonicalisation pipeline.
  - place_order happy path: signed envelope reaches a fake HTTP
    transport, comes back 202 PENDING, idempotent re-submission
    returns the cached result without re-signing.
  - Transport rejection: 4xx returns REJECTED with the body in
    `error`; 5xx triggers bounded retry then surfaces the final
    response.
  - Retry loop on 5xx: backoff is non-blocking under
    asyncio.sleep monkey patch, the same signed body is reused
    across attempts (not re-signed).
  - Cancel happy path: sends a signed CANCEL envelope, transitions
    the cached order to CANCELLED on 200.
  - Validation: malformed OrderRequest (missing size) returns a
    REJECTED result without ever calling the HTTP transport.
  - Keychain signer: raises AdapterError when no key is present
    (mocked keyring).

The adapter never talks to a real network or keychain. All HTTP
calls go through an injectable `http_client` callable; all signing
goes through an injectable `signer`. The cryptography dependency is
required, but if it is unavailable the entire test module skips so
the rest of the suite keeps running.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

cryptography = pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature  # noqa: E402

from engine.adapters import (  # noqa: E402
    AdapterError,
    HttpResponse,
    MT5Adapter,
    MT5Config,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    canonical_payload,
    inline_signer,
)


def _run(coro):
    return asyncio.run(coro)


# Fixed seed -> deterministic key in tests.
TEST_SEED = b"a" * 32


def _signer():
    return inline_signer(TEST_SEED)


def _public_key() -> Ed25519PublicKey:
    return Ed25519PrivateKey.from_private_bytes(TEST_SEED).public_key()


class _FakeHttp:
    """Records every call and replays a scripted sequence of responses."""

    def __init__(self, responses: List[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: List[Tuple[str, str, Dict[str, str], bytes]] = []

    async def __call__(
        self, method: str, url: str, headers: Dict[str, str], body: bytes
    ) -> HttpResponse:
        self.calls.append((method, url, dict(headers), body))
        if not self._responses:
            return HttpResponse(status=500, body=b"out of scripted responses", headers={})
        return self._responses.pop(0)


def _request(client_id: str = "ord-1", **overrides: Any) -> OrderRequest:
    base: Dict[str, Any] = dict(
        client_order_id=client_id,
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        notional=10_000.0,
        stop_loss=1.0750,
        take_profit=1.0900,
        strategy="momentum-v1",
    )
    base.update(overrides)
    return OrderRequest(**base)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_payload_is_deterministic_and_sorted() -> None:
    a = {"b": 1, "a": 2, "c": [3, 1, 2]}
    b = {"c": [3, 1, 2], "a": 2, "b": 1}
    assert canonical_payload(a) == canonical_payload(b)
    out = canonical_payload(a).decode("utf-8")
    # No whitespace, keys sorted alphabetically.
    assert out == '{"a":2,"b":1,"c":[3,1,2]}'


def test_canonical_payload_is_utf8_bytes() -> None:
    out = canonical_payload({"symbol": "EUR/USD"})
    assert isinstance(out, bytes)
    assert out == b'{"symbol":"EUR/USD"}'


def test_canonical_payload_preserves_non_ascii_as_utf8() -> None:
    """Python's default `ensure_ascii=True` would emit `\\uXXXX`
    escapes; JavaScript's JSON.stringify does not. If we let the
    default through, the bytes signed on the Python side would
    differ from what the Worker re-serialises, and every Ed25519
    verification would fail on non-ASCII payloads.
    """
    arabic = canonical_payload({"strategy": "إستراتيجية"})
    spanish = canonical_payload({"strategy": "mañana"})
    # Raw UTF-8 bytes, not \uXXXX escapes.
    assert "إستراتيجية".encode("utf-8") in arabic
    assert "mañana".encode("utf-8") in spanish
    assert b"\\u" not in arabic
    assert b"\\u" not in spanish


# ---------------------------------------------------------------------------
# Signing round trip
# ---------------------------------------------------------------------------


def test_inline_signer_round_trip_with_public_key() -> None:
    payload = canonical_payload({"x": 1, "y": "two"})
    sig = _signer()(payload)
    # Public key verifies the signature.
    _public_key().verify(sig, payload)


def test_flipped_byte_invalidates_signature() -> None:
    payload = canonical_payload({"x": 1, "y": "two"})
    sig = _signer()(payload)
    tampered = bytearray(payload)
    tampered[0] ^= 0x01
    with pytest.raises(InvalidSignature):
        _public_key().verify(sig, bytes(tampered))


def test_inline_signer_rejects_wrong_seed_length() -> None:
    with pytest.raises(AdapterError):
        inline_signer(b"too-short")


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------


def test_place_order_signs_and_posts_to_relay() -> None:
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=202, body=b'{"venue_order_id":"v-9"}', headers={})])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(oracle_url="https://oracle.test", max_retries=0),
        )

        result = await adapter.place_order(_request("ord-1"))

        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == "v-9"
        # Exactly one HTTP call, to the signals endpoint, with the
        # client_order_id surfaced in the header.
        assert len(http.calls) == 1
        method, url, headers, body = http.calls[0]
        assert method == "POST"
        assert url == "https://oracle.test/signals"
        assert headers["X-Client-Order-Id"] == "ord-1"
        # The body parses as an envelope with payload + signature +
        # public_key_id, and the signature verifies against the
        # canonical payload.
        envelope = json.loads(body)
        assert set(envelope.keys()) == {"payload", "signature", "public_key_id"}
        sig = bytes.fromhex(envelope["signature"])
        canonical = canonical_payload(envelope["payload"])
        _public_key().verify(sig, canonical)
        # Payload carries the order fields.
        assert envelope["payload"]["client_order_id"] == "ord-1"
        assert envelope["payload"]["symbol"] == "EURUSD"
        assert envelope["payload"]["side"] == "BUY"
        assert envelope["payload"]["stop_loss"] == 1.0750

    _run(scenario())


def test_place_order_is_idempotent_on_client_order_id() -> None:
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=202, body=b'{"venue_order_id":"v-9"}', headers={})])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=0),
        )

        first = await adapter.place_order(_request("ord-1"))
        second = await adapter.place_order(_request("ord-1"))

        assert first == second
        # Only one HTTP call: the second submission was served from cache.
        assert len(http.calls) == 1

    _run(scenario())


def test_place_order_rejected_on_4xx_without_retry() -> None:
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=400, body=b"bad signal shape", headers={})])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=3),
        )
        result = await adapter.place_order(_request("ord-bad"))
        assert result.status is OrderStatus.REJECTED
        assert "HTTP 400" in (result.error or "")
        assert "bad signal shape" in (result.error or "")
        # 4xx must NOT retry: exactly one call.
        assert len(http.calls) == 1

    _run(scenario())


def test_place_order_retries_on_5xx_then_succeeds() -> None:
    async def scenario(monkeypatch_sleep) -> None:
        http = _FakeHttp([
            HttpResponse(status=502, body=b"bad gateway", headers={}),
            HttpResponse(status=503, body=b"unavailable", headers={}),
            HttpResponse(status=202, body=b'{"venue_order_id":"v-12"}', headers={}),
        ])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=3, backoff_base_seconds=0.0),
        )
        result = await adapter.place_order(_request("ord-retry"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == "v-12"
        # Three attempts, same signed body each time.
        assert len(http.calls) == 3
        bodies = {tuple(c[3]) for c in http.calls}
        assert len(bodies) == 1, "retry must reuse the same signed body"

    async def main() -> None:
        # Speed test: avoid actually sleeping during backoff.
        original_sleep = asyncio.sleep

        async def _noop_sleep(_d: float) -> None:
            return None

        asyncio.sleep = _noop_sleep  # type: ignore[assignment]
        try:
            await scenario(_noop_sleep)
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

    _run(main())


def test_place_order_surfaces_final_failure_after_retries_exhausted() -> None:
    async def scenario() -> None:
        responses = [HttpResponse(status=500, body=b"oops", headers={})] * 4
        http = _FakeHttp(responses)
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=3, backoff_base_seconds=0.0),
        )

        original_sleep = asyncio.sleep
        async def _noop(_d: float) -> None:
            return None
        asyncio.sleep = _noop  # type: ignore[assignment]
        try:
            result = await adapter.place_order(_request("ord-fail"))
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert result.status is OrderStatus.REJECTED
        assert "HTTP 500" in (result.error or "")
        # max_retries=3 means 1 initial + 3 retries = 4 attempts.
        assert len(http.calls) == 4

    _run(scenario())


def test_place_order_validates_request_before_signing() -> None:
    async def scenario() -> None:
        http = _FakeHttp([])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        bad = OrderRequest(
            client_order_id="bad",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=None,  # neither size given
            quantity=None,
        )
        result = await adapter.place_order(bad)
        assert result.status is OrderStatus.REJECTED
        assert "either quantity or notional" in (result.error or "")
        # No HTTP call: validation short-circuits before signing.
        assert http.calls == []

    _run(scenario())


def test_transport_exception_becomes_synthetic_599() -> None:
    async def scenario() -> None:
        async def _boom(*_a: Any, **_kw: Any) -> HttpResponse:
            raise RuntimeError("dns failure")

        adapter = MT5Adapter(
            http_client=_boom,
            signer=_signer(),
            config=MT5Config(max_retries=0),
        )
        result = await adapter.place_order(_request("ord-tx"))
        assert result.status is OrderStatus.REJECTED
        assert "HTTP 599" in (result.error or "")
        assert "dns failure" in (result.error or "")

    _run(scenario())


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_transitions_pending_to_cancelled_on_200() -> None:
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v-1"}', headers={}),
            HttpResponse(status=200, body=b'{"cancelled":true}', headers={}),
        ])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=0),
        )

        await adapter.place_order(_request("ord-c"))
        cancelled = await adapter.cancel_order("ord-c")

        assert cancelled.status is OrderStatus.CANCELLED
        # Two HTTP calls: place + cancel.
        assert len(http.calls) == 2
        # Second call targets the cancel endpoint with the same client id.
        method, url, headers, _body = http.calls[1]
        assert method == "POST"
        assert url.endswith("/signals/cancel")
        assert headers["X-Client-Order-Id"] == "ord-c"

    _run(scenario())


def test_cancel_unknown_order_returns_rejection() -> None:
    async def scenario() -> None:
        http = _FakeHttp([])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.cancel_order("never-seen")
        assert result.status is OrderStatus.REJECTED
        assert "unknown" in (result.error or "").lower()
        # No HTTP call: unknown ids never hit the relay.
        assert http.calls == []

    _run(scenario())


# ---------------------------------------------------------------------------
# Keychain signer
# ---------------------------------------------------------------------------


def test_signing_failure_in_place_order_is_not_cached() -> None:
    """A transient signing failure (HSM blip, keychain race) must
    not be persisted to _order_cache. A later retry with the same
    client_order_id must be allowed to re-sign and re-POST.

    Mirror of the fix for cancel_order signing failures: signing
    is a pre-venue local operation, so no exposure exists. Caching
    the REJECTED would permanently short-circuit retries until
    process restart.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v-retry"}', headers={}),
        ])
        real = _signer()
        call_count = [0]

        def flaky_signer(payload: bytes) -> bytes:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient HSM outage")
            return real(payload)

        adapter = MT5Adapter(http_client=http, signer=flaky_signer)
        # First attempt: signing throws -> REJECTED, no HTTP call,
        # no cache entry.
        first = await adapter.place_order(_request("ord-retry"))
        assert first.status is OrderStatus.REJECTED
        assert "signing failed" in (first.error or "")
        assert http.calls == []

        # Second attempt with the SAME id: signing succeeds this
        # time and the order reaches the relay. If we had cached
        # the first REJECTED, this would return the cached failure
        # instead and never POST.
        second = await adapter.place_order(_request("ord-retry"))
        assert second.status is OrderStatus.PENDING
        assert second.venue_order_id == "v-retry"
        assert len(http.calls) == 1

    _run(scenario())


def test_cancel_signing_failure_returns_rejected_not_stale() -> None:
    """A signing failure during cancel must NOT silently return the
    stale PENDING order. Surface it as REJECTED with the error
    text so callers can distinguish 'cancel failed' from 'cancel
    succeeded'.
    """
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={})])

        # A signer that succeeds first (for place_order), then fails
        # on the second call (cancel).
        call_count = [0]
        real_signer = _signer()

        def flaky_signer(payload: bytes) -> bytes:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("hsm offline")
            return real_signer(payload)

        adapter = MT5Adapter(http_client=http, signer=flaky_signer)
        await adapter.place_order(_request("ord-cs"))
        result = await adapter.cancel_order("ord-cs")
        assert result.status is OrderStatus.REJECTED
        assert "signing failed" in (result.error or "")

    _run(scenario())


def test_cancel_relay_failure_returns_rejected_not_stale() -> None:
    """Non-2xx response on the cancel call must NOT return the
    stale order; return a REJECTED with the HTTP error text.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={}),
            HttpResponse(status=400, body=b"cancel rejected by relay", headers={}),
        ])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        await adapter.place_order(_request("ord-cr"))
        result = await adapter.cancel_order("ord-cr")
        assert result.status is OrderStatus.REJECTED
        assert "cancel relay HTTP 400" in (result.error or "")
        assert "cancel rejected by relay" in (result.error or "")

    _run(scenario())


def test_cancel_respects_relay_filled_response() -> None:
    """If the relay reports the order already filled during a cancel
    race, the cache must reflect FILLED, not CANCELLED.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={}),
            HttpResponse(
                status=200,
                body=b'{"status":"FILLED","filled_quantity":0.1,"average_fill_price":50000.0}',
                headers={},
            ),
        ])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        await adapter.place_order(_request("ord-race"))
        result = await adapter.cancel_order("ord-race")
        assert result.status is OrderStatus.FILLED
        assert result.filled_quantity == pytest.approx(0.1)
        assert result.metadata.get("cancel_race") == "filled_first"

    _run(scenario())


def test_cancel_request_includes_public_key_header() -> None:
    """The relay routes/authenticates on X-Public-Key-Id; the cancel
    endpoint must include the same header place_order uses.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={}),
            HttpResponse(status=200, body=b'{"cancelled":true}', headers={}),
        ])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            public_key_id="prod-key-1",
        )
        await adapter.place_order(_request("ord-h"))
        await adapter.cancel_order("ord-h")
        # Second call is the cancel.
        _, _, headers, _ = http.calls[1]
        assert headers.get("X-Public-Key-Id") == "prod-key-1"

    _run(scenario())


def test_response_non_dict_body_does_not_crash() -> None:
    """A relay that returns a JSON list or a bare string on success
    must not crash the adapter when we look up `venue_order_id`.
    """
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=202, body=b'["unexpected"]', headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-list"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id is None

    _run(scenario())


def test_concurrent_place_order_same_id_submits_once() -> None:
    """Two coroutines calling place_order with the same
    client_order_id in parallel must hit the relay exactly once.
    The second caller awaits the in-flight Future instead of
    re-signing and re-POSTing.
    """
    async def scenario() -> None:
        slow_event = asyncio.Event()

        async def slow_http(method, url, headers, body):
            # Hold the first request open so the second caller has
            # time to enter place_order and observe the in-flight
            # placeholder.
            await slow_event.wait()
            return HttpResponse(status=202, body=b'{"venue_order_id":"v-only"}', headers={})

        call_count = [0]

        async def counting_http(method, url, headers, body):
            call_count[0] += 1
            return await slow_http(method, url, headers, body)

        adapter = MT5Adapter(http_client=counting_http, signer=_signer())

        async def caller():
            return await adapter.place_order(_request("ord-concurrent"))

        first = asyncio.create_task(caller())
        # Give the first call a chance to mark in-flight.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = asyncio.create_task(caller())
        await asyncio.sleep(0)
        # Now release the HTTP response.
        slow_event.set()

        r1, r2 = await asyncio.gather(first, second)
        assert r1.status is OrderStatus.PENDING
        assert r1 == r2
        # The relay was hit ONCE despite two place_order calls.
        assert call_count[0] == 1

    _run(scenario())


def test_concurrent_cancel_order_same_id_hits_relay_once() -> None:
    """Two coroutines calling cancel_order with the same
    client_order_id concurrently must only POST /signals/cancel
    once. The second caller awaits the in-flight Future and
    observes the same result as the first.
    """
    async def scenario() -> None:
        place_event = asyncio.Event()
        cancel_event = asyncio.Event()
        cancel_calls = [0]

        async def http(method, url, headers, body):
            if url.endswith("/signals"):
                return HttpResponse(
                    status=202, body=b'{"venue_order_id":"v"}', headers={}
                )
            # /signals/cancel: count and gate.
            cancel_calls[0] += 1
            await cancel_event.wait()
            return HttpResponse(status=200, body=b'{"cancelled":true}', headers={})

        adapter = MT5Adapter(http_client=http, signer=_signer())
        await adapter.place_order(_request("ord-cc"))

        first = asyncio.create_task(adapter.cancel_order("ord-cc"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = asyncio.create_task(adapter.cancel_order("ord-cc"))
        await asyncio.sleep(0)
        cancel_event.set()

        r1, r2 = await asyncio.gather(first, second)
        assert r1.status is OrderStatus.CANCELLED
        assert r1 == r2
        # /signals/cancel was hit exactly once despite two concurrent
        # cancel_order calls.
        assert cancel_calls[0] == 1

    _run(scenario())


def test_cancel_filled_race_response_with_null_filled_quantity() -> None:
    """A relay that returns FILLED with `filled_quantity: null`
    must not raise; the adapter should coerce missing/null fields
    to a safe default rather than crash.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={}),
            HttpResponse(
                status=200,
                body=b'{"status":"FILLED","filled_quantity":null,"average_fill_price":null}',
                headers={},
            ),
        ])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        await adapter.place_order(_request("ord-null"))
        result = await adapter.cancel_order("ord-null")
        # Did not raise; FILLED is reported with safe defaults.
        assert result.status is OrderStatus.FILLED

    _run(scenario())


def test_place_order_cancellation_cleans_up_in_flight() -> None:
    """If place_order is cancelled mid HTTP retry (asyncio task
    cancellation), the in-flight slot must be cleared and any
    waiting concurrent caller must observe a deterministic result
    instead of awaiting forever. The synthetic 'aborted' result
    must NOT be persisted to `_order_cache`, so a later retry of
    the same client_order_id can still reach the relay.
    """
    async def scenario() -> None:
        gate = asyncio.Event()
        post_calls = [0]

        async def gated_http(method, url, headers, body):
            post_calls[0] += 1
            if post_calls[0] == 1:
                # First call blocks until released; the test cancels
                # the first task before the gate opens.
                await gate.wait()
                return HttpResponse(status=202, body=b'{"venue_order_id":"never"}', headers={})
            # Subsequent calls (the post-abort retry) succeed.
            return HttpResponse(status=202, body=b'{"venue_order_id":"v-retry"}', headers={})

        adapter = MT5Adapter(http_client=gated_http, signer=_signer())

        first = asyncio.create_task(adapter.place_order(_request("ord-cancel")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = asyncio.create_task(adapter.place_order(_request("ord-cancel")))
        await asyncio.sleep(0)

        first.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await first
        # Concurrent waiter resolves to the synthetic REJECTED.
        result = await asyncio.wait_for(second, timeout=2.0)
        assert result.status is OrderStatus.REJECTED
        assert "aborted" in (result.error or "")

        # A NEW place_order call for the same client_order_id must
        # be able to reach the relay; the abort-result was NOT
        # cached as a definitive outcome.
        gate.set()
        retry = await adapter.place_order(_request("ord-cancel"))
        assert retry.status is OrderStatus.PENDING
        assert retry.venue_order_id == "v-retry"
        # Exactly two HTTP calls: the cancelled first one and the
        # successful retry. The concurrent waiter never POSTed.
        assert post_calls[0] == 2

    _run(scenario())


def test_place_order_cancellation_during_post_lock_acquire_still_cleans_up() -> None:
    """Cancellation between `_post_with_retries` succeeding and the
    second `async with self._lock` completing must still clear the
    in-flight slot and resolve the Future. Otherwise concurrent
    callers can deadlock awaiting a Future that nothing will ever
    set.

    Engineering the race: we hold `self._lock` from outside the
    adapter so the second acquisition inside place_order blocks;
    while it is blocked we cancel the task. The in-flight cleanup
    must still run via the try/finally and any concurrent caller
    must observe a resolved Future.
    """
    async def scenario() -> None:
        http = _FakeHttp([HttpResponse(status=202, body=b'{}', headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())

        # Pre-acquire the adapter lock so the post-response lock
        # await inside place_order blocks.
        await adapter._lock.acquire()
        try:
            first = asyncio.create_task(adapter.place_order(_request("ord-late-cancel")))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Now a concurrent caller parks on the in-flight Future.
            second = asyncio.create_task(adapter.place_order(_request("ord-late-cancel")))
            await asyncio.sleep(0)

            # Cancel while `first` is waiting for the lock we hold.
            first.cancel()
            await asyncio.sleep(0)
        finally:
            # Release the lock so the cleanup paths can run.
            adapter._lock.release()

        with pytest.raises((asyncio.CancelledError, Exception)):
            await first
        # Concurrent waiter must NOT hang: the Future was resolved
        # by the finally block.
        result = await asyncio.wait_for(second, timeout=2.0)
        assert result.status in (OrderStatus.PENDING, OrderStatus.REJECTED)

    _run(scenario())


def test_get_open_orders_uses_lock_for_consistent_snapshot() -> None:
    """Sanity: even under no contention, get_open_orders returns a
    coherent snapshot list rather than iterating the live dict.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v1"}', headers={}),
            HttpResponse(status=202, body=b'{"venue_order_id":"v2"}', headers={}),
        ])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        await adapter.place_order(_request("o1"))
        await adapter.place_order(_request("o2"))
        opens = await adapter.get_open_orders()
        assert len(opens) == 2

    _run(scenario())


def test_mt5_import_does_not_pull_in_paper_or_evm_modules() -> None:
    """Importing MT5Adapter from engine.adapters must NOT
    transitively load PaperAdapter (which pulls pandas/numpy via
    engine.backtest) or the EVMAdapter (eth-account). An MT5-only
    runtime should not pay for adapters it does not use.

    Runs the import in a clean subprocess so it does not pollute
    the rest of the test process's sys.modules.
    """
    import os
    import subprocess

    script = (
        "import sys\n"
        "from engine.adapters import MT5Adapter\n"
        "print('paper_loaded=' + str('engine.adapters.paper' in sys.modules))\n"
        "print('evm_loaded=' + str('engine.adapters.evm' in sys.modules))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PYTHON)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "paper_loaded=False" in result.stdout, result.stdout
    assert "evm_loaded=False" in result.stdout, result.stdout


def test_get_balance_returns_configured_account_equity() -> None:
    """Pipeline reads adapter.get_balance() as equity; the MT5
    adapter must therefore return a usable value, not 0.0.
    """
    async def scenario() -> None:
        adapter = MT5Adapter(
            http_client=_FakeHttp([]),
            signer=_signer(),
            account_equity=25_000.0,
            positions={"EURUSD": 0.5, "GBPUSD": -0.25},
        )
        assert (await adapter.get_balance()) == pytest.approx(25_000.0)
        positions = await adapter.get_positions()
        assert positions == {"EURUSD": 0.5, "GBPUSD": -0.25}

    _run(scenario())


def test_set_account_equity_updates_balance_and_rejects_garbage() -> None:
    adapter = MT5Adapter(http_client=_FakeHttp([]), signer=_signer())
    adapter.set_account_equity(12_345.67)
    assert _run(adapter.get_balance()) == pytest.approx(12_345.67)
    with pytest.raises(AdapterError):
        adapter.set_account_equity(float("nan"))
    with pytest.raises(AdapterError):
        adapter.set_account_equity(-1.0)


def test_request_metadata_is_forwarded_in_signed_payload() -> None:
    """Strategies set MT5-specific overrides (instrument hints,
    routing flags) in OrderRequest.metadata. The adapter must
    serialise them into the signed envelope so the relay and the
    MQL EA can read them.
    """
    async def scenario() -> None:
        http = _FakeHttp([
            HttpResponse(status=202, body=b'{"venue_order_id":"v"}', headers={})
        ])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        req = _request("ord-meta")
        # Replace the default metadata with the routing hints the
        # strategy emitted.
        req = OrderRequest(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            notional=req.notional,
            strategy=req.strategy,
            metadata={
                "mt5_instrument": "EURUSD.r",
                "comment": "scalp-v1",
                "magic_number": 42,
            },
        )
        await adapter.place_order(req)
        _, _, _, body = http.calls[0]
        envelope = json.loads(body)
        meta = envelope["payload"].get("metadata")
        assert meta == {
            "mt5_instrument": "EURUSD.r",
            "comment": "scalp-v1",
            "magic_number": 42,
        }

    _run(scenario())


def test_keychain_signer_raises_when_key_missing(monkeypatch) -> None:
    from engine.adapters import mt5 as mt5_module

    class _FakeKeyring:
        @staticmethod
        def get_password(_service: str, _account: str) -> Optional[str]:
            return None

    monkeypatch.setitem(sys.modules, "keyring", _FakeKeyring())
    signer = mt5_module.keychain_signer()
    with pytest.raises(AdapterError):
        signer(b"anything")


# ---------------------------------------------------------------------------
# venue_order_id validation removal (PR: fail-closed check removed)
#
# Previously the adapter rejected any 2xx response whose venue_order_id was
# absent or not a non-empty string. That guard was removed so the adapter
# accepts all 2xx relay responses and passes venue_order_id through as-is
# (including None). The tests below lock down the new behaviour.
# ---------------------------------------------------------------------------


def test_2xx_with_null_venue_order_id_returns_pending() -> None:
    """A 2xx response with `"venue_order_id": null` in the JSON body must
    return PENDING with venue_order_id=None, not REJECTED.

    This is a regression guard: the old code rejected such a response with
    'relay HTTP success missing required venue_order_id; fail-closed'.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": null, "ok": true}'
        http = _FakeHttp([HttpResponse(status=202, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-null-vid"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id is None

    _run(scenario())


def test_2xx_with_missing_venue_order_id_key_returns_pending() -> None:
    """A 2xx response with no `venue_order_id` key at all must return
    PENDING with venue_order_id=None.

    The old validation would reject this as a missing required field.
    The new behaviour is to pass None through and let the caller decide.
    """
    async def scenario() -> None:
        body = b'{"status": "queued"}'
        http = _FakeHttp([HttpResponse(status=200, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-no-vid-key"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id is None

    _run(scenario())


def test_2xx_with_whitespace_only_venue_order_id_returns_pending() -> None:
    """A 2xx response with venue_order_id set to a whitespace-only string
    must return PENDING. The old code required a non-empty string after
    strip(); the new code accepts whatever the relay returns.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": "   "}'
        http = _FakeHttp([HttpResponse(status=202, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-ws-vid"))
        assert result.status is OrderStatus.PENDING
        # The whitespace string is preserved as-is (not converted to None).
        assert result.venue_order_id == "   "

    _run(scenario())


def test_2xx_with_empty_string_venue_order_id_returns_pending() -> None:
    """An empty string venue_order_id must pass through without rejection.

    Previously `"".strip()` evaluated falsy and the adapter returned REJECTED.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": ""}'
        http = _FakeHttp([HttpResponse(status=200, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-empty-vid"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == ""

    _run(scenario())


def test_2xx_with_valid_venue_order_id_still_returns_pending_with_id() -> None:
    """Positive-control: a well-formed 2xx response with a valid venue_order_id
    must still return PENDING and preserve the id unchanged after the refactor.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": "relay-v-9999"}'
        http = _FakeHttp([HttpResponse(status=201, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-valid-vid"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == "relay-v-9999"

    _run(scenario())


def test_2xx_with_numeric_venue_order_id_returns_pending() -> None:
    """A relay that echoes a numeric venue_order_id (not a string) must
    not cause a rejection. The value is extracted as-is from the parsed dict.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": 42}'
        http = _FakeHttp([HttpResponse(status=200, body=body, headers={})])
        adapter = MT5Adapter(http_client=http, signer=_signer())
        result = await adapter.place_order(_request("ord-int-vid"))
        assert result.status is OrderStatus.PENDING
        assert result.venue_order_id == 42

    _run(scenario())


# ---------------------------------------------------------------------------
# venue_order_id relaxation (PR: removed fail-closed check)
# ---------------------------------------------------------------------------


def test_place_order_accepts_missing_venue_order_id_in_2xx_dict_body() -> None:
    """After removing the fail-closed venue_order_id check, a 2xx
    response whose JSON dict does NOT contain `venue_order_id` must
    produce a PENDING result with venue_order_id=None instead of
    being rejected.
    """
    async def scenario() -> None:
        body = b'{"status":"queued"}'  # valid dict, no venue_order_id key
        http = _FakeHttp([HttpResponse(status=202, body=body, headers={})])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=0),
        )
        result = await adapter.place_order(_request("ord-no-vid"))
        assert result.status is OrderStatus.PENDING, result
        assert result.venue_order_id is None

    _run(scenario())


def test_place_order_accepts_empty_string_venue_order_id_in_2xx_body() -> None:
    """Before the PR the server would fail-closed if venue_order_id was
    an empty string. The relaxed code must now accept it and surface the
    empty string as venue_order_id rather than rejecting the order.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id":""}'
        http = _FakeHttp([HttpResponse(status=202, body=body, headers={})])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(max_retries=0),
        )
        result = await adapter.place_order(_request("ord-empty-vid"))
        assert result.status is OrderStatus.PENDING, result
        # Empty string is falsy but no longer causes a rejection.
        assert result.venue_order_id == ""

    _run(scenario())


def test_non_2xx_still_rejected_regardless_of_venue_order_id() -> None:
    """Non-2xx responses must still return REJECTED even if the body
    contains a venue_order_id. The venue_order_id validation removal must
    not accidentally promote error responses to PENDING.
    """
    async def scenario() -> None:
        body = b'{"venue_order_id": "should-not-matter", "error": "bad request"}'
        http = _FakeHttp([
            # All 5xx retries (adapter retries on 5xx; give it enough responses).
            HttpResponse(status=400, body=body, headers={}),
        ])
        adapter = MT5Adapter(
            http_client=http,
            signer=_signer(),
            config=MT5Config(oracle_url="https://example.com", max_retries=1),
        )
        result = await adapter.place_order(_request("ord-4xx-vid"))
        assert result.status is OrderStatus.REJECTED

    _run(scenario())
