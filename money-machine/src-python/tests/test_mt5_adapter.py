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
