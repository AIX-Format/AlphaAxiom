"""
Tests for the IPC auth layer and rate limiter.

Covers three required acceptance criteria from Sprint 0 Task 0.2:

    - valid token: command dispatches and returns a normal result.
    - invalid token: server responds with code 401 and never reaches
      the application command handler.
    - rate limit: once the bucket is drained the server responds with
      code 429.

The tests spin up a real `IPCServer` on an ephemeral port and talk to
it over loopback TCP, so they exercise the same wire protocol the
Tauri shell uses. No external network or keychain access is required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Tuple

# Allow running tests directly without installing the package.
SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from utils.ipc_auth import (  # noqa: E402  (import after sys.path tweak)
    TokenBucket,
    is_valid_token,
    parse_auth_header,
)
from utils.ipc_server import IPCServer  # noqa: E402


TEST_TOKEN = "test-token-deadbeef"


async def _echo_handler(command: str, payload: dict) -> dict:
    """Application handler that just echoes what it received."""
    return {"result": {"command": command, "payload": payload}, "error": None}


async def _send_raw(host: str, port: int, raw: bytes) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(raw)
        await writer.drain()
        line = await reader.readline()
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def _start_server(
    rate: float = 100.0, burst: float = 200.0, read_timeout_seconds: float = 5.0
) -> Tuple[IPCServer, asyncio.Task, Tuple[str, int]]:
    """
    Start a test IPCServer bound to an ephemeral loopback port and return the server instance, the background serve task, and the resolved (host, port) address.
    
    The server is configured for tests (uses the module's `_echo_handler` and `TEST_TOKEN`) and is started in a background task; the function waits until the server socket is bound before returning.
    
    Parameters:
        rate (float): Token refill rate (tokens per second) for the server's rate limiter.
        burst (float): Burst capacity (maximum tokens) for the server's rate limiter.
        read_timeout_seconds (float): Number of seconds the server will wait for a request body before timing out.
    
    Returns:
        Tuple[IPCServer, asyncio.Task, Tuple[str, int]]: A tuple containing
            - the started IPCServer instance,
            - the asyncio.Task running the server's serve loop,
            - a (host, port) tuple for the bound ephemeral endpoint.
    """
    server = IPCServer(
        command_handler=_echo_handler,
        host="127.0.0.1",
        port=0,  # ephemeral
        auth_token=TEST_TOKEN,
        rate_limit=rate,
        burst_limit=burst,
        read_timeout_seconds=read_timeout_seconds,
    )

    started = asyncio.Event()
    bound: dict = {}

    async def runner() -> None:
        srv = await asyncio.start_server(
            server.handle_client, server.host, server.port
        )
        server.server = srv
        sock = srv.sockets[0]
        bound["host"], bound["port"] = sock.getsockname()[:2]
        started.set()
        async with srv:
            await srv.serve_forever()

    task = asyncio.create_task(runner())
    await started.wait()
    return server, task, (bound["host"], bound["port"])


async def _shutdown(server: IPCServer, task: asyncio.Task) -> None:
    await server.stop()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ---------------------------------------------------------------------------
# Unit tests for pure helpers (no network).
# ---------------------------------------------------------------------------


def test_parse_auth_header_accepts_valid_format() -> None:
    assert parse_auth_header("X-Auth-Token: abc123") == "abc123"
    assert parse_auth_header("X-Auth-Token:abc123") == "abc123"
    assert parse_auth_header("X-Auth-Token:   abc123   ") == "abc123"


def test_parse_auth_header_rejects_bad_format() -> None:
    assert parse_auth_header("Authorization: abc123") is None
    assert parse_auth_header("X-Auth-Token:") is None
    assert parse_auth_header("") is None


def test_is_valid_token_handles_mismatch_and_empty() -> None:
    assert is_valid_token("a" * 16, "a" * 16) is True
    assert is_valid_token("a" * 16, "b" * 16) is False
    assert is_valid_token(None, "a" * 16) is False
    assert is_valid_token("", "a" * 16) is False
    assert is_valid_token("a" * 16, "") is False


def test_token_bucket_steady_state() -> None:
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    # Burst of 10 succeeds, 11th fails until refill.
    for _ in range(10):
        assert bucket.consume(now=0.0) is True
    assert bucket.consume(now=0.0) is False
    # After 1.0s we should regain ~10 tokens.
    assert bucket.consume(now=1.0) is True


# ---------------------------------------------------------------------------
# Integration tests: real TCP server, real protocol.
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_valid_token_dispatches_command() -> None:
    async def scenario() -> None:
        server, task, (host, port) = await _start_server()
        try:
            request = json.dumps({"command": "PING", "payload": {"hello": "world"}})
            wire = f"X-Auth-Token: {TEST_TOKEN}\n{request}\n".encode("utf-8")
            response = await _send_raw(host, port, wire)
            assert response.get("error") is None, response
            assert response["result"]["command"] == "PING"
            assert response["result"]["payload"] == {"hello": "world"}
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_invalid_token_returns_401() -> None:
    async def scenario() -> None:
        server, task, (host, port) = await _start_server()
        try:
            request = json.dumps({"command": "PING", "payload": {}})
            wire = f"X-Auth-Token: not-the-real-token\n{request}\n".encode("utf-8")
            response = await _send_raw(host, port, wire)
            assert response.get("code") == 401, response
            assert "auth" in response.get("error", "").lower()
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_missing_token_returns_401() -> None:
    async def scenario() -> None:
        server, task, (host, port) = await _start_server()
        try:
            request = json.dumps({"command": "PING", "payload": {}})
            # No auth header line at all, just the body twice so the
            # server reads a JSON line where it expects the header.
            wire = f"{request}\n{request}\n".encode("utf-8")
            response = await _send_raw(host, port, wire)
            assert response.get("code") == 401, response
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_rate_limit_returns_429_after_burst() -> None:
    """
    Verifies the server enforces the configured token-bucket rate limit by allowing requests up to the burst size and returning a 429 error once the burst is exhausted.
    
    The test sends multiple authenticated requests: the first N requests (where N equals the burst limit) must succeed, and the subsequent request must receive a response with code 429 and an error message referencing rate limiting.
    """
    async def scenario() -> None:
        # Tiny bucket so we can exhaust it in a handful of requests.
        server, task, (host, port) = await _start_server(rate=1.0, burst=3.0)
        try:
            request = json.dumps({"command": "PING", "payload": {}})
            wire = f"X-Auth-Token: {TEST_TOKEN}\n{request}\n".encode("utf-8")

            # First three should pass.
            for i in range(3):
                response = await _send_raw(host, port, wire)
                assert response.get("error") is None, f"req {i}: {response}"

            # Fourth should be rate-limited.
            response = await _send_raw(host, port, wire)
            assert response.get("code") == 429, response
            assert "rate" in response.get("error", "").lower()
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_oversized_ipc_body_returns_413() -> None:
    """
    Verifies the IPC server returns a 413 error when the request body exceeds IPCServer.MAX_BODY_BYTES.
    
    Sends an authenticated request whose JSON body is one byte larger than MAX_BODY_BYTES and asserts the server responds with code 413.
    """
    async def scenario() -> None:
        server, task, (host, port) = await _start_server()
        try:
            oversized = b'{"command":"PING","payload":"' + (b"x" * (IPCServer.MAX_BODY_BYTES + 1)) + b'"}\n'
            wire = f"X-Auth-Token: {TEST_TOKEN}\n".encode("utf-8") + oversized
            response = await _send_raw(host, port, wire)
            assert response.get("code") == 413, response
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_missing_body_times_out() -> None:
    """
    Verifies the IPC server responds with code 408 when the request body is not received before the read timeout.
    
    Starts an IPCServer with a short read timeout, sends only the authentication header (no JSON body), and asserts the server returns `code == 408`.
    """
    async def scenario() -> None:
        server, task, (host, port) = await _start_server(read_timeout_seconds=0.05)
        try:
            response = await _send_raw(host, port, f"X-Auth-Token: {TEST_TOKEN}\n".encode("utf-8"))
            assert response.get("code") == 408, response
        finally:
            await _shutdown(server, task)

    _run(scenario())


if __name__ == "__main__":
    # Allow running this file directly: `python tests/test_ipc_auth.py`.
    import traceback

    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok] {name}")
            except Exception:
                failures += 1
                print(f"[fail] {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
