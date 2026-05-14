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
    async def scenario() -> None:
        server, task, (host, port) = await _start_server(read_timeout_seconds=0.05)
        try:
            response = await _send_raw(host, port, f"X-Auth-Token: {TEST_TOKEN}\n".encode("utf-8"))
            assert response.get("code") == 408, response
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_oversized_auth_header_returns_413() -> None:
    """An auth header that exceeds MAX_AUTH_LINE_BYTES should be rejected with 413."""
    async def scenario() -> None:
        server, task, (host, port) = await _start_server()
        try:
            # Build an auth line longer than the server's MAX_AUTH_LINE_BYTES limit.
            oversized_token = "x" * (IPCServer.MAX_AUTH_LINE_BYTES + 1)
            oversized_auth = f"X-Auth-Token: {oversized_token}\n".encode("utf-8")
            response = await _send_raw(host, port, oversized_auth)
            assert response.get("code") == 413, response
        finally:
            await _shutdown(server, task)

    _run(scenario())


def test_custom_read_timeout_is_applied_to_server() -> None:
    """IPCServer should store the custom read_timeout_seconds value."""
    server = IPCServer(
        command_handler=_echo_handler,
        host="127.0.0.1",
        port=0,
        auth_token=TEST_TOKEN,
        read_timeout_seconds=2.5,
    )
    assert server.read_timeout_seconds == 2.5


def test_default_read_timeout_equals_class_constant() -> None:
    """When no timeout is given, the instance should use DEFAULT_READ_TIMEOUT_SECONDS."""
    server = IPCServer(
        command_handler=_echo_handler,
        host="127.0.0.1",
        port=0,
        auth_token=TEST_TOKEN,
    )
    assert server.read_timeout_seconds == IPCServer.DEFAULT_READ_TIMEOUT_SECONDS


def test_max_body_bytes_class_constant() -> None:
    assert IPCServer.MAX_BODY_BYTES == 64 * 1024


def test_max_auth_line_bytes_class_constant() -> None:
    assert IPCServer.MAX_AUTH_LINE_BYTES == 256


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
