"""
TCP-based IPC server for Rust <-> Python communication.

The wire protocol expects two lines per request:

    X-Auth-Token: <hex_token>\\n
    {"command": "...", "payload": {...}}\\n

Requests without a valid token are answered with `{"error": "...",
"code": 401}` and the connection is closed. Once authenticated, the
server enforces a global token-bucket rate limit before dispatching
to the application command handler.

Bound to 127.0.0.1 only; remote clients cannot reach this socket.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from .ipc_auth import (
    TokenBucket,
    is_valid_token,
    parse_auth_header,
    resolve_token,
)

logger = logging.getLogger(__name__)


CommandHandler = Callable[[str, dict], Awaitable[dict]]


class IPCServer:
    """TCP server for inter-process communication with Tauri/Rust backend."""

    # Rate limit: 100 sustained requests/second, 200 burst. Tunable via
    # constructor args when the engine needs more headroom in tests.
    DEFAULT_RATE = 100.0
    DEFAULT_BURST = 200.0
    MAX_AUTH_LINE_BYTES = 256
    MAX_BODY_BYTES = 64 * 1024
    DEFAULT_READ_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        command_handler: CommandHandler,
        host: str = "127.0.0.1",
        port: int = 19284,
        auth_token: Optional[str] = None,
        rate_limit: Optional[float] = None,
        burst_limit: Optional[float] = None,
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
    ):
        """
        Initialize the IPC server with network, authentication, rate-limiting, and read timeout configuration.
        
        Parameters:
            command_handler (CommandHandler): Async callable invoked as (command, payload) to handle requests.
            host (str): IP address to bind the server to.
            port (int): TCP port to listen on.
            auth_token (Optional[str]): Hex authentication token to require from clients; if None a default token is resolved.
            rate_limit (Optional[float]): Tokens-per-second rate for the global token bucket; uses DEFAULT_RATE when None.
            burst_limit (Optional[float]): Capacity (burst size) for the token bucket; uses DEFAULT_BURST when None.
            read_timeout_seconds (float): Per-read timeout in seconds applied when reading the auth header and JSON body.
        """
        self.host = host
        self.port = port
        self.command_handler = command_handler
        self.server: Optional[asyncio.AbstractServer] = None
        self.auth_token = auth_token or resolve_token()
        self._bucket = TokenBucket(
            rate=rate_limit if rate_limit is not None else self.DEFAULT_RATE,
            capacity=burst_limit if burst_limit is not None else self.DEFAULT_BURST,
        )
        self.read_timeout_seconds = read_timeout_seconds

    async def start(self) -> None:
        """Start the TCP server and run until cancelled."""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        addr = self.server.sockets[0].getsockname()
        logger.info("IPC Server listening on %s:%s (auth enforced)", addr[0], addr[1])

        async with self.server:
            await self.server.serve_forever()

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one incoming client connection (one request per connection)."""
        addr = writer.get_extra_info("peername")
        logger.debug("New connection from %s", addr)

        try:
            response = await self._process_request(reader)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("IPC handler error: %s", exc)
            response = {"error": str(exc), "code": 500}

        try:
            writer.write((json.dumps(response) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception as exc:  # pragma: no cover - peer disconnected
            logger.debug("Failed to send IPC response: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover
                pass
            logger.debug("Connection closed from %s", addr)

    async def _process_request(self, reader: asyncio.StreamReader) -> dict:
        """
        Process a single IPC request from the provided StreamReader using the two-line protocol: an auth header line followed by a JSON body line.
        
        Reads and validates the auth header, applies the global token-bucket rate limit after successful authentication, reads and parses the JSON request body, ensures a non-empty "command" field is present, and dispatches to the configured command handler. On protocol, authentication, rate-limit, or JSON-parsing errors, returns an error dictionary with keys "error" and "code".
        
        Parameters:
            reader (asyncio.StreamReader): Stream reader to read the incoming request lines from.
        
        Returns:
            dict: The response produced by the command handler, or an error dictionary containing "error" and "code".
        """
        auth_line_bytes, auth_error = await self._read_limited_line(
            reader, self.MAX_AUTH_LINE_BYTES, "auth header"
        )
        if auth_error:
            return auth_error

        if not auth_line_bytes:
            return {"error": "Empty request", "code": 400}

        auth_line = auth_line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
        provided_token = parse_auth_header(auth_line)
        if not is_valid_token(provided_token, self.auth_token):
            return {"error": "Invalid or missing IPC auth token", "code": 401}

        # Rate limit gate, applied only after auth so attackers cannot
        # exhaust the bucket without knowing the secret.
        if not self._bucket.consume():
            return {"error": "Rate limit exceeded", "code": 429}

        body_bytes, body_error = await self._read_limited_line(
            reader, self.MAX_BODY_BYTES, "request body"
        )
        if body_error:
            return body_error

        if not body_bytes:
            return {"error": "Missing request body after auth header", "code": 400}

        body_str = body_bytes.decode("utf-8", errors="replace").strip()
        try:
            request = json.loads(body_str)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON body: {exc}", "code": 400}

        command = request.get("command", "")
        payload = request.get("payload", {})
        if not command:
            return {"error": "Missing 'command' field", "code": 400}

        return await self.command_handler(command, payload)

    async def _read_limited_line(
        self,
        reader: asyncio.StreamReader,
        limit: int,
        label: str,
    ) -> tuple[bytes, Optional[dict]]:
        """
        Read a single newline-terminated line from `reader`, enforcing a per-read timeout and a maximum byte limit.
        
        Parameters:
            reader (asyncio.StreamReader): Stream reader to read the line from.
            limit (int): Maximum allowed bytes for the line (inclusive). Lines longer than this produce an error.
            label (str): Human-readable label used in error messages (e.g., "auth header" or "request body").
        
        Returns:
            tuple[bytes, Optional[dict]]: A pair (line, error). On success `line` contains the bytes read (including the trailing newline) and `error` is `None`. On failure `line` is `b""` and `error` is a dict with keys `error` (message) and `code` (HTTP-like status code):
                - Timeout reading the line → code 408.
                - Line exceeds `limit` → code 413.
                - Other read failures → code 400.
        
        Notes:
            If the underlying read raises `asyncio.IncompleteReadError`, the function returns the partial bytes read as `line` (and no error) so the caller can decide how to handle an unterminated stream.
        """
        try:
            line = await asyncio.wait_for(
                reader.readuntil(b"\n"),
                timeout=self.read_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return b"", {"error": f"Timed out reading IPC {label}", "code": 408}
        except asyncio.LimitOverrunError:
            return b"", {"error": f"IPC {label} exceeds {limit} bytes", "code": 413}
        except asyncio.IncompleteReadError as exc:
            line = exc.partial
        except Exception as exc:
            return b"", {"error": f"Failed to read {label}: {exc}", "code": 400}

        if len(line) > limit:
            return b"", {"error": f"IPC {label} exceeds {limit} bytes", "code": 413}
        return line, None

    async def stop(self) -> None:
        """
        Close the running asyncio server and wait for it to finish closing.
        
        If the server was not started, this is a no-op.
        """
        if self.server:
            self.server.close()
            await self.server.wait_closed()
