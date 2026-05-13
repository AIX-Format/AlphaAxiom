"""
IPC authentication and rate limiting for the Rust <-> Python TCP channel.

The TCP IPC server binds to 127.0.0.1, but any local process can still
open a connection and send commands. The auth layer mitigates that by
requiring every request to carry a shared secret that only legitimate
clients (the Tauri Rust shell, the test harness) know.

Token resolution precedence at startup:

1. The IPC_AUTH_TOKEN environment variable. The Tauri shell sets this
   on the sidecar process so the secret never touches disk in dev.
2. The OS keychain, under service "money-machine" account
   "ipc-auth-token", via the `keyring` package. The Tauri shell
   provisions this entry on first launch.
3. If neither exists, the server generates a fresh 32-byte token, logs
   a warning, and stores it in the keychain when possible. This keeps
   stand-alone dev runs working without a manual setup step.

The wire protocol is unchanged at the JSON layer; auth is carried in a
single line that precedes the JSON request:

    X-Auth-Token: <hex_token>\\n
    {"command": "...", "payload": {...}}\\n

Requests without a valid header are rejected with HTTP-style status
codes embedded in the JSON response (401, 429) so clients can branch
on a stable field.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "money-machine"
KEYCHAIN_ACCOUNT = "ipc-auth-token"
ENV_VAR = "IPC_AUTH_TOKEN"


def _load_from_keychain() -> Optional[str]:
    """Return the IPC token from the OS keychain, or None if unavailable."""
    try:
        import keyring  # type: ignore
    except ImportError:
        return None

    try:
        return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
    except Exception as exc:  # pragma: no cover - keychain backends vary
        logger.debug("Keychain lookup failed: %s", exc)
        return None


def _store_in_keychain(token: str) -> bool:
    """Best-effort persist the token to the OS keychain. Returns success."""
    try:
        import keyring  # type: ignore
    except ImportError:
        return False

    try:
        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, token)
        return True
    except Exception as exc:  # pragma: no cover - keychain backends vary
        logger.debug("Keychain store failed: %s", exc)
        return False


def resolve_token() -> str:
    """Resolve the active IPC token following the documented precedence.

    Always returns a non-empty hex string. Falls back to generating and
    persisting a new token if no source has one.
    """
    env_token = os.environ.get(ENV_VAR, "").strip()
    if env_token:
        return env_token

    stored = _load_from_keychain()
    if stored:
        return stored

    new_token = secrets.token_hex(32)
    persisted = _store_in_keychain(new_token)
    if persisted:
        logger.warning(
            "Generated a new IPC auth token and stored it in the OS keychain "
            "(service=%s, account=%s). The Tauri shell should now read it "
            "from there.",
            KEYCHAIN_SERVICE,
            KEYCHAIN_ACCOUNT,
        )
    else:
        logger.warning(
            "Generated a new IPC auth token but could not persist it. The "
            "token will only live in this process. Set %s in the environment "
            "to avoid this warning.",
            ENV_VAR,
        )
    return new_token


# ---------------------------------------------------------------------------
# Auth header parsing
# ---------------------------------------------------------------------------


AUTH_HEADER_PREFIX = "X-Auth-Token:"


def parse_auth_header(line: str) -> Optional[str]:
    """Extract the token from an `X-Auth-Token: <token>` line.

    Returns None when the line is not a well-formed auth header.
    Whitespace around the value is stripped.
    """
    if not line.startswith(AUTH_HEADER_PREFIX):
        return None
    value = line[len(AUTH_HEADER_PREFIX):].strip()
    return value or None


def is_valid_token(provided: Optional[str], expected: str) -> bool:
    """Constant-time comparison of an incoming token against the expected one."""
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided, expected)


# ---------------------------------------------------------------------------
# Rate limiting (token bucket)
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Classic token bucket rate limiter.

    `rate` is the steady-state allowed rate in tokens per second.
    `capacity` is the maximum burst size. Each `consume()` call removes
    one token if available and returns True; otherwise returns False.
    """

    rate: float
    capacity: float
    _tokens: float = 0.0
    _last_refill: float = 0.0

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def consume(self, amount: float = 1.0, now: Optional[float] = None) -> bool:
        """Try to remove `amount` tokens. Returns True on success."""
        ts = now if now is not None else time.monotonic()
        elapsed = max(0.0, ts - self._last_refill)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = ts

        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False
