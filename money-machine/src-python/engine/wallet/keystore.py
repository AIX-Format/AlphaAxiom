"""
Encrypted at-rest storage for blockchain private keys.

Threat model:

- Adversary with read access to the keystore directory: must not
  be able to recover any private key without the operator password.
- Adversary intercepting a single keystore file: same.
- Operator password compromise: limited to what the password
  unlocks (rotating the password re-derives the keys; rotating the
  underlying private key requires reissuing).
- In-memory leak: decrypted bytes live only for the duration of the
  call site that requested them. Callers must clear copies. We do
  NOT pin pages or mlock; that is out of scope and would lock us
  into platform-specific syscalls.

Cryptography:

- Key derivation: scrypt with N=2^15 (32768), r=8, p=1. These are
  the EIP-2335 recommended parameters and resist commodity GPU
  cracking by orders of magnitude over PBKDF2 at the same time
  budget. Each entry gets a fresh 32-byte salt.
- Authenticated encryption: AES-256-GCM with a fresh 96-bit nonce
  per encryption. The associated data covers the version and the
  entry id so a stored ciphertext cannot be silently moved to a
  different entry slot.

File format (one JSON file per entry under keystore_dir):

    {
      "version": 1,
      "id": "<uuid>",
      "label": "<human label>",
      "chain": "ethereum" | "solana" | ...,
      "metadata": {...},                  # free-form, NOT encrypted
      "kdf": {
        "name": "scrypt",
        "params": {"n": 32768, "r": 8, "p": 1, "dklen": 32,
                   "salt": "<hex>"}
      },
      "cipher": {
        "name": "aes-256-gcm",
        "params": {"nonce": "<hex>"},
        "ciphertext": "<hex>"
      }
    }

The cleartext payload that gets encrypted is the raw private-key
bytes plus a checksum, so a wrong password aborts at decryption
time rather than silently returning garbage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KeystoreError(Exception):
    """Base class for keystore failures (missing entry, malformed file)."""


class WrongPasswordError(KeystoreError):
    """The supplied password failed authentication on decryption."""


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class KeystoreEntry:
    """In-memory representation of one keystore record."""

    id: str
    label: str
    chain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """Public summary safe to expose to UI / logs (no key material)."""
        return {
            "id": self.id,
            "label": self.label,
            "chain": self.chain,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Keystore
# ---------------------------------------------------------------------------


_KDF_N = 32768
_KDF_R = 8
_KDF_P = 1
_KEY_LENGTH = 32
_NONCE_LENGTH = 12
_SALT_LENGTH = 32
_VERSION = 1
_FORMAT_PREFIX = b"AAKS"  # AlphaAxiom KeyStore magic for checksum scope


class Keystore:
    """Directory-backed encrypted private-key store."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Lock down the directory: owner-only on POSIX.
        try:
            os.chmod(self.directory, 0o700)
        except (PermissionError, NotImplementedError):
            # Read-only filesystem or Windows: skip silently.
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_key(
        self,
        *,
        private_key: bytes,
        password: str,
        label: str,
        chain: str,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> KeystoreEntry:
        """Encrypt and persist a private key. Returns the new entry."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover
            raise KeystoreError(
                "Keystore requires the `cryptography` package"
            ) from exc

        if not isinstance(private_key, (bytes, bytearray)) or not private_key:
            raise KeystoreError("private_key must be non-empty bytes")
        # Normalise to immutable bytes so the later
        # _frame_plaintext call (which does b"" + private_key)
        # works for both bytes and bytearray inputs without
        # raising TypeError.
        private_key = bytes(private_key)
        if not password:
            raise KeystoreError("password must be non-empty")
        if not chain:
            raise KeystoreError("chain must be non-empty")

        salt = secrets.token_bytes(_SALT_LENGTH)
        derived = _scrypt(password.encode("utf-8"), salt)
        nonce = secrets.token_bytes(_NONCE_LENGTH)
        eid = entry_id or str(uuid.uuid4())

        # Wrap the key bytes in a small framed payload so a wrong
        # password aborts at decryption time with the GCM tag check,
        # AND so a successful decrypt that yields a corrupted key
        # length is caught explicitly.
        plaintext = _frame_plaintext(private_key)
        aad = _associated_data(eid)
        ciphertext = AESGCM(derived).encrypt(nonce, plaintext, aad)

        envelope: Dict[str, Any] = {
            "version": _VERSION,
            "id": eid,
            "label": label,
            "chain": chain,
            "metadata": dict(metadata or {}),
            "kdf": {
                "name": "scrypt",
                "params": {
                    "n": _KDF_N,
                    "r": _KDF_R,
                    "p": _KDF_P,
                    "dklen": _KEY_LENGTH,
                    "salt": salt.hex(),
                },
            },
            "cipher": {
                "name": "aes-256-gcm",
                "params": {"nonce": nonce.hex()},
                "ciphertext": ciphertext.hex(),
            },
        }
        path = self._path_for(eid)
        self._atomic_write_json(path, envelope)
        try:
            os.chmod(path, 0o600)
        except (PermissionError, NotImplementedError):
            pass
        return KeystoreEntry(
            id=eid,
            label=label,
            chain=chain,
            metadata=dict(metadata or {}),
        )

    def load_key(self, entry_id: str, password: str) -> bytes:
        """Decrypt and return the raw private-key bytes for an entry."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.exceptions import InvalidTag
        except ImportError as exc:  # pragma: no cover
            raise KeystoreError(
                "Keystore requires the `cryptography` package"
            ) from exc

        envelope = self._read_envelope(entry_id)
        salt = bytes.fromhex(envelope["kdf"]["params"]["salt"])
        nonce = bytes.fromhex(envelope["cipher"]["params"]["nonce"])
        ciphertext = bytes.fromhex(envelope["cipher"]["ciphertext"])

        derived = _scrypt(password.encode("utf-8"), salt)
        aad = _associated_data(entry_id)
        try:
            framed = AESGCM(derived).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise WrongPasswordError("invalid password or corrupted entry") from exc

        try:
            return _unframe_plaintext(framed)
        except ValueError as exc:
            raise KeystoreError(f"corrupted keystore payload: {exc}") from exc

    def list_entries(self) -> List[KeystoreEntry]:
        """Return summaries of every entry. Reads files but does NOT
        decrypt; safe to expose to UI.

        A *.json file that is syntactically valid but is not a JSON
        object (e.g. an array left by a manual edit, a partial
        write, or a corrupted envelope) is skipped silently with a
        log line. Without this guard one malformed file would raise
        AttributeError inside `envelope.get(...)` and the whole
        listing would abort, hiding healthy entries from operators.
        """
        entries: List[KeystoreEntry] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    envelope = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(envelope, dict):
                logger.warning(
                    "Skipping non-object keystore file %s (top-level "
                    "is %s, not a dict)",
                    path, type(envelope).__name__,
                )
                continue
            entries.append(
                KeystoreEntry(
                    id=envelope.get("id", path.stem),
                    label=envelope.get("label", ""),
                    chain=envelope.get("chain", ""),
                    metadata=envelope.get("metadata", {}),
                )
            )
        return entries

    def delete_entry(self, entry_id: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        path = self._path_for(entry_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def rotate_password(
        self, entry_id: str, old_password: str, new_password: str
    ) -> KeystoreEntry:
        """Re-encrypt an entry under a new password without changing
        the underlying private key.

        Implemented as decrypt-with-old + encrypt-with-new; if the
        new write fails the on-disk file is left untouched because
        the atomic rename has not happened yet.
        """
        envelope = self._read_envelope(entry_id)
        plaintext = self.load_key(entry_id, old_password)
        try:
            return self.store_key(
                private_key=plaintext,
                password=new_password,
                label=envelope.get("label", ""),
                chain=envelope.get("chain", ""),
                metadata=envelope.get("metadata", {}),
                entry_id=entry_id,
            )
        finally:
            # Best-effort zeroing. CPython does not let us clear a
            # bytes object in place, so we just drop the reference
            # and let GC handle it. mlock / sodium_memzero would
            # require a C extension out of scope here.
            del plaintext

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, entry_id: str) -> Path:
        # Reject any disallowed character outright instead of
        # silently stripping it. Otherwise two distinct ids ("ab/c"
        # and "abc") could collapse to the same filename and
        # store_key would silently overwrite the wrong entry.
        if not isinstance(entry_id, str) or not entry_id:
            raise KeystoreError(f"invalid entry_id {entry_id!r}")
        for ch in entry_id:
            if not (ch.isalnum() or ch in {"-", "_"}):
                raise KeystoreError(
                    f"invalid entry_id {entry_id!r}: contains {ch!r}; "
                    f"only alphanumerics, '-' and '_' are allowed"
                )
        return self.directory / f"{entry_id}.json"

    def _read_envelope(self, entry_id: str) -> Dict[str, Any]:
        path = self._path_for(entry_id)
        if not path.exists():
            raise KeystoreError(f"keystore entry not found: {entry_id}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
        except (OSError, ValueError) as exc:
            raise KeystoreError(f"failed to read entry {entry_id}: {exc}") from exc
        if envelope.get("version") != _VERSION:
            raise KeystoreError(
                f"unsupported keystore version: {envelope.get('version')}"
            )
        for key in ("kdf", "cipher", "id"):
            if key not in envelope:
                raise KeystoreError(f"malformed keystore entry, missing {key!r}")
        return envelope

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent)
        )
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Crypto helpers
# ---------------------------------------------------------------------------


def _scrypt(password: bytes, salt: bytes) -> bytes:
    """scrypt KDF with EIP-2335 default parameters.

    Uses `cryptography.hazmat.primitives.kdf.scrypt.Scrypt` rather
    than the stdlib `hashlib.scrypt` because the latter inherits
    OpenSSL's 32 MiB default memory cap, which clips right at the
    edge of our N=32768 / r=8 / p=1 working set on some platforms.
    The `cryptography` implementation has no such cap.
    """
    try:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:  # pragma: no cover
        raise KeystoreError(
            "Keystore requires the `cryptography` package"
        ) from exc
    kdf = Scrypt(
        salt=salt, length=_KEY_LENGTH, n=_KDF_N, r=_KDF_R, p=_KDF_P
    )
    return kdf.derive(password)


def _associated_data(entry_id: str) -> bytes:
    """AAD binds a ciphertext to (version, entry_id) so a ciphertext
    cannot be silently moved between entries.
    """
    return f"{_VERSION}|{entry_id}".encode("utf-8")


def _frame_plaintext(private_key: bytes) -> bytes:
    """Wrap raw key bytes with a magic + HMAC checksum.

    The GCM tag already gives us authenticated decryption, but the
    extra checksum catches the case where a future bug ships a
    truncated decrypt path or a wrong-key match against the same
    nonce (vanishingly unlikely but cheap to defend against).
    """
    body = _FORMAT_PREFIX + len(private_key).to_bytes(2, "big") + private_key
    digest = hmac.new(_FORMAT_PREFIX, body, hashlib.sha256).digest()[:16]
    return body + digest


def _unframe_plaintext(framed: bytes) -> bytes:
    if len(framed) < len(_FORMAT_PREFIX) + 2 + 16:
        raise ValueError("framed payload too short")
    if framed[: len(_FORMAT_PREFIX)] != _FORMAT_PREFIX:
        raise ValueError("framed payload has wrong magic")
    key_len = int.from_bytes(
        framed[len(_FORMAT_PREFIX) : len(_FORMAT_PREFIX) + 2], "big"
    )
    body = framed[: len(_FORMAT_PREFIX) + 2 + key_len]
    expected_digest = framed[len(_FORMAT_PREFIX) + 2 + key_len :]
    digest = hmac.new(_FORMAT_PREFIX, body, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError("framed payload checksum mismatch")
    return body[len(_FORMAT_PREFIX) + 2 : len(_FORMAT_PREFIX) + 2 + key_len]
