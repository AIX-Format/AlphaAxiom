"""
Tests for the encrypted wallet keystore.

The keystore is the trust anchor every on-chain adapter depends on,
so it gets a wide coverage net: round-trip, wrong password, tampered
ciphertext, tampered envelope id, password rotation, listing entries
without decrypting, deletion, malformed file handling, and the
plaintext framing's checksum.

All tests use a `tmp_path` directory so they leave no global state
behind, and `cryptography` is `pytest.importorskip`-ed so the whole
module is skipped cleanly on systems that lack it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_PYTHON = Path(__file__).resolve().parent.parent
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

pytest.importorskip("cryptography")

from engine.wallet import (  # noqa: E402
    Keystore,
    KeystoreEntry,
    KeystoreError,
    WrongPasswordError,
)


PRIVATE_KEY_32 = bytes.fromhex(
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
PRIVATE_KEY_64 = b"S" * 64


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_store_and_load_round_trip(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password="correct horse battery staple",
        label="anvil-test",
        chain="ethereum",
        metadata={"derivation": "m/44'/60'/0'/0/0"},
    )
    assert isinstance(entry, KeystoreEntry)

    recovered = ks.load_key(entry.id, "correct horse battery staple")
    assert recovered == PRIVATE_KEY_32


def test_round_trip_with_solana_64_byte_key(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_64,
        password="hunter2",
        label="solana-test",
        chain="solana",
    )
    recovered = ks.load_key(entry.id, "hunter2")
    assert recovered == PRIVATE_KEY_64


def test_round_trip_for_unicode_password(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    pwd = "كلمة-سر-قوية-😀"
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password=pwd,
        label="unicode",
        chain="ethereum",
    )
    assert ks.load_key(entry.id, pwd) == PRIVATE_KEY_32


# ---------------------------------------------------------------------------
# Wrong password / tampering
# ---------------------------------------------------------------------------


def test_wrong_password_raises_wrong_password_error(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password="real-password",
        label="t",
        chain="ethereum",
    )
    with pytest.raises(WrongPasswordError):
        ks.load_key(entry.id, "wrong-password")


def test_tampered_ciphertext_fails_authentication(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password="pw",
        label="t",
        chain="ethereum",
    )
    path = tmp_path / f"{entry.id}.json"
    envelope = json.loads(path.read_text())
    # Flip a hex digit in the ciphertext.
    ct = envelope["cipher"]["ciphertext"]
    envelope["cipher"]["ciphertext"] = ("0" if ct[0] != "0" else "1") + ct[1:]
    path.write_text(json.dumps(envelope))

    with pytest.raises(WrongPasswordError):
        ks.load_key(entry.id, "pw")


def test_tampered_envelope_id_fails_authentication(tmp_path: Path) -> None:
    """The AAD binds ciphertext to (version, id); renaming the id
    after the fact must invalidate decryption.
    """
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password="pw",
        label="t",
        chain="ethereum",
    )
    path = tmp_path / f"{entry.id}.json"
    envelope = json.loads(path.read_text())
    # Change the id field but not the filename: load_key looks up by
    # filename / entry_id arg, so we get this exact ciphertext but
    # the AAD has a different id binding. Actually our load_key
    # uses the entry_id argument as the AAD; tampering only the
    # in-file id is silent to load_key.
    # So instead simulate moving the ciphertext to a NEW filename
    # and asking for that id: should fail authentication.
    new_id = "moved-entry-12345"
    new_envelope = dict(envelope)
    new_envelope["id"] = new_id
    (tmp_path / f"{new_id}.json").write_text(json.dumps(new_envelope))

    with pytest.raises(WrongPasswordError):
        ks.load_key(new_id, "pw")


def test_corrupted_framed_payload_raises_keystore_error(tmp_path: Path) -> None:
    """If a ciphertext somehow decrypts (wrong password collision is
    cryptographically impossible, but corrupted bytes that happen to
    pass the GCM tag are too) but the framed checksum mismatches,
    we raise KeystoreError, not WrongPasswordError.

    To exercise this without a real collision we monkey-patch the
    AESGCM decrypt to return a hand-rolled bad payload.
    """
    import engine.wallet.keystore as ks_mod

    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32,
        password="pw",
        label="t",
        chain="ethereum",
    )

    original = ks_mod._unframe_plaintext

    def _bad_unframe(_data: bytes) -> bytes:
        raise ValueError("framed payload checksum mismatch")

    ks_mod._unframe_plaintext = _bad_unframe  # type: ignore[assignment]
    try:
        with pytest.raises(KeystoreError):
            ks.load_key(entry.id, "pw")
    finally:
        ks_mod._unframe_plaintext = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Listing / deletion
# ---------------------------------------------------------------------------


def test_list_entries_is_safe_without_password(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    ks.store_key(
        private_key=PRIVATE_KEY_32, password="a",
        label="alpha", chain="ethereum",
    )
    ks.store_key(
        private_key=PRIVATE_KEY_64, password="b",
        label="beta", chain="solana",
    )
    entries = ks.list_entries()
    labels = sorted(e.label for e in entries)
    chains = sorted(e.chain for e in entries)
    assert labels == ["alpha", "beta"]
    assert chains == ["ethereum", "solana"]
    # Summary helper exposes nothing sensitive.
    for entry in entries:
        summary = entry.to_summary()
        assert "key" not in summary
        assert "private" not in summary
        assert "password" not in summary


def test_delete_entry_removes_file(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32, password="pw",
        label="t", chain="ethereum",
    )
    assert (tmp_path / f"{entry.id}.json").exists()
    assert ks.delete_entry(entry.id) is True
    assert not (tmp_path / f"{entry.id}.json").exists()
    # Second delete is a no-op that returns False.
    assert ks.delete_entry(entry.id) is False


# ---------------------------------------------------------------------------
# Password rotation
# ---------------------------------------------------------------------------


def test_rotate_password_preserves_key_under_new_password(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32, password="old-pw",
        label="rotate-me", chain="ethereum",
        metadata={"chain_id": 1},
    )
    ks.rotate_password(entry.id, "old-pw", "new-pw-shiny")

    # Old password no longer works.
    with pytest.raises(WrongPasswordError):
        ks.load_key(entry.id, "old-pw")

    # New password recovers the same key bytes and metadata.
    assert ks.load_key(entry.id, "new-pw-shiny") == PRIVATE_KEY_32
    entries = ks.list_entries()
    assert entries[0].metadata == {"chain_id": 1}
    assert entries[0].label == "rotate-me"


def test_rotate_password_rejects_wrong_old_password(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    entry = ks.store_key(
        private_key=PRIVATE_KEY_32, password="old",
        label="t", chain="ethereum",
    )
    with pytest.raises(WrongPasswordError):
        ks.rotate_password(entry.id, "wrong-old", "new")
    # Original password still works.
    assert ks.load_key(entry.id, "old") == PRIVATE_KEY_32


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"private_key": b""},
        {"password": ""},
        {"chain": ""},
    ],
)
def test_store_rejects_empty_inputs(tmp_path: Path, kwargs) -> None:
    ks = Keystore(tmp_path)
    base = dict(
        private_key=PRIVATE_KEY_32,
        password="pw",
        label="t",
        chain="ethereum",
    )
    base.update(kwargs)
    with pytest.raises(KeystoreError):
        ks.store_key(**base)


def test_load_unknown_entry_raises_keystore_error(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    with pytest.raises(KeystoreError):
        ks.load_key("does-not-exist", "pw")


def test_malformed_envelope_raises(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    bad = tmp_path / "broken.json"
    bad.write_text("{}")
    with pytest.raises(KeystoreError):
        ks.load_key("broken", "pw")


def test_envelope_with_unknown_version_raises(tmp_path: Path) -> None:
    ks = Keystore(tmp_path)
    bad = tmp_path / "future.json"
    bad.write_text(json.dumps({
        "version": 99,
        "id": "future",
        "kdf": {},
        "cipher": {},
    }))
    with pytest.raises(KeystoreError):
        ks.load_key("future", "pw")


# ---------------------------------------------------------------------------
# Plaintext framing checksum (white-box)
# ---------------------------------------------------------------------------


def test_frame_and_unframe_round_trip() -> None:
    import engine.wallet.keystore as ks_mod
    framed = ks_mod._frame_plaintext(PRIVATE_KEY_32)
    assert ks_mod._unframe_plaintext(framed) == PRIVATE_KEY_32


def test_unframe_rejects_truncated_payload() -> None:
    import engine.wallet.keystore as ks_mod
    with pytest.raises(ValueError):
        ks_mod._unframe_plaintext(b"too-short")


def test_unframe_rejects_bad_magic() -> None:
    import engine.wallet.keystore as ks_mod
    framed = ks_mod._frame_plaintext(PRIVATE_KEY_32)
    tampered = b"XXXX" + framed[4:]
    with pytest.raises(ValueError):
        ks_mod._unframe_plaintext(tampered)


def test_unframe_rejects_bad_checksum() -> None:
    import engine.wallet.keystore as ks_mod
    framed = bytearray(ks_mod._frame_plaintext(PRIVATE_KEY_32))
    framed[-1] ^= 0xFF  # flip a bit in the checksum
    with pytest.raises(ValueError):
        ks_mod._unframe_plaintext(bytes(framed))
