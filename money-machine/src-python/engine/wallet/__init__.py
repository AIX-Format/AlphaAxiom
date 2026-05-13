"""
Local encrypted wallet storage.

`Keystore` holds blockchain private keys encrypted at rest on the
operator's machine. The Ed25519 (Solana) and secp256k1 (EVM) keys
that on-chain adapters need to sign transactions are stored in
files under `keystore_dir/` and only ever decrypted with the
operator-supplied password.

The format is deliberately compatible with the spirit of the EIP-2335
keystore JSON: a small, well-documented envelope that includes the
KDF (scrypt) and cipher (AES-256-GCM) parameters so future password
rotations or KDF upgrades can be done in-place. The implementation
is custom rather than depending on `eth-keyfile` because we want a
single primitive that handles both EVM and Solana keys and stores
arbitrary metadata (chain id, key derivation index, label).
"""

from .keystore import (
    Keystore,
    KeystoreEntry,
    KeystoreError,
    WrongPasswordError,
)

__all__ = [
    "Keystore",
    "KeystoreEntry",
    "KeystoreError",
    "WrongPasswordError",
]
