"""
Manual smoke test for Money Machine IPC communication.

Run with: python test_ipc.py

Assumes the Python engine is running on 127.0.0.1:19284 and that the
IPC_AUTH_TOKEN environment variable (or the OS keychain entry) matches
on both sides. Use the same env var here as the engine reads at boot.
"""

import asyncio
import json
import os
import socket  # noqa: F401  (kept for backward compat with prior script users)

from utils.ipc_auth import resolve_token


async def _send(host: str, port: int, token: str, command: str) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request = json.dumps({"command": command, "payload": {}})
        wire = f"X-Auth-Token: {token}\n{request}\n"
        writer.write(wire.encode("utf-8"))
        await writer.drain()

        response = await reader.readline()
        return json.loads(response.decode())
    finally:
        writer.close()
        await writer.wait_closed()


async def test_ipc_connection() -> bool:
    """Test IPC connection to Python trading engine."""
    print("Testing IPC Connection to Money Machine Engine...")
    print("-" * 50)

    host = "127.0.0.1"
    port = int(os.environ.get("TAURI_PORT", 19284))
    token = resolve_token()

    for command in ("PING", "GET_STATUS", "GET_PORTFOLIO"):
        try:
            result = await _send(host, port, token, command)
            print(f"[ok] {command} -> {result}")
        except ConnectionRefusedError:
            print("[fail] Connection refused. Is the Python engine running?")
            print("       Start it with: python src-python/main.py")
            return False
        except Exception as exc:
            print(f"[fail] {command} -> {exc}")
            return False

    print("-" * 50)
    print("All IPC smoke checks passed.")
    return True


if __name__ == "__main__":
    asyncio.run(test_ipc_connection())
