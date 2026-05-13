"""
Standalone research and analysis scripts that consume the engine.

These tools are intentionally separate from `main.py` (the IPC server
for the Tauri frontend) so they can be invoked from the command line
without touching the live trading surface.

Nothing in this package touches `engine/adapters/` or any live venue;
it is read-only orchestration over the existing primitives.
"""
