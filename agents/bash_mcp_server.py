#!/usr/bin/env python3
"""
bash_mcp_server.py — MCP Server: exposes run_bash as an HTTP tool

Starts an MCP server over streamable-HTTP on:
    http://127.0.0.1:8765/mcp

Exposes one tool:
    bash(command: str) -> str

The run_bash() implementation contains the same 8 safety / quality
guards as the original inline version in s01_agent_loop.py:
    1. Dangerous command blocklist
    2. Streaming / non-terminating command blocklist
    3. Binary / non-UTF-8 output detection
    4. Separate stdout / stderr labelling
    5. Empty output signal
    6. Non-zero exit code prefix
    7. Output truncation with marker
    8. Timeout (30 s)

Usage
-----
Terminal 1 — start the server:
    python3 bash_mcp_server.py

Terminal 2 — run the MCP-aware agent:
    python3 s01_agent_loop_v4.py

The server stays running and handles tool calls from any MCP client.
"""

import os
import subprocess

from mcp.server.fastmcp import FastMCP

# ── Server setup ───────────────────────────────────────────────────────────────
# host / port are passed here; mount path defaults to /mcp (streamable-http).
mcp = FastMCP(
    "bash-server",
    host="127.0.0.1",
    port=8765,
)


# ── Core bash executor (same 8-case logic as s01_agent_loop.py) ───────────────

def run_bash(command: str) -> str:
    # ── Edge case 1: Dangerous command blocklist ────────────────────────────
    DANGEROUS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in DANGEROUS):
        return "Error: Dangerous command blocked"

    # ── Edge case 2: Streaming / non-terminating commands ──────────────────
    STREAMING = ["tail -f", "watch ", "ping ", "top", "htop",
                 "while true", "sleep inf"]
    for s in STREAMING:
        if s in command:
            return (f"Error: Streaming command blocked ('{s}' runs indefinitely). "
                    "Use a time-limited alternative, e.g. 'timeout 5 ping ...' "
                    "or 'head -n 20 file' instead of 'tail -f file'.")

    try:
        r = subprocess.run(
            command, shell=True, cwd=os.getcwd(),
            capture_output=True, timeout=30,
        )

        # ── Edge case 3: Binary / non-UTF-8 output ─────────────────────────
        try:
            stdout = r.stdout.decode("utf-8")
            stderr = r.stderr.decode("utf-8")
        except UnicodeDecodeError:
            return (f"[exit {r.returncode}] "
                    "[binary/non-UTF-8 output — not displayable as text. "
                    "Use xxd, file, or strings if you need to inspect it.]")

        # ── Edge case 4: Separate stdout and stderr ─────────────────────────
        parts = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            stderr_lines = "\n".join(
                f"[stderr] {line}" for line in stderr.strip().splitlines()
            )
            parts.append(stderr_lines)

        out = "\n".join(parts).strip()

        # ── Edge case 5: Empty output ───────────────────────────────────────
        if not out:
            if r.returncode != 0:
                return f"[exit {r.returncode}] (no output)"
            return "(no output)"

        # ── Edge case 6: Non-zero exit code ────────────────────────────────
        if r.returncode != 0:
            out = f"[exit {r.returncode}]\n{out}"

        # ── Edge case 7: Truncation with explicit marker ────────────────────
        LIMIT = 10_000
        if len(out) > LIMIT:
            omitted = len(out) - LIMIT
            out = (out[:LIMIT]
                   + f"\n\n[...output truncated — {omitted} additional characters omitted. "
                   "If you need the full output, pipe to a file and read it in chunks.]")

        return out

    except subprocess.TimeoutExpired:
        # ── Edge case 8: Timeout ────────────────────────────────────────────
        return "Error: Timeout (30s). Consider breaking this into smaller steps."


# ── MCP tool registration ─────────────────────────────────────────────────────

@mcp.tool()
def bash(command: str) -> str:
    """Run a bash shell command.
    Use for file operations, running scripts, checking output,
    and any OS interaction. Avoid streaming or infinite commands."""
    return run_bash(command)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[bash-mcp-server] starting on http://127.0.0.1:8765/mcp")
    print("[bash-mcp-server] press Ctrl-C to stop\n")
    mcp.run(transport="streamable-http")
