#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import readline  # noqa: F401 — enables arrow-key / history editing for input()
import subprocess

from compat import make_client
from agent_logger import AgentLogger

client, MODEL = make_client("claude")
logger = AgentLogger("s01_agent_loop")  # → logs/s01_agent_loop.log (refreshed each run)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    # ── Edge case 1: Dangerous command blocklist ────────────────────────────
    # Prevent catastrophic or irreversible operations before they run.
    # The LLM receives an explicit error so it knows to try a safer approach.
    DANGEROUS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in DANGEROUS):
        return "Error: Dangerous command blocked"

    # ── Edge case 2: Streaming / non-terminating commands ──────────────────
    # These commands run forever and would block the agent until timeout (120s).
    # We reject them early with a helpful suggestion rather than hanging.
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
            capture_output=True, timeout=30,   # reduced from 120s — fail fast
            # text=False so we receive raw bytes and can detect binary ourselves
        )

        # ── Edge case 3: Binary / non-UTF-8 output ─────────────────────────
        # Executables, images, compiled files etc. produce binary output that
        # looks like garbage to the LLM and can cause hallucinations.
        # Decode carefully; replace undecodable bytes with a clear marker.
        try:
            stdout = r.stdout.decode("utf-8")
            stderr = r.stderr.decode("utf-8")
        except UnicodeDecodeError:
            return (f"[exit {r.returncode}] "
                    "[binary/non-UTF-8 output — not displayable as text. "
                    "Use xxd, file, or strings if you need to inspect it.]")

        # ── Edge case 4: Separate stdout and stderr ─────────────────────────
        # Merging stdout+stderr hides whether a line is a warning or real output.
        # Label each stream so the LLM can distinguish them.
        parts = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            # Prefix stderr lines so LLM knows these are warnings/errors,
            # not the normal output of the command.
            stderr_lines = "\n".join(
                f"[stderr] {line}" for line in stderr.strip().splitlines()
            )
            parts.append(stderr_lines)

        out = "\n".join(parts).strip()

        # ── Edge case 5: Empty output ───────────────────────────────────────
        # Many successful commands (cp, mkdir, chmod…) print nothing.
        # Return an explicit string so the LLM knows the command ran and
        # succeeded — without this it may assume the command failed or repeat it.
        if not out:
            # Still surface a non-zero exit even with no output
            if r.returncode != 0:
                return f"[exit {r.returncode}] (no output)"
            return "(no output)"

        # ── Edge case 6: Non-zero exit code ────────────────────────────────
        # Prepend the exit code so the LLM has an unambiguous failure signal.
        # Without this, output like "error: ..." is easy for the LLM to miss
        # or misinterpret; "[exit 1]" at the start is unmistakable.
        if r.returncode != 0:
            out = f"[exit {r.returncode}]\n{out}"

        # ── Edge case 7: Truncation with explicit marker ────────────────────
        # A hard character cut can slice a JSON object, stack trace, or file
        # in half, causing the LLM to parse corrupt data and draw wrong conclusions.
        # Cap at 10K characters and tell the LLM exactly how much was omitted.
        LIMIT = 10_000
        if len(out) > LIMIT:
            omitted = len(out) - LIMIT
            out = (out[:LIMIT]
                   + f"\n\n[...output truncated — {omitted} additional characters omitted. "
                   "If you need the full output, pipe to a file and read it in chunks.]")

        return out

    except subprocess.TimeoutExpired:
        # ── Edge case 8: Timeout ────────────────────────────────────────────
        # Command ran longer than our limit. Return a clear message so the LLM
        # knows to break the task into smaller steps or use a faster approach.
        return "Error: Timeout (30s). Consider breaking this into smaller steps."


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    turn = 0
    while True:
        turn += 1
        logger.loop_turn(turn)
        logger.llm_request(MODEL, SYSTEM, messages, TOOLS)

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        logger.llm_response(response)

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        # Execute each tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                logger.tool_execution("bash", block.input, output)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "ollama")
    logger.session_start(provider=provider, model=MODEL)
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    logger.final_response(block.text)
                    print(block.text)
        print()
    logger.session_end()
