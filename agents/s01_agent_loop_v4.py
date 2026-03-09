#!/usr/bin/env python3
"""
s01_agent_loop_v4.py — Agent Loop with MCP Tool Discovery

Same agent loop as v1, but tools are no longer defined inline.
Instead they are discovered at startup from an **MCP server** over HTTP,
and each tool call is dispatched to the server via MCP RPC.

    ┌──────────────────────────────────────────────────────┐
    │                   This agent (client)                │
    │                                                      │
    │  TOOLS  ←── MCP list_tools() ──► bash_mcp_server.py │
    │  run    ──── MCP call_tool()  ──►                    │
    └──────────────────────────────────────────────────────┘

Start the MCP bash server first:
    python3 bash_mcp_server.py

Then run the agent:
    python3 s01_agent_loop_v4.py

Everything else — agent loop, logging, LangSmith, provider support —
is identical to s01_agent_loop.py.
"""

import asyncio
import os
import readline  # noqa: F401 — enables arrow-key / history editing for input()

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from dotenv import load_dotenv
# Explicitly resolve .env relative to this file so it works regardless of CWD
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from compat import make_client
from agent_logger import AgentLogger

client, MODEL, PROVIDER = make_client("claude")

# Wrap the Anthropic client so every messages.create() call is traced in LangSmith.
# Requires env vars: LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT
try:
    from langsmith.wrappers import wrap_anthropic
    client = wrap_anthropic(client)
except ImportError:
    pass  # langsmith not installed — tracing silently disabled
logger = AgentLogger("s01_agent_loop_v4")  # → logs/s01_agent_loop_v4.log (refreshed each run)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── MCP server connection ─────────────────────────────────────────────────────
MCP_SERVER_URL = "http://127.0.0.1:8765/mcp"


async def _discover_tools_async() -> list[dict]:
    """Connect to the MCP server and return Anthropic-formatted tool schemas."""
    async with streamablehttp_client(MCP_SERVER_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            # Convert MCP Tool → Anthropic tool dict
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in result.tools
            ]


async def _call_tool_async(name: str, arguments: dict) -> str:
    """Dispatch a tool call to the MCP server and return the text result."""
    async with streamablehttp_client(MCP_SERVER_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            # Concatenate all text content blocks
            parts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(parts) or "(no output)"


# Discover tools at startup — fail loudly if server is not running
print(f"[mcp] connecting to {MCP_SERVER_URL} ...")
try:
    TOOLS = asyncio.run(_discover_tools_async())
    names = ", ".join(t["name"] for t in TOOLS)
    print(f"[mcp] discovered {len(TOOLS)} tool(s): {names}")
except Exception as _mcp_err:
    print(f"\n[mcp] ERROR: cannot connect to MCP server at {MCP_SERVER_URL}")
    print(f"  {_mcp_err}")
    print("  Start it first:  python3 bash_mcp_server.py\n")
    raise SystemExit(1)


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
                # Dispatch to MCP server via RPC instead of calling locally
                output = asyncio.run(_call_tool_async(block.name, block.input))
                print(output[:200])
                logger.tool_execution(block.name, block.input, output)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    logger.session_start(provider=PROVIDER, model=MODEL)
    history = []
    while True:
        try:
            query = input("\033[36mv4 >> \033[0m")
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
