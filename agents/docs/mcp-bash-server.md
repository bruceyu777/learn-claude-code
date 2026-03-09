# MCP Bash Server — Implementation & Usage Guide

> How the `bash` tool was extracted from an inline agent into a standalone
> MCP HTTP server, and how `s01_agent_loop_v4.py` discovers and calls it.

---

## Files

| File | Role |
|---|---|
| `bash_mcp_server.py` | MCP server — hosts the `bash` tool over HTTP |
| `s01_agent_loop_v4.py` | MCP client agent — discovers tools at startup, calls them via RPC |

Compare with `s01_agent_loop.py` (v1) where the tool is defined and run inline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Terminal 1: bash_mcp_server.py                                 │
│                                                                 │
│  FastMCP (host=127.0.0.1, port=8765)                           │
│  Transport: streamable-http                                     │
│  Endpoint:  http://127.0.0.1:8765/mcp                          │
│                                                                 │
│  @mcp.tool()  bash(command: str) -> str                        │
│    └─ run_bash(): 8 safety / quality guards                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │  JSON-RPC over HTTP
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Terminal 2: s01_agent_loop_v4.py (MCP client)                 │
│                                                                 │
│  Startup:  list_tools()  →  TOOLS = [{name, description,       │
│                                        input_schema}, ...]     │
│                                                                 │
│  Per turn: LLM says tool_use  →  call_tool(name, args)         │
│                                →  result fed back to LLM       │
│                                                                 │
│  - AgentLogger  →  logs/s01_agent_loop_v4.log                  │
│  - LangSmith    →  tracing via wrap_anthropic                  │
└─────────────────────────────────────────────────────────────────┘
```

The **LLM API protocol does not change**. The agent still sends
`tools=[...]` to the Anthropic API and handles `stop_reason: tool_use`
responses. MCP only changes:
1. *Where the `tools` list comes from* → from the server, not hardcoded
2. *Where the tool runs* → inside the server process, not the agent process

---

## The MCP Server (`bash_mcp_server.py`)

### Setup

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "bash-server",
    host="127.0.0.1",
    port=8765,
)
```

`FastMCP` is Anthropic's high-level MCP server framework (part of the
official `mcp` Python SDK ≥ 1.0). The constructor accepts `host`, `port`,
and many optional settings (auth, lifespan, log level, etc.).

### Tool registration

```python
@mcp.tool()
def bash(command: str) -> str:
    """Run a bash shell command.
    Use for file operations, running scripts, checking output,
    and any OS interaction. Avoid streaming or infinite commands."""
    return run_bash(command)
```

`@mcp.tool()` does three things automatically:
- Registers the function as an MCP tool named `"bash"`
- Generates the `inputSchema` JSON Schema from the Python type hints
  (`command: str` → `{"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}`)
- Uses the docstring as the tool `description` the LLM reads to decide
  *when* to call the tool

The actual implementation (`run_bash`) is kept separate so it can be
tested independently.

### Starting the server

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

`transport="streamable-http"` starts an HTTP server with a single
endpoint at `<host>:<port>/mcp` (the `mount_path` default).
The server speaks the MCP Streamable HTTP transport — JSON-RPC over POST.

```
$ python3 bash_mcp_server.py
[bash-mcp-server] starting on http://127.0.0.1:8765/mcp
INFO: Uvicorn running on http://127.0.0.1:8765
```

---

## The MCP Client Agent (`s01_agent_loop_v4.py`)

### Imports

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
```

- `streamablehttp_client` — async context manager that opens an HTTP
  connection to the MCP server
- `ClientSession` — high-level MCP session object (`initialize`,
  `list_tools`, `call_tool`, etc.)
- `asyncio.run()` — bridges the async MCP calls into the synchronous
  agent loop

### Step 1 — Tool Discovery (startup)

```python
MCP_SERVER_URL = "http://127.0.0.1:8765/mcp"

async def _discover_tools_async() -> list[dict]:
    async with streamablehttp_client(MCP_SERVER_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()          # MCP handshake
            result = await session.list_tools() # → list of Tool objects
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,   # already a dict
                }
                for t in result.tools
            ]

TOOLS = asyncio.run(_discover_tools_async())
# → [{"name": "bash", "description": "...", "input_schema": {...}}]
```

`TOOLS` is now in **Anthropic API format** — the exact same structure
that was hardcoded in v1. From this point on `agent_loop()` is unchanged.

### Step 2 — Tool Execution (per turn)

```python
async def _call_tool_async(name: str, arguments: dict) -> str:
    async with streamablehttp_client(MCP_SERVER_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            parts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(parts) or "(no output)"
```

`call_tool` sends a `CallToolRequest` JSON-RPC message to the server.
The server runs `run_bash(command)` and returns a list of content blocks.
The client concatenates all text blocks into a single string which is
returned as the `tool_result` to the LLM.

### Step 3 — Inside `agent_loop()` (unchanged structure)

```python
for block in response.content:
    if block.type == "tool_use":
        # v1: output = run_bash(block.input["command"])
        # v4: dispatched to MCP server instead
        output = asyncio.run(_call_tool_async(block.name, block.input))
        logger.tool_execution(block.name, block.input, output)
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

The only change from v1: `run_bash(...)` → `asyncio.run(_call_tool_async(...))`.
Everything around it — logging, LangSmith tracing, message history — is identical.

---

## Running It

```bash
# 1. Install dependencies (once)
pip install "mcp[cli]>=1.0.0"

# 2. Start the tool server (keep this terminal open)
cd agents/
python3 bash_mcp_server.py
# → [bash-mcp-server] starting on http://127.0.0.1:8765/mcp

# 3. Run the agent in a new terminal
cd agents/
python3 s01_agent_loop_v4.py
# → [mcp] connecting to http://127.0.0.1:8765/mcp ...
# → [mcp] discovered 1 tool(s): bash
# → v4 >>
```

If the server is not running, the agent fails with a clear message:
```
[mcp] ERROR: cannot connect to MCP server at http://127.0.0.1:8765/mcp
  Start it first:  python3 bash_mcp_server.py
```

---

## Adding More Tools

To expose a second tool (e.g. `read_file`), add it to `bash_mcp_server.py`:

```python
@mcp.tool()
def read_file(path: str) -> str:
    """Read the full text content of a file at the given path."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except UnicodeDecodeError:
        return "Error: file contains binary/non-UTF-8 content"
```

Restart the server. The agent discovers it automatically at startup
with **no code changes**:
```
[mcp] discovered 2 tool(s): bash, read_file
```

The new tool is sent to the LLM on the next `messages.create()` call
and the LLM can immediately start using it. This is the core value of
MCP — the agent needs zero changes to gain new capabilities.

---

## Connection Lifecycle

Each `_discover_tools_async()` and `_call_tool_async()` call opens a
**fresh HTTP connection** to the server:

```
client                                 server
  │── POST /mcp (initialize) ────────► │
  │◄─ 200 OK (session ID) ────────────  │
  │── POST /mcp (list_tools) ────────► │
  │◄─ 200 OK (tool list) ─────────────  │
  │── DELETE /mcp (terminate) ───────► │
```

This is the stateless pattern (`stateless_http=True` makes this default).
Each call is independent. Alternative: keep the session open across turns
to save round-trip overhead — suitable for high-frequency agents.

---

## Log File

The agent writes to `logs/s01_agent_loop_v4.log` (same `AgentLogger` as v1).
Tool calls appear as:

```
── TOOL EXECUTION  [bash] ────────────────────────
  input  : {"command": "ls /tmp"}
  output :
    file1.txt
    file2.txt
```

The tool name (`bash`, `read_file`, etc.) in `logger.tool_execution(block.name, ...)`
comes directly from the MCP response — no hardcoding needed.

---

## Key Differences: v1 vs v4

| | `s01_agent_loop.py` (v1) | `s01_agent_loop_v4.py` (v4) |
|---|---|---|
| `TOOLS` definition | Hardcoded `TOOLS = [...]` in the file | Discovered via `list_tools()` at startup |
| Tool execution | `run_bash(command)` in-process | `call_tool(name, args)` RPC to server |
| Adding tools | Edit the agent file and add a function | Edit only the server file, restart server |
| Dependencies | None beyond stdlib | `mcp` SDK + running server |
| `agent_loop()` body | Calls `run_bash` | Calls `asyncio.run(_call_tool_async(...))` |
| Everything else | — | Identical |
