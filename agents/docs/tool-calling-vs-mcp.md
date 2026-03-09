# Tool Calling vs MCP — Comparison & Best Practices

> When people say "give the LLM tools", there are two completely different mechanisms.
> This doc explains both, compares them, and tells you when to reach for each one.

---

## 1. Two Ways to Give an LLM Tools

### Approach A — Inline Tool Definition (what s01_agent_loop.py does)

The host application defines the tool schema in JSON and passes it directly
to the LLM API on every call. The application also runs the tool when the
LLM asks for it.

```python
# s01_agent_loop.py — tool defined inline in Python
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS,          # ← LLM sees this list on every request
    max_tokens=8000,
)

# When LLM returns stop_reason="tool_use", the HOST runs the tool:
if block.type == "tool_use":
    output = run_bash(block.input["command"])   # ← executed right here in Python
```

**What happens under the hood:**
```
Host App  ──(POST /messages + tools=[...])──>  LLM API
          <──(tool_use: bash, command="ls")──
          
Host App runs bash("ls")
          ──(tool_result: "file1.py\nfile2.py")──>  LLM API
          <──(text: "Here are the files...")──
```

The LLM never "runs" anything. It outputs a structured JSON block asking
the host to run a named tool. The host executes it and feeds the result back.

---

### Approach B — MCP (Model Context Protocol)

MCP is an open protocol (introduced by Anthropic, Nov 2024) that separates
**tool servers** from **host applications**. Tool implementations live in
standalone MCP Servers. The host (Claude Desktop, VS Code Copilot, your
custom client) connects to those servers and asks "what tools do you have?"
at startup, then includes those descriptions when calling the LLM.

```
┌──────────────┐    MCP protocol     ┌────────────────────┐
│  MCP Client  │ ◄────────────────── │   MCP Server       │
│  (Your Host) │    list_tools()     │  (e.g. filesystem, │
│              │ ──────────────────► │   git, postgres,   │
│              │    call_tool()      │   your custom one) │
└──────┬───────┘                     └────────────────────┘
       │  tool schemas (discovered)
       ▼
  LLM API  (same /messages call as Approach A)
```

An MCP server exposes three primitives:
- **Tools** — callable functions the LLM can invoke
- **Resources** — readable data (files, DB rows, API responses)
- **Prompts** — reusable prompt templates

Example: the official `mcp-server-filesystem` gives any MCP host tools like
`read_file`, `write_file`, `list_directory`, `search_files` — without the
host needing to implement any of them.

---

## 2. Side-by-Side Comparison

| Dimension | Inline Tool Definition | MCP |
|---|---|---|
| **Who defines tools** | Your host application (Python/JS/etc.) | Separate MCP server process |
| **Discovery** | Hardcoded `TOOLS` list in source | Dynamic: host queries server at startup |
| **Execution** | Host runs tool directly (e.g. `subprocess.run`) | Host sends `call_tool` RPC to the server |
| **Scope** | One application, one set of tools | One server can serve many different hosts |
| **Communication** | Direct function call (in-process) | JSON-RPC over stdio or HTTP/SSE |
| **Tool reuse** | Copy-paste to each new agent | Point a new host at the same MCP server |
| **Security boundary** | None — tool runs in same process | Server is a separate process; can sandbox it |
| **Latency** | Zero (in-process) | Small RPC overhead (~1–5ms local) |
| **Protocol spec** | Proprietary to each LLM provider | Open standard (spec at modelcontextprotocol.io) |
| **Ecosystem** | Your own tools only | 1000+ pre-built servers (filesystem, git, DB, browser, etc.) |
| **Best for** | Custom one-off agents, learning, prototyping | Reusable tools, multi-host setups, production |
| **Complexity** | Low — a Python dict + an `if` block | Medium — separate server, transport setup |

---

## 3. The Protocol in Detail

### How an LLM API call uses tools (both approaches)

Regardless of whether tools came from inline code or MCP, the LLM API call
is **identical**. MCP just controls *how the `tools` list gets populated*
and *who executes the tool* when the LLM asks.

```
Step 1 — Host sends:
  POST /messages
  {
    "model": "claude-haiku-4-5",
    "tools": [{"name": "bash", "description": "...", "input_schema": {...}}],
    "messages": [{"role": "user", "content": "list files in /tmp"}]
  }

Step 2 — LLM decides to use a tool, responds with:
  {
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "id": "t_01", "name": "bash",
                 "input": {"command": "ls /tmp"}}]
  }

Step 3 — Host executes the tool (inline: directly; MCP: via RPC to server)
  result = "file1  file2  file3"

Step 4 — Host feeds result back:
  POST /messages
  {
    "messages": [
      ...,
      {"role": "assistant", "content": [tool_use block]},
      {"role": "user",      "content": [{"type": "tool_result",
                                          "tool_use_id": "t_01",
                                          "content": "file1  file2  file3"}]}
    ]
  }

Loop continues until stop_reason != "tool_use"
```

This loop is exactly what `agent_loop()` in `s01_agent_loop.py` implements.
MCP does **not** change this loop — it only changes where the `tools` list
comes from and who runs step 3.

---

## 4. What MCP Actually Adds

```
Without MCP (Approach A):
  Agent A  defines bash, read_file, write_file  → works for Agent A only
  Agent B  must re-implement the same tools     → code duplication

With MCP (Approach B):
  MCP filesystem server  exposes read_file, write_file, list_directory
  Agent A  connects → gets those tools for free
  Agent B  connects → gets the same tools for free
  VS Code Copilot  connects → gets the same tools for free
  Claude Desktop   connects → gets the same tools for free
```

MCP is essentially **a standard plugin interface for LLM tools** — the same
idea as VS Code extensions or browser extensions, but for AI agent tooling.

---

## 5. When to Use Each

### Use inline tool definition when:

- **Learning / prototyping** — you want to understand the full loop
  without protocol overhead (s01_agent_loop.py is the perfect starting point)
- **Tight integration** — the tool is inseparable from your application
  logic (e.g. a tool that reads your in-memory state)
- **One-off agent** — the tools will never be reused across other agents
- **Maximum control** — you want to intercept, log, or transform every
  tool call in one place (see `run_bash()` with its 8 safety checks)
- **No infrastructure available** — you cannot run a separate server

### Use MCP when:

- **Sharing tools across agents** — multiple scripts / apps need the same
  tools (filesystem, git, database queries)
- **Using pre-built tool servers** — why rewrite `read_file` when
  `mcp-server-filesystem` already handles edge cases?
- **Integrating with MCP-native hosts** — Claude Desktop, VS Code Copilot,
  Cursor, and others speak MCP natively; your server works in all of them
- **Security isolation** — running tool execution in a separate process
  with tighter permissions
- **Cross-language tooling** — your tool server can be written in any
  language (Go, Rust, Node, Python); the host just speaks JSON-RPC

---

## 6. Best Practices

### For inline tool definition

1. **Start simple.** One tool (`bash`) is enough for most coding agents.
   Add tools only when the LLM asks for something it can't do with bash.

2. **Make tool descriptions behavioural, not structural.** The LLM reads
   `description` to decide *when* to call the tool. Be specific:
   ```python
   # weak:
   "description": "Run a command"
   # strong:
   "description": "Run a bash shell command. Use for file operations,
                   running scripts, checking output, and any OS interaction."
   ```

3. **Add safety guards in the executor, not the schema.** The LLM cannot
   be fully trusted to avoid dangerous inputs. Validate in `run_bash()` as
   `s01_agent_loop.py` already does (dangerous commands, streaming commands,
   binary output, truncation).

4. **Log every tool call.** Use `AgentLogger.tool_execution()` so you can
   replay exactly what the agent did during a session.

5. **Return structured error strings, not exceptions.** If the tool fails,
   return `"Error: ..."`. Never let an exception propagate to the LLM loop —
   it will crash the session.

### For MCP

1. **Use `stdio` transport for local tools.** It is the simplest and most
   portable transport — no ports to open, no network config.

2. **Use `streamable-http` (HTTP/SSE) for remote / multi-client tools.**
   Needed when multiple hosts need to connect simultaneously.

3. **One concern per server.** Don't jam filesystem + database + browser
   into one MCP server. Keep them separate so you can mix and match.

4. **Version your tool schemas.** Tool `input_schema` changes can break
   existing agents. Treat them like API versioning.

5. **Always test with `mcp dev`** (the MCP inspector) before wiring up
   to a real host. Catches schema errors before the LLM ever sees them.

---

## 7. Migration Path: Inline → MCP

If you start with inline tools (right choice for learning) and later want
to promote them to MCP servers, the pattern is mechanical:

```python
# BEFORE: inline in s01_agent_loop.py
def run_bash(command: str) -> str:
    ...  # 80 lines of logic

TOOLS = [{"name": "bash", "description": "...", "input_schema": {...}}]

# AFTER: promote to an MCP server
# 1. Create bash_server.py that wraps the same run_bash() logic
# 2. Register the tool with @mcp.tool()
# 3. In s01_agent_loop.py, replace TOOLS with discovered tools from the server
# 4. Replace run_bash() call with mcp_client.call_tool("bash", {"command": ...})
```

The agent loop itself (`while stop_reason == "tool_use"`) does not change.

---

## 8. Quick Reference

```
"I want to learn how tool calling works"           → Inline (s01_agent_loop.py)
"I want to use Claude Desktop / VS Code Copilot"  → MCP
"I'm building one agent for a specific task"       → Inline
"I'm building a tool library for many agents"      → MCP
"I need full control over every shell command"     → Inline (keep run_bash guards)
"I want to add database queries to multiple bots"  → MCP server for DB
"I'm demoing/prototyping"                          → Inline
"I'm shipping to production"                       → Consider MCP for reusable parts,
                                                      keep inline for app-specific logic
```

---

## References

- [Anthropic tool use docs](https://docs.anthropic.com/en/docs/tool-use)
- [MCP specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP server registry](https://github.com/modelcontextprotocol/servers)
- This project: `agents/s01_agent_loop.py` — canonical inline example
