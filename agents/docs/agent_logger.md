# agent_logger.py — Annotated Source Documentation

This document is a **line-by-line walkthrough** of `agent_logger.py`.
Every design decision is explained so you can understand, modify, or
reuse the module confidently.

---

## Why this module exists

When an AI agent runs, a lot happens invisibly:

```
user types query
  → agent adds it to message history
    → history is sent to the LLM
      → LLM decides to call a tool
        → tool runs, output appended to history
          → history sent to LLM again
            → LLM decides to answer
              → answer printed to user
```

Without logging, you only see the final answer.  
With `AgentLogger`, you see **every step** in a structured file —
ideal for learning, debugging, and understanding how the agent loop works.

---

## File layout overview

```
agent_logger.py
│
├── LOGS_DIR                      # where log files are stored
│
├── _block_to_str(block)          # helper: render one message block → string
├── _render_messages(messages)    # helper: render full message history → lines
│
└── class AgentLogger
    ├── __init__(script_name)     # opens/creates the log file
    │
    ├── session_start(...)        # ═══ SESSION START banner
    ├── session_end()             # ═══ SESSION END banner
    ├── user_input(text)          # ── USER INPUT section
    ├── loop_turn(n)              # ─── LOOP TURN n divider
    ├── llm_request(...)          # ── LLM REQUEST section (DEBUG level)
    ├── llm_response(response)    # ── LLM RESPONSE section
    ├── tool_execution(...)       # ── TOOL EXECUTION section (DEBUG level)
    ├── final_response(text)      # ── AGENT FINAL RESPONSE section
    │
    └── _header(title)            # internal: prints a labelled section header
```

---

## Annotated source

### Imports

```python
import json      # used to serialise tool inputs/outputs to readable JSON strings
import logging   # Python's standard logging framework — handles file I/O and formatting
import os        # used for path operations (not directly, but kept for extension)
from pathlib import Path   # modern path handling; cleaner than os.path
```

**Why `logging` instead of plain `print` / `open`?**  
Python's `logging` module gives us:
- Timestamps on every line automatically
- Log levels (`DEBUG` vs `INFO`) to separate verbose detail from key events
- Safe file handling (buffering, encoding, flush on crash)
- Multiple handlers (add a console handler later with one line)

---

### `LOGS_DIR`

```python
LOGS_DIR = Path(__file__).parent / "logs"
# Path(__file__).parent  →  the directory containing agent_logger.py
#                              (i.e. agents/)
# / "logs"               →  agents/logs/
```

`Path(__file__).parent` means "wherever this file lives", so the module
works correctly regardless of where you `cd` before running a script.

---

### `_block_to_str(block)` — serialise one content block

```python
def _block_to_str(block) -> str:
```

The LLM communicates using **content blocks**. Each block has a `type`:

| `type`        | Meaning                                           | Who creates it         |
|---------------|---------------------------------------------------|------------------------|
| `text`        | A plain text message or final answer              | LLM                    |
| `tool_use`    | A request to call a specific tool                 | LLM                    |
| `tool_result` | The output of a tool that was called              | Agent (us)             |

These blocks appear in two forms depending on context:
- **Plain dicts** — when we build them manually, or when Ollama returns them
- **SDK objects** — when the real Anthropic SDK returns `TextBlock` / `ToolUseBlock`

The function handles both forms transparently:

```python
if isinstance(block, dict):
    # Block was built as a plain dictionary (tool_result we constructed,
    # or Ollama's response translated back to dict form)
    t = block.get("type", "?")
    if t == "text":
        return f"[text] {block.get('text', '')}"
    if t == "tool_use":
        # Log tool name + its full JSON input so we can see what the LLM asked for
        return (
            f"[tool_use] id={block.get('id', '?')}  "
            f"name={block.get('name', '?')}  "
            f"input={json.dumps(block.get('input', {}), ensure_ascii=False)}"
        )
    if t == "tool_result":
        content = str(block.get("content", ""))
        # Truncate long tool outputs to 400 chars to keep logs readable.
        # The full output is logged separately in tool_execution().
        truncated = content[:400] + ("…" if len(content) > 400 else "")
        return f"[tool_result] tool_use_id={block.get('tool_use_id', '?')}  content={truncated}"

if hasattr(block, "type"):
    # Block is an Anthropic SDK object (TextBlock / ToolUseBlock).
    # These have attributes, not dict keys, so we access them with dot notation.
    if block.type == "text":
        return f"[text] {block.text}"
    if block.type == "tool_use":
        return (
            f"[tool_use] id={block.id}  name={block.name}  "
            f"input={json.dumps(block.input, ensure_ascii=False)}"
        )
return str(block)   # fallback for any unknown block type
```

---

### `_render_messages(messages)` — serialise full conversation history

```python
def _render_messages(messages: list) -> list:
    """Return a list of human-readable lines for a full message history."""
```

The agent's `messages` list is the **entire conversation so far**,
accumulated across all loop turns. It grows like this:

```
Turn 1 starts:
  messages = [
    {"role": "user",      "content": "write helloworld.py"}
  ]

Turn 2 starts (after tool call):
  messages = [
    {"role": "user",      "content": "write helloworld.py"},
    {"role": "assistant", "content": [ToolUseBlock(...)]},   ← LLM's tool request
    {"role": "user",      "content": [{"type": "tool_result", ...}]}  ← our tool output
  ]
```

This function renders that list so you can see the full state at each turn:

```python
for i, msg in enumerate(messages):
    role = msg.get("role", "?")        # "user" or "assistant"
    content = msg.get("content", "")

    if isinstance(content, str):
        # Simple text message — truncate at 300 chars for readability
        preview = content[:300] + ("…" if len(content) > 300 else "")
        lines.append(f"    [{i}] {role}: {preview}")

    elif isinstance(content, list):
        # Multi-block message (tool calls, tool results, mixed text+tools)
        # Print each block on its own indented line
        lines.append(f"    [{i}] {role}:")
        for block in content:
            lines.append(f"         {_block_to_str(block)}")

    else:
        lines.append(f"    [{i}] {role}: {content}")   # fallback
```

The index `[i]` makes it easy to count how many turns have accumulated.

---

### `class AgentLogger` — the main class

```python
class AgentLogger:
    _SEP  = "─" * 64   # thin separator  ────────── (used for loop turns)
    _WIDE = "═" * 64   # thick separator ══════════ (used for session start/end)
```

The two separator strings create visual hierarchy in the log:
- `═══` marks the outer session boundary
- `───` marks each agent loop turn within the session

---

### `__init__` — creating the logger

```python
def __init__(self, script_name: str):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # parents=True  → creates agents/logs/ if it doesn't exist
    # exist_ok=True → no error if it already exists

    self.log_path = LOGS_DIR / f"{script_name}.log"
    # e.g. AgentLogger("s01_agent_loop") → agents/logs/s01_agent_loop.log
```

**Why a unique logger name?**

```python
    self._log = logging.getLogger(f"agent_logger.{script_name}")
    # Python's logging module is global — loggers are singletons identified by name.
    # If you imported two agent scripts in the same process (e.g. in tests),
    # they'd share the same logger and write to each other's files.
    # Namespacing by script_name prevents that.

    self._log.handlers.clear()
    # Clear any handlers left from a previous run in the same process.
    # Prevents duplicate log lines if the module is reloaded.

    self._log.propagate = False
    # Don't forward our messages to the root logger.
    # Without this, messages also appear in the console if the root logger
    # has a StreamHandler configured.
```

**`mode="w"` — the "refresh each run" behaviour:**

```python
    fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
    #                                        ──────
    #  "w" = write mode → truncates (empties) the file on open.
    #  This means every time you run the script, you get a clean log
    #  showing only the current session — no clutter from previous runs.
    #  Change to mode="a" (append) if you want to keep history across runs.
```

**Log format:**

```python
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # Example output:
    # 2026-03-08 21:03:18  INFO     ── USER INPUT ──────────────
    # 2026-03-08 21:03:18  DEBUG      model  : claude-haiku-4-5
    #
    # %(levelname)-7s pads the level name to 7 chars so INFO and DEBUG
    # columns align neatly.
```

---

### `session_start` / `session_end` — outer session markers

```python
def session_start(self, provider: str = "", model: str = ""):
    self._log.info(self._WIDE)           # ════════════════════
    self._log.info("  SESSION START")
    self._log.info(f"  script   : {self.log_path.stem}")
    self._log.info(f"  provider : {provider}")   # e.g. "ollama" or "claude"
    self._log.info(f"  model    : {model}")      # e.g. "llama3.1:latest"
    self._log.info(self._WIDE)
```

These create a clear boundary in the file — you can immediately see:
- *which* script produced this log
- *which* LLM backend and model were used

---

### `loop_turn(n)` — per-iteration divider

```python
def loop_turn(self, turn: int):
    self._log.info(self._SEP)
    self._log.info(f"  LOOP TURN {turn}")
    self._log.info(self._SEP)
```

Called at the **top** of each `while True:` iteration in `agent_loop()`.
Count the `LOOP TURN` entries in a log to know how many LLM roundtrips
a task required.  Simple tasks → 2 turns.  Complex multi-step tasks → 4+.

---

### `llm_request` — what we send to the LLM

```python
def llm_request(self, model: str, system: str, messages: list, tools: list = None):
```

Logged at **`DEBUG` level** because the message history grows large and is
verbose — but it's the most important section for understanding what context
the LLM sees at each turn.

```python
    self._log.debug(f"  messages ({len(messages)} total):")
    for line in _render_messages(messages):
        self._log.debug(line)
    # Logs every message in the history, not just the latest one.
    # This lets you see the full context window the LLM is working with.
```

---

### `llm_response` — what the LLM sends back

```python
def llm_response(self, response):
    self._header("LLM RESPONSE")
    self._log.info(f"  stop_reason : {response.stop_reason}")
    for i, block in enumerate(response.content):
        self._log.info(f"  content[{i}]  : {_block_to_str(block)}")
```

`stop_reason` is the key decision indicator:
- `"tool_use"` → LLM wants to call a tool; loop continues
- `"end_turn"` → LLM is done; loop exits and answer is returned to user

---

### `tool_execution` — what the tool did

```python
def tool_execution(self, tool_name: str, tool_input: dict, output: str):
    self._header(f"TOOL EXECUTION  [{tool_name}]")
    self._log.debug(f"  input  : {json.dumps(tool_input, ensure_ascii=False)}")
    # ↑ The exact JSON the LLM sent as the tool call argument
    #   e.g. {"command": "cat > helloworld.py << 'EOF'\nprint('Hello')\nEOF"}

    self._log.debug("  output :")
    lines = output.splitlines() if output.strip() else ["(no output)"]
    for line in lines:
        self._log.debug(f"    {line}")
    # ↑ Full tool output, one line at a time — nothing truncated here.
    #   This lets you see exactly what the agent's "eyes" saw after running
    #   the command, which is what gets fed back into the next LLM request.
```

---

### `_header(title)` — section label helper

```python
def _header(self, title: str):
    pad = max(0, 60 - len(title))
    self._log.info(f"── {title} {'─' * pad}")
    # Produces a line like:
    # ── LLM REQUEST ─────────────────────────────────────────────
    # ── TOOL EXECUTION  [bash] ──────────────────────────────────
    #
    # The padding ensures headers always reach the same total width
    # regardless of the title length, keeping the log visually consistent.
```

---

## Log level design

| Level   | Used for                                      | Why                                              |
|---------|-----------------------------------------------|--------------------------------------------------|
| `INFO`  | Session markers, user input, LLM response summary, final answer | Key events you always want to see |
| `DEBUG` | Full LLM request (system + history), tool I/O | Verbose detail; useful for deep inspection       |

Both levels are written to the file (`fh.setLevel(logging.DEBUG)`).  
If you ever want a less verbose log, change the handler level to `INFO`
and tool I/O + message history will be suppressed.

---

## Extending the logger

**Keep history across runs** — change `mode="w"` to `mode="a"` in `__init__`.

**Also print to console** — add a `StreamHandler` in `__init__`:
```python
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)   # only key events, not full history
ch.setFormatter(fh.formatter)
self._log.addHandler(ch)
```

**Add a new section** — follow the pattern:
```python
def my_event(self, data):
    self._header("MY EVENT")
    self._log.info(f"  {data}")
```

**Use in a multi-threaded agent** (s09–s11) — create one `AgentLogger` per
worker thread with a unique name:
```python
logger = AgentLogger(f"s09_worker_{worker_id}")
```
Each worker gets its own file; Python's `logging` module is thread-safe.
