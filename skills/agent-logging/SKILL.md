---
name: agent-logging
description: |
  Add transparent, structured logging to any AI agent script so every
  interaction — user input, LLM requests/responses, tool executions, and
  final answers — is recorded to a file for inspection and learning.
  Use when:
  (1) you want to understand exactly what the agent sends to/receives from the LLM
  (2) you need to debug unexpected agent behaviour
  (3) you are learning how the agent loop works step-by-step
  (4) you want a permanent audit trail of agent sessions
  Keywords: logging, tracing, debug, transparency, audit, agent loop, LLM request
---

# Agent Logging Skill

Add transparent interaction logging to any agent script so the full flow —
user → agent → LLM → tools → LLM → user — is readable in a log file.

## Files involved

| File | Role |
|---|---|
| `agents/agent_logger.py` | Reusable module. **Copy once, shared by all agents.** |
| `agents/logs/<script>.log` | Auto-created per script. **Overwritten each run.** |
| `agents/<script>.py` | The agent to instrument. Apply the 4 steps below. |

---

## Step-by-step: instrument any agent

### Step 1 — Import and instantiate at module level

```python
from agent_logger import AgentLogger

logger = AgentLogger("s03_todo_write")   # → logs/s03_todo_write.log
```

Use the **script filename without `.py`** as the name. The log file is
created (or overwritten) at import time.

---

### Step 2 — Session boundaries in `__main__`

```python
if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "ollama")
    logger.session_start(provider=provider, model=MODEL)

    # ... your existing interaction loop ...

    logger.session_end()
```

---

### Step 3 — Log user input (in the `__main__` loop)

```python
    query = input(">> ")
    logger.user_input(query)                 # ← add before appending to history
    history.append({"role": "user", "content": query})
```

---

### Step 4 — Instrument the agent loop function

Add **four logging calls** inside the `while True:` loop:

```python
def agent_loop(messages: list):
    turn = 0                                          # ← add turn counter
    while True:
        turn += 1
        logger.loop_turn(turn)                        # ① loop divider
        logger.llm_request(MODEL, SYSTEM, messages, TOOLS)  # ② full request

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        logger.llm_response(response)                 # ③ full response

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                logger.tool_execution(               # ④ tool I/O
                    block.name, block.input, output
                )
                results.append(...)
        messages.append({"role": "user", "content": results})
```

---

### Step 5 — Log the final response (after `agent_loop` returns)

```python
    agent_loop(history)
    for block in history[-1]["content"]:
        if hasattr(block, "text"):
            logger.final_response(block.text)        # ← add before printing
            print(block.text)
```

---

## AgentLogger API reference

| Method | When to call | What it logs |
|---|---|---|
| `session_start(provider, model)` | Once, at program start | Script name, provider, model |
| `session_end()` | Once, at program exit | Session end marker |
| `user_input(text)` | Each user query | Raw query text |
| `loop_turn(n)` | Top of each `while True` iteration | Turn number divider |
| `llm_request(model, system, messages, tools)` | Before `client.messages.create()` | Full model, system, message history, tool schemas |
| `llm_response(response)` | After `client.messages.create()` | `stop_reason` + all content blocks |
| `tool_execution(tool_name, input_dict, output)` | After each tool call | Tool name, JSON input, full output |
| `final_response(text)` | After `agent_loop` returns | Final text shown to user |

---

## What the log looks like

```
════════════════════════════════════════════════════════════════
  SESSION START
  script   : s01_agent_loop
  provider : claude
  model    : claude-haiku-4-5
════════════════════════════════════════════════════════════════
── USER INPUT ───────────────────────────────────────────────────────
  write helloworld.py
────────────────────────────────────────────────────────────────
  LOOP TURN 1
────────────────────────────────────────────────────────────────
── LLM REQUEST ──────────────────────────────────────────────────────
  model  : claude-haiku-4-5
  system : You are a coding agent at /home/user/agents. Use bash…
  tools  : bash
  messages (1 total):
    [0] user: write helloworld.py
── LLM RESPONSE ─────────────────────────────────────────────────────
  stop_reason : tool_use
  content[0]  : [tool_use] id=toolu_01…  name=bash  input={"command": "cat > helloworld.py …"}
── TOOL EXECUTION  [bash] ───────────────────────────────────────────
  input  : {"command": "cat > helloworld.py << 'EOF'\nprint('Hello, World!')\nEOF"}
  output :
    (no output)
────────────────────────────────────────────────────────────────
  LOOP TURN 2
────────────────────────────────────────────────────────────────
── LLM REQUEST ──────────────────────────────────────────────────────
  …
── LLM RESPONSE ─────────────────────────────────────────────────────
  stop_reason : end_turn
  content[0]  : [text] helloworld.py has been created.
── AGENT FINAL RESPONSE ─────────────────────────────────────────────
  helloworld.py has been created.
════════════════════════════════════════════════════════════════
  SESSION END
════════════════════════════════════════════════════════════════
```

---

## Notes

- **Log is always overwritten** on each run (`mode="w"`). To keep history,
  rename the file or append a timestamp to `AgentLogger("s01_"+timestamp)`.
- The module is **thread-safe per logger instance**. For multi-threaded agents
  (s09–s11), create one `AgentLogger` per worker with a unique name.
- `llm_request` logs at `DEBUG` level (full message history can be verbose);
  all other methods log at `INFO` level. Both levels are written to the file.
