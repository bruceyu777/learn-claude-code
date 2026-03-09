# The AI Agent Loop — How It Works

> A learn & share session on building transparent AI agents from scratch.

---

## Agenda

1. [What is an AI Agent?](#1-what-is-an-ai-agent)
2. [The Secret: One Simple Loop](#2-the-secret-one-simple-loop)
3. [Key Concepts: Tools, Messages, Stop Reason](#3-key-concepts)
4. [Code Walkthrough: s01_agent_loop.py](#4-code-walkthrough)
5. [How the Message History Grows](#5-how-the-message-history-grows)
6. [Multi-Provider Support](#6-multi-provider-support)
7. [Transparent Logging](#7-transparent-logging)
8. [Live Demo: Read the Log](#8-live-demo-read-the-log)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. What is an AI Agent?

A regular LLM call is **one question, one answer**:

```
You:  "What is the capital of France?"
LLM:  "Paris."
```

An **AI Agent** is different — it can **take actions** in the real world
and keep going until the task is done:

```
You:   "Find all TODO comments in the codebase and write a summary report."
Agent: [searches files]  →  [reads 12 files]  →  [creates report.md]  →  "Done. Found 7 TODOs."
```

The agent doesn't just answer — it **acts, observes, and decides what to do next**,
repeating until the task is complete.

### Real-world analogy

Think of it like hiring a junior developer:
- You give them a task
- They have access to tools: terminal, file system, browser
- They work through it step by step, checking results as they go
- They report back when done

The LLM plays the role of the developer's brain. The tools are their hands.

---

## 2. The Secret: One Simple Loop

Here is the **entire architecture** of an AI coding agent:

```
┌─────────────────────────────────────────────────────────┐
│                      AGENT LOOP                         │
│                                                         │
│   ┌──────────┐     ┌───────┐     ┌─────────────────┐   │
│   │   User   │────▶│  LLM  │────▶│  Tool execution │   │
│   │  prompt  │     │       │     │  (bash, files…) │   │
│   └──────────┘     └───┬───┘     └────────┬────────┘   │
│                        │◀────────────────┘             │
│                        │    tool result fed back       │
│                        │                               │
│                   stop_reason?                         │
│                   ┌────┴─────┐                         │
│               tool_use     end_turn                    │
│                   │            │                       │
│               (loop)       (return to user)            │
└─────────────────────────────────────────────────────────┘
```

In Python, this is literally:

```python
while True:
    response = LLM(messages, tools)         # ask the LLM what to do next
    if response.stop_reason != "tool_use":
        return                               # LLM is done — exit loop
    for tool_call in response.content:
        output = execute_tool(tool_call)     # run what the LLM asked for
        messages.append(output)              # feed result back to LLM
```

That's it. **The entire "agent" is this while-loop.**  
Everything else — safety checks, logging, multi-provider support — is layered on top.

---

## 3. Key Concepts

### 3.1 Tools

A **tool** is a function the LLM is allowed to call.
You describe it in JSON, and the LLM decides when and how to use it.

```python
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"}
        },
        "required": ["command"],
    },
}]
```

> The LLM never executes code directly — it asks *you* to run it,
> then you feed the output back. **You are always in control.**

### 3.2 Messages — the growing conversation

The agent maintains a `messages` list that grows with every turn.
This is the LLM's **entire memory** — it sees the full history on every call.

```python
messages = [
    {"role": "user",      "content": "write helloworld.py and run it"},
    {"role": "assistant", "content": [ToolUseBlock(name="bash", ...)]},
    {"role": "user",      "content": [{"type": "tool_result", "content": "(no output)"}]},
    {"role": "assistant", "content": [ToolUseBlock(name="bash", command="python3 helloworld.py")]},
    {"role": "user",      "content": [{"type": "tool_result", "content": "Hello, World!"}]},
    {"role": "assistant", "content": [TextBlock("Done! helloworld.py runs correctly.")]},
]
```

Each turn adds 2 messages: the LLM's response + our tool result.
A 3-turn task = 7 messages total (1 initial + 2×3).

### 3.3 `stop_reason` — the loop controller

After every LLM call, we check `response.stop_reason`:

| Value | Meaning | What we do |
|---|---|---|
| `"tool_use"` | LLM wants to call a tool | Execute it, append result, loop again |
| `"end_turn"` | LLM is satisfied and done | Return the answer to the user |

This single field drives the entire loop.

---

## 4. Code Walkthrough

### Full annotated `s01_agent_loop.py`

```python
#!/usr/bin/env python3

import os
import readline    # gives arrow-key / history editing for input() — one import, free feature
import subprocess

from compat import make_client       # unified client for any LLM backend
from agent_logger import AgentLogger # transparent logging of every interaction

# ── Initialise once at module load ──────────────────────────────────────────
client, MODEL = make_client()        # reads LLM_PROVIDER from .env
logger = AgentLogger("s01_agent_loop")  # → logs/s01_agent_loop.log

# System prompt: sets the LLM's persona and primary instruction
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# The one tool this agent has
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# ── Safety wrapper around subprocess ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"     # never run these
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"  # cap at 50K chars
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# ── THE AGENT LOOP ───────────────────────────────────────────────────────────
def agent_loop(messages: list):
    turn = 0
    while True:
        turn += 1
        logger.loop_turn(turn)                            # log: LOOP TURN n
        logger.llm_request(MODEL, SYSTEM, messages, TOOLS)  # log: full request

        # ① Ask the LLM: given all messages so far, what should I do next?
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        logger.llm_response(response)                     # log: full response

        # ② Add LLM's reply to history (it may contain text AND/OR tool calls)
        messages.append({"role": "assistant", "content": response.content})

        # ③ If no tool call requested — we're done
        if response.stop_reason != "tool_use":
            return

        # ④ Execute each requested tool call, collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")   # yellow
                output = run_bash(block.input["command"])
                print(output[:200])
                logger.tool_execution("bash", block.input, output)    # log: tool I/O
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # links result back to the specific call
                    "content": output,
                })

        # ⑤ Feed all results back to LLM as next "user" message → loop continues
        messages.append({"role": "user", "content": results})


# ── User interaction loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "ollama")
    logger.session_start(provider=provider, model=MODEL)
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")   # cyan prompt
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)                            # ← the magic happens here
        # Print the final text response
        for block in history[-1]["content"]:
            if hasattr(block, "text"):
                logger.final_response(block.text)
                print(block.text)
        print()
    logger.session_end()
```

---

## 5. How the Message History Grows

Let's trace a 2-turn task: *"write helloworld.py and run it"*

```
INITIAL STATE
─────────────
messages = [
  {role: user, content: "write helloworld.py and run it"}
]

───────────────── LOOP TURN 1 ─────────────────

→ Send to LLM:  (1 message)

← LLM replies:  stop_reason = "tool_use"
                content = [ToolUseBlock(id="t1", name="bash",
                           input={"command": "printf '...' > helloworld.py"})]

messages = [
  {role: user,      content: "write helloworld.py and run it"},   ← original
  {role: assistant, content: [ToolUseBlock(id="t1", ...)]},        ← LLM turn 1
  {role: user,      content: [{type: tool_result,
                                tool_use_id: "t1",
                                content: "(no output)"}]},          ← tool result
]

───────────────── LOOP TURN 2 ─────────────────

→ Send to LLM:  (3 messages — full history)

← LLM replies:  stop_reason = "tool_use"
                content = [ToolUseBlock(id="t2", name="bash",
                           input={"command": "python3 helloworld.py"})]

messages = [
  ... (3 from before),
  {role: assistant, content: [ToolUseBlock(id="t2", ...)]},        ← LLM turn 2
  {role: user,      content: [{type: tool_result,
                                tool_use_id: "t2",
                                content: "Hello, World!"}]},        ← tool result
]

───────────────── LOOP TURN 3 ─────────────────

→ Send to LLM:  (5 messages — full history)

← LLM replies:  stop_reason = "end_turn"   ← done!
                content = [TextBlock("helloworld.py created and runs correctly.")]

RETURN TO USER
```

**Observation:** The LLM sees the *entire* history on every call.
It reasons like a human reading a chat thread — context is everything.

---

## 6. Multi-Provider Support

The same agent code runs against any LLM. Provider is selected via `.env`:

```
# .env

# Use local Ollama (free, no internet needed) — DEFAULT
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest

# Use Anthropic Claude (paid API)
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-xxx
MODEL_ID=claude-haiku-4-5
```

### How it works — the Adapter pattern

```
agent script
    │
    │  client, MODEL = make_client()
    ▼
compat.py  (the compatibility layer)
    │
    ├─ LLM_PROVIDER=claude ──► Anthropic SDK     (native, no translation)
    │
    └─ LLM_PROVIDER=ollama ──► OllamaCompatClient
                                    │
                                    ├─ translates Anthropic message format → OpenAI format
                                    ├─ calls Ollama's /v1/chat/completions endpoint
                                    └─ translates OpenAI response → Anthropic response shape
```

The agent scripts never know which backend is running.
Switching providers = **one line in `.env`**.

### Why not just use OpenAI format everywhere?

The agents are written in Anthropic style because:
- Anthropic's tool-calling API is cleaner for agentic use
- The course material targets Claude
- `compat.py` absorbs all the translation — agents stay clean

### Extending with a custom provider

```python
from compat import register_provider, OllamaCompatClient

def my_vllm_provider():
    client = OllamaCompatClient("http://mygpu:8000", "mistral-7b")
    return client, "mistral-7b"

register_provider("vllm", my_vllm_provider)
# Then set LLM_PROVIDER=vllm in .env
```

---

## 7. Transparent Logging

Every session writes a detailed log to `logs/s01_agent_loop.log`.
The file is **overwritten on each run** — always shows the latest session.

### Log structure

```
════════════════════════════════════════════════════════════════
  SESSION START
  script   : s01_agent_loop
  provider : ollama
  model    : llama3.1:latest
════════════════════════════════════════════════════════════════

── USER INPUT ───────────────────────────────────────────────────
  write helloworld.py and run it

────────────────────────────────────────────────────────────────
  LOOP TURN 1
────────────────────────────────────────────────────────────────

── LLM REQUEST ──────────────────────────────────────────────────
  model  : llama3.1:latest
  system : You are a coding agent at /agents. Use bash to solve tasks.
  tools  : bash
  messages (1 total):
    [0] user: write helloworld.py and run it

── LLM RESPONSE ─────────────────────────────────────────────────
  stop_reason : tool_use
  content[0]  : [tool_use] id=toolu_01  name=bash  input={"command": "printf ..."}

── TOOL EXECUTION  [bash] ───────────────────────────────────────
  input  : {"command": "printf 'print(\"Hello, World!\")' > helloworld.py"}
  output :
    (no output)

────────────────────────────────────────────────────────────────
  LOOP TURN 2
────────────────────────────────────────────────────────────────
  ...

── AGENT FINAL RESPONSE ─────────────────────────────────────────
  helloworld.py created and runs correctly. Output: Hello, World!

════════════════════════════════════════════════════════════════
  SESSION END
════════════════════════════════════════════════════════════════
```

### What to look for in the log

| What you see | What it tells you |
|---|---|
| Number of `LOOP TURN` sections | How many LLM roundtrips the task took |
| `messages (N total)` growing each turn | The context window accumulating |
| `stop_reason: tool_use` → `end_turn` | The exact moment the LLM decided it was done |
| `TOOL EXECUTION` input | The exact command the LLM chose |
| `TOOL EXECUTION` output | What the agent's "eyes" saw — feeds into next turn |

---

## 8. Live Demo: Read the Log

After running:
```
$ python3 s01_agent_loop.py
s01 >> count how many python files are in this folder and list them
```

Open `logs/s01_agent_loop.log` and answer:
1. How many loop turns did it take?
2. What bash command did the LLM choose?
3. At which turn did `stop_reason` change to `end_turn`?
4. How many messages were in history by the final turn?

---

## 9. Key Takeaways

### The agent loop in one sentence
> Feed tool results back to the LLM in a loop until it stops asking for tools.

### The three things that make it work

```
1. TOOLS          — give the LLM hands (bash, file I/O, search, APIs…)
2. MESSAGE HISTORY — give the LLM memory (accumulate everything in messages[])
3. STOP REASON    — give the LLM a voice ("I'm done" vs "run this for me")
```

### What makes a *good* agent

| Factor | Bad | Good |
|---|---|---|
| Tool design | One giant "do everything" tool | Small, focused tools |
| System prompt | Vague instructions | Clear persona + constraint |
| Safety | No guards | Block dangerous commands |
| Observability | Silent, black box | Logged, transparent |
| Backend | Hardcoded | Provider-agnostic |

### The surprising truth

The complexity is **not** in the agent loop — that's 15 lines.  
The complexity is in:
- Writing good tool descriptions (the LLM's instructions)
- Writing good system prompts (the LLM's personality)
- Handling edge cases in tool output (the LLM's reality)

The model already knows how to be an agent.  
Your code just gives it the opportunity.

---

## References

| File | Purpose |
|---|---|
| `agents/s01_agent_loop.py` | The agent itself |
| `agents/compat.py` | Multi-provider compatibility layer |
| `agents/agent_logger.py` | Transparent interaction logger |
| `agents/logs/s01_agent_loop.log` | Latest session log |
| `agents/docs/compat.md` | Deep-dive: how provider translation works |
| `agents/docs/agent_logger.md` | Deep-dive: how logging works |
| `skills/agent-logging/SKILL.md` | How to add logging to any agent |

---

*Built with the `learn_claude_code` project — a hands-on tour of AI agent architecture.*
