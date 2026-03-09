# compat.py — Annotated Source Documentation

---

## About the filename: "compat"

**`compat` is short for "compatibility".**

The name comes from the file's original purpose: making Ollama (which speaks
the OpenAI API) *compatible* with agent code written for the Anthropic SDK.
Rather than rewriting every agent when you switch backends, this one file
absorbs all the translation work.

### Is it a good name?

It is concise and conventional — Python projects commonly use `compat.py` for
"code that bridges incompatible interfaces". But it doesn't tell you *what* is
being made compatible. Better alternatives depending on taste:

| Name | Tone | Best when |
|---|---|---|
| `compat.py` | Terse, traditional | You know the codebase well |
| `llm_client.py` | Descriptive | Team project, first glance clarity |
| `llm_adapter.py` | Architectural | You want to signal the Adapter design pattern |
| `llm_provider.py` | Domain-driven | Focus is on provider switching |
| `backends.py` | Django-style | You prefer ecosystems over patterns |

**Recommendation:** `llm_provider.py` — it immediately tells a new reader
"this is where LLM provider selection happens", which is the file's main job
after the recent refactor.

The file is kept as `compat.py` here for backwards compatibility (ironic, but
true) since every agent imports it by that name.

---

## What this file does in one sentence

It hides the **API differences between Anthropic Claude and Ollama** behind a
single unified interface, so all agent scripts can call the same
`client.messages.create()` regardless of which backend is running.

---

## The core problem it solves

The agents in this project were written for the **Anthropic Python SDK**:

```python
# What every agent does:
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
print(response.stop_reason)    # "tool_use" or "end_turn"
print(response.content)        # list of TextBlock / ToolUseBlock objects
```

But **Ollama** uses a completely different API shape (OpenAI-compatible):

```python
# What Ollama actually speaks:
response = openai_client.chat.completions.create(
    model=model, messages=openai_messages,   # different message format!
    tools=openai_tools,                       # different tool format!
)
print(response.choices[0].finish_reason)     # "stop" or "tool_calls"
print(response.choices[0].message.content)  # string, not list of blocks
```

This file bridges the two worlds so agents never need to know which one
they're talking to.

---

## File layout overview

```
compat.py
│
├── TextBlock, ToolUseBlock, FakeResponse, _FakeMessages
│     ↑ Data classes that mimic the Anthropic SDK's return types.
│       Returned by OllamaCompatClient so agents see the same objects.
│
├── _block_to_dict(block)
│     ↑ Normalise any block (SDK object or plain dict) → plain dict.
│       Used when converting Anthropic history to OpenAI format.
│
├── _convert_tools_to_openai(tools)
│     ↑ Translate Anthropic tool schema → OpenAI function-calling schema.
│
├── _convert_history_to_openai(messages, system)
│     ↑ Translate the full Anthropic message history → OpenAI message list.
│       This is the most complex translation — see detailed section below.
│
├── _convert_response_to_anthropic(openai_response)
│     ↑ Translate OpenAI response → FakeResponse (Anthropic-shaped).
│
├── OllamaCompatClient
│     ↑ The main bridge class. Wraps openai.OpenAI with an
│       Anthropic-compatible client.messages.create() interface.
│
├── _PROVIDER_REGISTRY                   ← dict of provider factories
├── register_provider(name, factory)     ← add custom providers
├── _make_ollama_client()                ← built-in Ollama factory
├── _make_claude_client()                ← built-in Claude factory
│
└── make_client(provider=None)           ← PUBLIC API — the only import agents need
```

---

## Annotated source: data classes

### Why do we need fake Anthropic classes?

```python
class TextBlock:
    """Mimics anthropic.types.TextBlock"""
    def __init__(self, text: str):
        self.type = "text"
        self.text = text
```

When an agent checks the LLM response, it does things like:

```python
for block in response.content:
    if block.type == "tool_use":       # ← attribute access
        print(block.input["command"])  # ← attribute access
    if hasattr(block, "text"):
        print(block.text)              # ← attribute access
```

The real Anthropic SDK returns actual `TextBlock` and `ToolUseBlock` objects
with those attributes. When Ollama replies, we create our own `TextBlock` /
`ToolUseBlock` instances so agents never need an `if ollama: ... else: ...`
branch anywhere.

```python
class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id       # unique call ID, must be echoed back in tool_result
        self.name = name   # tool name e.g. "bash"
        self.input = input # dict of arguments e.g. {"command": "ls -la"}
```

```python
class FakeResponse:
    """Mimics anthropic.types.Message — the top-level response object."""
    def __init__(self, content: list, stop_reason: str):
        self.content = content
        # ↑ list of TextBlock and/or ToolUseBlock objects

        self.stop_reason = stop_reason
        # ↑ "tool_use"  → agent loop continues (LLM wants to call a tool)
        #   "end_turn"  → agent loop exits (LLM is done answering)
```

```python
class _FakeMessages:
    """Mimics the client.messages namespace so you can call client.messages.create()."""
    def __init__(self, create_fn):
        self._create = create_fn

    def create(self, model, messages, max_tokens, system=None, tools=None):
        return self._create(...)
    # ↑ The real Anthropic SDK is accessed as:
    #     client.messages.create(...)
    #   client is an Anthropic object, .messages is a Messages object.
    #   OllamaCompatClient needs the same nested structure, so we fake
    #   the .messages namespace with this simple wrapper class.
```

---

## Annotated source: `_block_to_dict`

```python
def _block_to_dict(block) -> dict:
```

The message history accumulated by an agent can contain a **mix** of types:
- Real `anthropic.types.TextBlock` objects (when Claude backend was used)
- Our fake `TextBlock` / `ToolUseBlock` objects (when Ollama backend was used)
- Plain dicts `{"type": "tool_result", ...}` (always — we build these ourselves)

This function normalises all three forms to plain dicts before translation:

```python
    if isinstance(block, dict):
        return block             # already a dict, nothing to do

    if hasattr(block, "type"):   # SDK object (real or fake) — use attributes
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input}

    return {"type": "text", "text": str(block)}   # safe fallback
```

---

## Annotated source: `_convert_tools_to_openai`

The two APIs describe tools (functions the LLM can call) differently:

```python
# Anthropic format — used by agent scripts:
{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {              # ← Anthropic calls it "input_schema"
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
}

# OpenAI format — required by Ollama:
{
    "type": "function",            # ← wraps everything in {"type":"function", "function":{}}
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {            # ← OpenAI calls it "parameters"
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
    }
}
```

The function maps `input_schema` → `parameters` and adds the `"type": "function"` wrapper.

---

## Annotated source: `_convert_history_to_openai` (the hard part)

This is the most complex function. The two APIs represent the same
conversation history in fundamentally different ways.

### The core structural difference

```
Anthropic model:                    OpenAI model:
──────────────────────────────      ────────────────────────────────────────
role=user                           role=system  (separate message!)
  content="..."
                                    role=user
role=assistant                        content="..."
  content=[                         role=assistant
    TextBlock("thinking..."),         content=None   (no text when calling tools)
    ToolUseBlock(id, name, input)     tool_calls=[{id, type, function:{name,args}}]
  ]
                                    role=tool        (separate message per result!)
role=user                             tool_call_id=id
  content=[                           content="..."
    {type:tool_result,
     tool_use_id:id,
     content:"..."}
  ]
```

**Three key differences:**

1. **System prompt**: Anthropic passes it as a separate `system=` parameter.
   OpenAI requires it as the first message with `role=system`.

2. **Tool calls in assistant turn**: Anthropic puts them inside the `content`
   list alongside text. OpenAI puts them in a separate `tool_calls` field.

3. **Tool results**: Anthropic puts them as a `user` message containing a
   `tool_result` block, linking back via `tool_use_id`. OpenAI requires a
   completely separate `role=tool` message per result, linking via `tool_call_id`.

### Annotated conversion logic

```python
def _convert_history_to_openai(anthropic_messages: list, system: str | None) -> list:
    openai_msgs = []

    if system:
        # Difference #1: system prompt becomes first message in OpenAI
        openai_msgs.append({"role": "system", "content": system})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                # Simple text query — identical in both APIs
                openai_msgs.append({"role": "user", "content": content})

            elif isinstance(content, list):
                # This user message may contain:
                #   (a) tool_result blocks   → become role=tool messages in OpenAI
                #   (b) text blocks          → become a regular user message
                tool_results = []
                text_parts = []
                for part in content:
                    part_dict = _block_to_dict(part)
                    if part_dict.get("type") == "tool_result":
                        tool_results.append(part_dict)
                    else:
                        text_parts.append(part_dict.get("text", str(part_dict)))

                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    if isinstance(tool_content, list):
                        # tool content can itself be a list of text blocks
                        tool_content = " ".join(
                            c.get("text", str(c)) if isinstance(c, dict) else str(c)
                            for c in tool_content
                        )
                    # Difference #3: each tool result → separate role=tool message
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", "unknown"),
                        # ↑ must match the id from the tool_calls in the assistant turn
                        "content": str(tool_content),
                    })

                if text_parts:
                    openai_msgs.append({"role": "user",
                                        "content": "\n".join(text_parts)})

        elif role == "assistant":
            if isinstance(content, str):
                # Plain string assistant turn (synthetic, used in s08-s11)
                openai_msgs.append({"role": "assistant", "content": content})

            elif isinstance(content, list):
                blocks = [_block_to_dict(b) for b in content]
                text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
                tool_uses  = [b for b in blocks if b.get("type") == "tool_use"]

                assistant_msg = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    # ↑ OpenAI requires content=None (not omitted) when tool_calls present
                }
                if tool_uses:
                    # Difference #2: tool calls move to separate tool_calls field
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tu["id"],
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu["input"]),
                                # ↑ OpenAI requires arguments as a JSON *string*, not dict
                            }
                        }
                        for tu in tool_uses
                    ]
                openai_msgs.append(assistant_msg)

    return openai_msgs
```

---

## Annotated source: `_convert_response_to_anthropic`

Takes an OpenAI response and wraps it in our `FakeResponse` so agents
see the same shape regardless of backend:

```python
def _convert_response_to_anthropic(openai_response) -> FakeResponse:
    choice = openai_response.choices[0]
    # ↑ OpenAI always returns a list of "choices" (for n>1 completions).
    #   We always use n=1, so choices[0] is the only response.

    message = choice.message
    finish_reason = choice.finish_reason
    # ↑ "stop"       → LLM finished naturally
    #   "tool_calls" → LLM wants to call a tool

    content_blocks = []

    if message.content:
        content_blocks.append(TextBlock(message.content))
        # ↑ Regular text reply → wrap in our TextBlock mimic

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
                # ↑ OpenAI returns arguments as a JSON string — parse it back to dict
            except (json.JSONDecodeError, AttributeError):
                arguments = {}   # malformed JSON from some models — safe default
            content_blocks.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=arguments,
            ))

    # Map finish_reason → stop_reason
    if message.tool_calls:
        stop_reason = "tool_use"
        # ↑ Prioritise this check: some Ollama versions return finish_reason="stop"
        #   even when tool_calls is populated (a known Ollama bug).
        #   Checking message.tool_calls directly is more reliable.
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    return FakeResponse(content=content_blocks, stop_reason=stop_reason)
```

---

## Annotated source: `OllamaCompatClient`

```python
class OllamaCompatClient:
    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        # ↑ Lazy import: only imported when Ollama is actually selected.
        #   If you're using Claude, openai package is never touched.

        self._openai_client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            # ↑ Ollama's OpenAI-compatible endpoint lives at {host}/v1
            #   e.g. http://localhost:11434/v1
            api_key="ollama",
            # ↑ The openai SDK requires a non-empty api_key to initialise,
            #   but Ollama doesn't actually validate or use it.
        )
        self._model = model
        self.messages = _FakeMessages(self._create)
        # ↑ Creates the .messages namespace so agents can call
        #   client.messages.create(...) — same as the Anthropic SDK.

    def _create(self, model, messages, max_tokens, system=None, tools=None):
        # Step 1: translate Anthropic history → OpenAI messages
        openai_messages = _convert_history_to_openai(messages, system)

        kwargs = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"
            # ↑ "auto" = let the model decide whether to call a tool.
            #   Alternative: "required" (always call), "none" (never call).

        # Step 2: call Ollama
        response = self._openai_client.chat.completions.create(**kwargs)

        # Step 3: translate OpenAI response → Anthropic-shaped FakeResponse
        return _convert_response_to_anthropic(response)
```

---

## Annotated source: provider registry

```python
_NO_TOOL_MODELS = ("llama3:latest", "llama3:8b", "llama3:70b", "llama2")
# ↑ These older Llama versions don't reliably support tool/function calling.
#   The agent loop requires tools, so we warn the user before they hit
#   cryptic failures deep inside a session.

_PROVIDER_REGISTRY: dict = {}
# ↑ Maps provider name string → factory function.
#   e.g. {"ollama": _make_ollama_client, "claude": _make_claude_client}
#   Populated below by register_provider() calls.
```

```python
def register_provider(name: str, factory):
    _PROVIDER_REGISTRY[name] = factory
    # ↑ Simple dict assignment. Call this before make_client() to add
    #   a new backend. The factory is a zero-argument callable that
    #   returns (client, model_string).
```

```python
def _make_ollama_client():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.getenv("OLLAMA_MODEL",    "llama3.1:latest")
    # ↑ Sensible defaults mean you can run locally with zero env vars set.
    ...
    return OllamaCompatClient(base_url=base_url, model=model), model

def _make_claude_client():
    from anthropic import Anthropic
    # ↑ Lazy import: only pulled in when Claude provider is selected.
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        # ↑ Some Anthropic-compatible relay providers (MiniMax, DeepSeek, etc.)
        #   reject requests that include ANTHROPIC_AUTH_TOKEN in headers.
        #   Remove it when using a custom base URL.
    client = Anthropic(base_url=base_url)
    # ↑ base_url=None uses the default https://api.anthropic.com
    model = os.environ.get("MODEL_ID", "claude-sonnet-4-6")
    return client, model

# Register all built-in providers at module load time
register_provider("ollama",    _make_ollama_client)
register_provider("claude",    _make_claude_client)
register_provider("anthropic", _make_claude_client)   # alias — same factory
```

---

## Annotated source: `make_client` — the public API

```python
def make_client(provider: str | None = None):
    resolved = provider or os.getenv("LLM_PROVIDER", "ollama")
    # ↑ Resolution order:
    #   1. Explicit argument:  make_client("claude")
    #   2. Environment var:    LLM_PROVIDER=claude
    #   3. Hard default:       "ollama"
    #
    # This means changing the backend for all agents is a single line in .env.

    factory = _PROVIDER_REGISTRY.get(resolved)
    if factory is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(
            f"[compat] Unknown provider '{resolved}'. "
            f"Available: {available}. ..."
        )
        # ↑ Fail fast with a clear message rather than a cryptic AttributeError
        #   later. The error lists all registered names so the fix is obvious.

    return factory()
    # ↑ Calls e.g. _make_ollama_client() which returns (client, model_string).
    #   Every agent does: client, MODEL = make_client()
```

---

## Data flow summary

```
agent script
    │
    │  make_client()
    ▼
compat.py
    │
    ├─ LLM_PROVIDER=claude ──────────────────► Anthropic()
    │                                          native SDK, no translation needed
    │
    └─ LLM_PROVIDER=ollama ──────────────────► OllamaCompatClient
                                                │
                    agent calls                 │
                    client.messages.create()    │
                                                │
                    Anthropic history ──────────┤ _convert_history_to_openai()
                    Anthropic tools  ──────────►│ _convert_tools_to_openai()
                                                │
                                                │ openai.chat.completions.create()
                                                │       ↓  Ollama
                                                │ OpenAI response
                                                │
                    FakeResponse ◄──────────────┤ _convert_response_to_anthropic()
                    (looks like Anthropic SDK)  │
                                                │
    agent reads response.stop_reason ◄──────────┘
    agent reads response.content[].type
```

Everything above the `OllamaCompatClient` box is identical regardless of
which backend is chosen — that's the whole point of `compat.py`.
