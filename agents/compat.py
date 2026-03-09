"""
compat.py - Multi-Provider LLM Compatibility Adapter

Provides make_client() that returns (client, model) for any supported LLM provider.
All agent files call make_client() without changes; the provider is selected at
runtime via the LLM_PROVIDER environment variable (default: "ollama").

Provider selection
------------------
  LLM_PROVIDER=ollama     → OllamaCompatClient  (local Ollama, default)
  LLM_PROVIDER=claude     → Anthropic SDK client
  LLM_PROVIDER=anthropic  → alias for "claude"
  (or pass provider= arg directly: make_client("claude"))

Custom providers can be registered before calling make_client():
  from compat import register_provider
  register_provider("my-llm", lambda: (MyClient(), "my-model"))

API surface emulated by OllamaCompatClient (same as Anthropic SDK):
    client.messages.create(model, messages, max_tokens, system=None, tools=None)
    -> response with:
        response.stop_reason  ("tool_use" | "end_turn")
        response.content      list of TextBlock | ToolUseBlock objects

Message history translation (Anthropic <-> OpenAI/Ollama):
    Anthropic "assistant" turn stores response.content (list of SDK objects or dicts)
    OpenAI "assistant" turn stores content (str|None) + tool_calls (list|None)

    Tool results in Anthropic: user message with content=[{type:tool_result, ...}]
    Tool results in OpenAI:    separate {"role":"tool", "tool_call_id":..., ...} messages
"""

import json
import os

from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Anthropic-like data classes (returned by OllamaCompatClient)
# ---------------------------------------------------------------------------

class TextBlock:
    """Mimics anthropic.types.TextBlock"""
    def __init__(self, text: str):
        self.type = "text"
        self.text = text

    def __repr__(self):
        return f"TextBlock(text={self.text[:40]!r})"


class ToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock"""
    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input

    def __repr__(self):
        return f"ToolUseBlock(id={self.id!r}, name={self.name!r})"


class FakeResponse:
    """Mimics anthropic.types.Message"""
    def __init__(self, content: list, stop_reason: str):
        self.content = content          # list[TextBlock | ToolUseBlock]
        self.stop_reason = stop_reason  # "tool_use" | "end_turn"


class _FakeMessages:
    """Mimics client.messages namespace"""
    def __init__(self, create_fn):
        self._create = create_fn

    def create(self, model, messages, max_tokens, system=None, tools=None):
        return self._create(
            model=model, messages=messages, max_tokens=max_tokens,
            system=system, tools=tools,
        )


# ---------------------------------------------------------------------------
# Message history translation helpers
# ---------------------------------------------------------------------------

def _block_to_dict(block) -> dict:
    """
    Normalize an Anthropic SDK object (TextBlock, ToolUseBlock) or plain dict
    to a plain dict. Handles mixed histories where real SDK objects and our
    mimics coexist alongside plain dicts.
    """
    if isinstance(block, dict):
        return block
    if hasattr(block, "type"):
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": "text", "text": str(block)}


def _convert_tools_to_openai(tools: list) -> list:
    """
    Anthropic:  {"name": "bash", "description": "...", "input_schema": {...}}
    OpenAI:     {"type": "function", "function": {"name": "bash", "description": "...", "parameters": {...}}}
    """
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            }
        })
    return result


def _convert_history_to_openai(anthropic_messages: list, system: str | None) -> list:
    """
    Translate a full Anthropic-format message history to OpenAI format.

    Anthropic history shape (accumulated over turns):
        [
          {"role": "user", "content": "string"},
          {"role": "assistant", "content": [TextBlock, ToolUseBlock, ...]},
          {"role": "user", "content": [{"type":"tool_result", "tool_use_id":"...", "content":"..."}]},
          {"role": "assistant", "content": [TextBlock]},
        ]

    OpenAI output shape:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": None, "tool_calls": [{...}]},
          {"role": "tool", "tool_call_id": "...", "content": "..."},
          {"role": "assistant", "content": "..."},
        ]
    """
    openai_msgs = []

    if system:
        openai_msgs.append({"role": "system", "content": system})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                openai_msgs.append({"role": "user", "content": content})

            elif isinstance(content, list):
                tool_results = []
                text_parts = []
                for part in content:
                    part_dict = _block_to_dict(part)
                    if part_dict.get("type") == "tool_result":
                        tool_results.append(part_dict)
                    else:
                        # Injected reminder text or other text blocks
                        text_parts.append(part_dict.get("text", str(part_dict)))

                # Tool results → separate role=tool messages (one per result)
                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = " ".join(
                            c.get("text", str(c)) if isinstance(c, dict) else str(c)
                            for c in tool_content
                        )
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", "unknown"),
                        "content": str(tool_content),
                    })

                # Non-tool text parts → regular user message
                if text_parts:
                    openai_msgs.append({"role": "user", "content": "\n".join(text_parts)})

            else:
                openai_msgs.append({"role": "user", "content": str(content)})

        elif role == "assistant":
            if isinstance(content, str):
                # Synthetic injected turns like "Noted." (s08-s11)
                openai_msgs.append({"role": "assistant", "content": content})

            elif isinstance(content, list):
                blocks = [_block_to_dict(b) for b in content]
                text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
                tool_uses = [b for b in blocks if b.get("type") == "tool_use"]

                assistant_msg = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_uses:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tu["id"],
                            "type": "function",
                            "function": {
                                "name": tu["name"],
                                "arguments": json.dumps(tu["input"]),
                            }
                        }
                        for tu in tool_uses
                    ]
                openai_msgs.append(assistant_msg)

            else:
                openai_msgs.append({"role": "assistant", "content": str(content)})

    return openai_msgs


def _convert_response_to_anthropic(openai_response) -> FakeResponse:
    """
    Translate an OpenAI chat completion response to an Anthropic-like FakeResponse.

    OpenAI:    choices[0].message.content (str|None) + .tool_calls (list|None)
               choices[0].finish_reason ("stop" | "tool_calls")
    Anthropic: response.content (list of TextBlock/ToolUseBlock)
               response.stop_reason ("end_turn" | "tool_use")
    """
    choice = openai_response.choices[0]
    message = choice.message
    finish_reason = choice.finish_reason

    content_blocks = []

    if message.content:
        content_blocks.append(TextBlock(message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            content_blocks.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=arguments,
            ))

    # Map finish_reason to Anthropic stop_reason.
    # Some Ollama versions send "stop" even when tool_calls are present.
    if message.tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    return FakeResponse(content=content_blocks, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# OllamaCompatClient
# ---------------------------------------------------------------------------

class OllamaCompatClient:
    """
    Wraps openai.OpenAI pointed at Ollama's OpenAI-compatible endpoint ({host}/v1).
    Exposes client.messages.create() with the same signature as the Anthropic SDK.

    Stateless per-call — safe for concurrent use across threads (s09-s11).
    """

    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        self._openai_client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",  # Ollama ignores the key; SDK requires a non-empty value
        )
        self._model = model
        self.messages = _FakeMessages(self._create)

    def _create(self, model, messages, max_tokens, system=None, tools=None):
        openai_messages = _convert_history_to_openai(messages, system)
        kwargs = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        response = self._openai_client.chat.completions.create(**kwargs)
        return _convert_response_to_anthropic(response)


# ---------------------------------------------------------------------------
# Provider registry & built-in factories
# ---------------------------------------------------------------------------

# Models known to lack reliable tool-calling support (Ollama)
_NO_TOOL_MODELS = ("llama3:latest", "llama3:8b", "llama3:70b", "llama2")

# Registry: provider name -> zero-arg factory returning (client, model_str)
_PROVIDER_REGISTRY: dict = {}


def register_provider(name: str, factory):
    """
    Register a custom LLM provider factory.

    ``factory`` must be a callable with no required arguments that returns
    ``(client, model_str)``.  Once registered it can be selected via the
    ``LLM_PROVIDER`` env var or the ``provider`` argument of ``make_client()``.

    Example::

        from compat import register_provider, OllamaCompatClient

        def my_provider():
            client = OllamaCompatClient("http://mygpu:11434", "mistral")
            return client, "mistral"

        register_provider("mistral-local", my_provider)
    """
    _PROVIDER_REGISTRY[name] = factory


def _make_ollama_client():
    """Built-in factory: Ollama via its OpenAI-compatible endpoint."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    if any(model.startswith(m) for m in _NO_TOOL_MODELS):
        print(
            f"[compat] WARNING: '{model}' does not support tool calling. "
            "All sessions require tools. Recommend: ollama pull llama3.1:latest"
        )
    print(f"[compat] Provider=ollama  {base_url}  model={model}")
    return OllamaCompatClient(base_url=base_url, model=model), model


def _make_claude_client():
    """Built-in factory: Anthropic Claude (also covers Anthropic-compatible APIs)."""
    from anthropic import Anthropic
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        # Some relay providers conflict with ANTHROPIC_AUTH_TOKEN
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    client = Anthropic(base_url=base_url)
    model = os.environ.get("MODEL_ID", "claude-sonnet-4-6")
    print(f"[compat] Provider=claude  model={model}")
    return client, model


# Register built-in providers
register_provider("ollama",     _make_ollama_client)
register_provider("claude",     _make_claude_client)
register_provider("anthropic",  _make_claude_client)   # alias


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_client(provider: str | None = None):
    """
    Return ``(client, model)`` for the chosen LLM provider.

    Provider resolution order
    -------------------------
    1. ``provider`` argument  — explicit programmatic override
    2. ``LLM_PROVIDER`` environment variable
    3. Default: ``"ollama"``

    Built-in providers
    ------------------
    =========== ===============================================================
    Name        Description / required env vars
    =========== ===============================================================
    ollama      Local Ollama inference (OLLAMA_BASE_URL, OLLAMA_MODEL)
    claude      Anthropic Claude API  (ANTHROPIC_BASE_URL optional, MODEL_ID)
    anthropic   Alias for ``claude``
    =========== ===============================================================

    Custom providers
    ----------------
    Register a factory before calling ``make_client()``::

        from compat import register_provider
        register_provider("my-llm", lambda: (MyClient(), "my-model"))

    Usage in agent files
    --------------------
    ::

        from compat import make_client

        client, MODEL = make_client()           # honours LLM_PROVIDER env var
        client, MODEL = make_client("claude")   # force Claude regardless of env
    """
    resolved = provider or os.getenv("LLM_PROVIDER", "ollama")
    factory = _PROVIDER_REGISTRY.get(resolved)
    if factory is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(
            f"[compat] Unknown provider '{resolved}'. "
            f"Available: {available}. "
            "Register custom providers with register_provider()."
        )
    return factory()
