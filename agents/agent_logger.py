"""
agent_logger.py - Transparent Agent Interaction Logger

Reusable structured logging module for AI agent scripts.
Creates logs/<script_name>.log, overwriting on each run so the file
always shows the most recent session in full detail.

Apply to any agent in 4 steps — see skills/agent-logging/SKILL.md.

Log sections emitted
--------------------
  SESSION START / END         — script name, provider, model
  USER INPUT                  — raw user query
  LOOP TURN n                 — divider for each agent-loop iteration
  LLM REQUEST                 — model, system prompt, full message history, tools
  LLM RESPONSE                — stop_reason + all content blocks
  TOOL EXECUTION [tool_name]  — tool input (JSON) + full output
  AGENT FINAL RESPONSE        — final text returned to the user
"""

import json
import logging
import os
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"


# ── serialisation helpers ──────────────────────────────────────────────────────

def _block_to_str(block) -> str:
    """Render a TextBlock / ToolUseBlock / dict to a readable one-liner."""
    if isinstance(block, dict):
        t = block.get("type", "?")
        if t == "text":
            return f"[text] {block.get('text', '')}"
        if t == "tool_use":
            return (
                f"[tool_use] id={block.get('id', '?')}  "
                f"name={block.get('name', '?')}  "
                f"input={json.dumps(block.get('input', {}), ensure_ascii=False)}"
            )
        if t == "tool_result":
            content = str(block.get("content", ""))
            truncated = content[:400] + ("…" if len(content) > 400 else "")
            return f"[tool_result] tool_use_id={block.get('tool_use_id', '?')}  content={truncated}"
        return str(block)

    if hasattr(block, "type"):
        if block.type == "text":
            return f"[text] {block.text}"
        if block.type == "tool_use":
            return (
                f"[tool_use] id={block.id}  name={block.name}  "
                f"input={json.dumps(block.input, ensure_ascii=False)}"
            )
    return str(block)


def _render_messages(messages: list) -> list:
    """Return a list of human-readable lines for a full message history."""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            preview = content[:300] + ("…" if len(content) > 300 else "")
            lines.append(f"    [{i}] {role}: {preview}")
        elif isinstance(content, list):
            lines.append(f"    [{i}] {role}:")
            for block in content:
                lines.append(f"         {_block_to_str(block)}")
        else:
            lines.append(f"    [{i}] {role}: {content}")
    return lines


# ── main class ─────────────────────────────────────────────────────────────────

class AgentLogger:
    """
    Structured logger for transparent AI agent interaction tracing.

    Each instantiation opens (or creates) logs/<script_name>.log in write
    mode, so every run starts with a clean file.

    Usage::

        from agent_logger import AgentLogger

        logger = AgentLogger("s01_agent_loop")   # → logs/s01_agent_loop.log

        # in __main__
        logger.session_start(provider="claude", model="claude-haiku-4-5")
        logger.user_input(query)

        # in agent_loop
        logger.loop_turn(turn)
        logger.llm_request(MODEL, SYSTEM, messages, TOOLS)
        # ... call LLM ...
        logger.llm_response(response)
        # ... execute tools ...
        logger.tool_execution("bash", block.input, output)

        # back in __main__ after agent_loop returns
        logger.final_response(text)
        logger.session_end()
    """

    _SEP  = "─" * 64
    _WIDE = "═" * 64

    def __init__(self, script_name: str):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGS_DIR / f"{script_name}.log"

        # Use a unique logger name to avoid cross-script handler leaks
        self._log = logging.getLogger(f"agent_logger.{script_name}")
        self._log.setLevel(logging.DEBUG)
        self._log.handlers.clear()
        self._log.propagate = False

        fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._log.addHandler(fh)

    # ── public API ──────────────────────────────────────────────────────────────

    def session_start(self, provider: str = "", model: str = ""):
        """Log the beginning of a new agent session."""
        self._log.info(self._WIDE)
        self._log.info("  SESSION START")
        self._log.info(f"  script   : {self.log_path.stem}")
        self._log.info(f"  provider : {provider}")
        self._log.info(f"  model    : {model}")
        self._log.info(self._WIDE)

    def session_end(self):
        """Log the end of the agent session."""
        self._log.info(self._WIDE)
        self._log.info("  SESSION END")
        self._log.info(self._WIDE)

    def user_input(self, text: str):
        """Log a raw query from the user."""
        self._header("USER INPUT")
        self._log.info(f"  {text}")

    def loop_turn(self, turn: int):
        """Log a divider marking the start of agent loop iteration N."""
        self._log.info(self._SEP)
        self._log.info(f"  LOOP TURN {turn}")
        self._log.info(self._SEP)

    def llm_request(self, model: str, system: str, messages: list, tools: list = None):
        """Log the full payload being sent to the LLM."""
        self._header("LLM REQUEST")
        self._log.debug(f"  model  : {model}")
        sys_preview = system[:300] + ("…" if len(system) > 300 else "")
        self._log.debug(f"  system : {sys_preview}")
        names = [t.get("name", str(t)) for t in (tools or [])]
        self._log.debug(f"  tools  : {', '.join(names) if names else '(none)'}")
        self._log.debug(f"  messages ({len(messages)} total):")
        for line in _render_messages(messages):
            self._log.debug(line)

    def llm_response(self, response):
        """Log the full response received from the LLM."""
        self._header("LLM RESPONSE")
        self._log.info(f"  stop_reason : {response.stop_reason}")
        for i, block in enumerate(response.content):
            self._log.info(f"  content[{i}]  : {_block_to_str(block)}")

    def tool_execution(self, tool_name: str, tool_input: dict, output: str):
        """Log a single tool call — input (JSON) and full output."""
        self._header(f"TOOL EXECUTION  [{tool_name}]")
        self._log.debug(f"  input  : {json.dumps(tool_input, ensure_ascii=False)}")
        self._log.debug("  output :")
        lines = output.splitlines() if output.strip() else ["(no output)"]
        for line in lines:
            self._log.debug(f"    {line}")

    def final_response(self, text: str):
        """Log the final text answer delivered to the user."""
        self._header("AGENT FINAL RESPONSE")
        for line in text.splitlines():
            self._log.info(f"  {line}")

    # ── internal ────────────────────────────────────────────────────────────────

    def _header(self, title: str):
        pad = max(0, 60 - len(title))
        self._log.info(f"── {title} {'─' * pad}")
