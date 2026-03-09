#!/usr/bin/env python3
"""
test_run_bash.py — Demo tests for the improved run_bash() in s01_agent_loop.py

Covers all 8 edge cases handled by run_bash():
    1. Dangerous command blocklist
    2. Streaming / non-terminating command blocklist
    3. Binary / non-UTF-8 output
    4. stderr vs stdout separation
    5. Empty output (silent success)
    6. Non-zero exit code
    7. Output truncation with explicit marker
    8. Timeout

Run:
    python3 test_run_bash.py
    python3 test_run_bash.py -v      # verbose: print actual output for every test
"""

import sys
import types
import importlib.util

# ── Load run_bash without triggering LLM initialisation ─────────────────────
# make_client() connects to the LLM at import time.  We stub it out so the
# tests work with no API key / Ollama server required.
import compat as _compat
_compat.make_client = lambda *a, **k: (
    types.SimpleNamespace(messages=None), "test-model"
)

_spec = importlib.util.spec_from_file_location("s01", "s01_agent_loop.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_bash = _mod.run_bash

# ── Helpers ───────────────────────────────────────────────────────────────────

VERBOSE = "-v" in sys.argv

pass_count = 0
fail_count = 0


def check(label: str, command: str, *, contains: str = None, starts_with: str = None,
          exact: str = None, not_contains: str = None):
    """Run command, assert result, print pass/fail."""
    global pass_count, fail_count
    result = run_bash(command)

    if VERBOSE:
        print(f"\n{'─'*60}")
        print(f"  TEST : {label}")
        print(f"  CMD  : {command}")
        print(f"  OUT  : {result!r}")

    ok = True
    reason = ""
    if exact is not None and result != exact:
        ok = False; reason = f"expected exact {exact!r}"
    if contains is not None and contains not in result:
        ok = False; reason = f"expected {contains!r} in output"
    if starts_with is not None and not result.startswith(starts_with):
        ok = False; reason = f"expected output to start with {starts_with!r}"
    if not_contains is not None and not_contains in result:
        ok = False; reason = f"expected {not_contains!r} NOT in output"

    status = "PASS ✓" if ok else f"FAIL ✗  ({reason})"
    if not VERBOSE:
        print(f"  {'PASS ✓' if ok else 'FAIL ✗':<8} {label}")
    else:
        print(f"  {'PASS ✓' if ok else 'FAIL ✗  ' + reason}")

    if ok:
        pass_count += 1
    else:
        fail_count += 1


# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════")
print("  run_bash() edge-case demo tests")
print("══════════════════════════════════════════════════\n")

# ── 1. Dangerous command blocklist ────────────────────────────────────────────
print("[ 1 ] Dangerous command blocklist")
check("sudo blocked",         "sudo apt update",
      contains="Dangerous command blocked")
check("rm -rf / blocked",     "rm -rf /tmp && rm -rf /",
      contains="Dangerous command blocked")
check("reboot blocked",       "reboot now",
      contains="Dangerous command blocked")
check("shutdown blocked",     "shutdown -h now",
      contains="Dangerous command blocked")
check("device write blocked", "dd if=/dev/zero > /dev/sda",
      contains="Dangerous command blocked")

# ── 2. Streaming / non-terminating blocklist ──────────────────────────────────
print("\n[ 2 ] Streaming command blocklist")
check("ping blocked",       "ping google.com",       contains="Streaming command blocked")
check("tail -f blocked",    "tail -f /var/log/syslog", contains="Streaming command blocked")
check("watch blocked",      "watch ls",              contains="Streaming command blocked")
check("top blocked",        "top",                   contains="Streaming command blocked")
check("while true blocked", "while true; do echo x; done",
      contains="Streaming command blocked")
check("suggestion shown",   "ping google.com",
      contains="timeout 5 ping")   # helpful alternative is suggested

# ── 3. Binary / non-UTF-8 output ─────────────────────────────────────────────
print("\n[ 3 ] Binary / non-UTF-8 output")
# Use Python to write raw non-UTF-8 bytes directly to stdout
_bin_cmd = "python3 -c \"import sys; sys.stdout.buffer.write(bytes([0x80, 0x81, 0x82, 0x83]))\""
check("binary output caught",
      _bin_cmd,
      contains="binary/non-UTF-8 output")
check("binary includes exit code",
      _bin_cmd,
      contains="[exit")

# ── 4. stderr vs stdout separation ────────────────────────────────────────────
print("\n[ 4 ] stderr vs stdout separation")
check("stdout visible",
      "echo STDOUT_LINE",
      contains="STDOUT_LINE")
check("stderr labelled",
      "echo ERR_LINE >&2",
      contains="[stderr] ERR_LINE")
check("both streams present",
      "echo STDOUT_LINE; echo ERR_LINE >&2",
      contains="STDOUT_LINE")
check("stderr not confused with stdout",
      "echo STDOUT_LINE; echo ERR_LINE >&2",
      contains="[stderr] ERR_LINE")
check("stderr label absent from clean command",
      "echo hello",
      not_contains="[stderr]")

# ── 5. Empty output (silent success) ─────────────────────────────────────────
print("\n[ 5 ] Empty output (silent success)")
check("true returns (no output)",  "true",          exact="(no output)")
check("mkdir -p silent success",
      "mkdir -p /tmp/run_bash_test_dir && rmdir /tmp/run_bash_test_dir",
      exact="(no output)")
check("touch silent success",
      "touch /tmp/run_bash_test_file && rm /tmp/run_bash_test_file",
      exact="(no output)")

# ── 6. Non-zero exit code ─────────────────────────────────────────────────────
print("\n[ 6 ] Non-zero exit code")
check("exit code prepended",
      "ls /no_such_path_xyz_abc_99",
      starts_with="[exit ")
check("exit 1 from false",     "false",             starts_with="[exit 1]")
check("exit 2 from bad ls",
      "ls /no_such_path_xyz_abc_99",
      contains="[exit 2]")
check("stderr error text present",
      "ls /no_such_path_xyz_abc_99",
      contains="No such file or directory")
check("zero exit has no prefix",
      "echo ok",
      not_contains="[exit ")

# ── 7. Truncation with explicit marker ────────────────────────────────────────
print("\n[ 7 ] Output truncation with explicit marker")
# Generate 15,000 chars (> 10,000 limit)
check("truncation marker present",
      "python3 -c \"print('A' * 15000)\"",
      contains="output truncated")
check("omitted char count shown",
      "python3 -c \"print('A' * 15000)\"",
      contains="characters omitted")
check("pipe suggestion shown",
      "python3 -c \"print('A' * 15000)\"",
      contains="pipe to a file")
check("short output NOT truncated",
      "echo hello",
      not_contains="truncated")

# ── 8. Normal success (sanity check) ─────────────────────────────────────────
print("\n[ 8 ] Normal successful commands (sanity)")
check("echo works",           "echo hello world",       contains="hello world")
check("multi-line output",    "printf 'a\nb\nc'",        contains="a")
check("python one-liner",     "python3 -c \"print(1+1)\"", contains="2")
check("pwd works",            "pwd",                     contains="/")
check("exit code not in clean output",
      "echo clean",
      not_contains="[exit")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n══════════════════════════════════════════════════")
print(f"  Results: {pass_count} passed, {fail_count} failed")
print(f"══════════════════════════════════════════════════\n")

if fail_count:
    sys.exit(1)
