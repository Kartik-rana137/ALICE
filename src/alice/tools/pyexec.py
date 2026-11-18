"""Run model-written Python in a subprocess.

NOT a security sandbox: the code runs as the user, can read/write files and use the network.
Mitigations: the user must confirm every run (shown the code first), it runs in a fresh temp
directory with a timeout, isolated mode (-I), and output is truncated.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import Tool

TIMEOUT_S = 20
MAX_OUTPUT = 4000


def run_python(code: str, python_exe: str | None = None, timeout: int = TIMEOUT_S) -> str:
    exe = python_exe or sys.executable
    env = {k: v for k, v in os.environ.items()
           if k.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "WINDIR", "PATHEXT", "COMSPEC")}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="alice_py_") as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([exe, "-I", str(script)], cwd=tmp, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"error: timed out after {timeout}s (infinite loop or too slow?)"
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    parts = []
    if out:
        parts.append(out[:MAX_OUTPUT] + (" ...[stdout truncated]" if len(out) > MAX_OUTPUT else ""))
    if err:
        parts.append("stderr:\n" + err[-MAX_OUTPUT:])
    status = f"exit code {p.returncode}"
    if not parts:
        return f"{status}; no output (use print() to show results)"
    return f"{status}\n" + "\n".join(parts)


def make_tool(python_exe: str | None = None) -> Tool:
    return Tool(
        name="run_python",
        description=("Run a short Python 3 script and get its stdout/stderr. Use it to test or verify code, "
                     "or for calculations too complex for 'calculate'. Always print() the results. "
                     "Runs in an empty temp directory with a 20 s timeout; standard library only "
                     "unless the user has installed more. The user must approve each run."),
        parameters={"type": "object",
                    "properties": {"code": {"type": "string", "description": "complete Python script"}},
                    "required": ["code"]},
        run=lambda code: run_python(code, python_exe),
        needs_confirmation=True,
    )
