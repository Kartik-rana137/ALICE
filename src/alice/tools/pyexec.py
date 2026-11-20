"""Run model-written Python (kept for callers of run_python; the chat uses the run_code tool in coderun.py,
which also runs C, C++ and Java)."""
from __future__ import annotations

from .coderun import RUN_TIMEOUT_S, run_code

TIMEOUT_S = RUN_TIMEOUT_S


def run_python(code: str, python_exe: str | None = None, timeout: int = TIMEOUT_S, stdin: str = "") -> str:
    return run_code("python", code, stdin=stdin, timeout=timeout, python_exe=python_exe)
