"""Run model-written code: Python, C, C++ and Java.

NOT a security sandbox: the code runs as the user and can read/write files and use the network.
Mitigations: the student approves every run after seeing the code; it runs in a fresh temporary folder with
a timeout; the environment is cut down to what compilers and runtimes need (no API keys); input comes only
from the `stdin` argument - never the terminal, where a waiting input() used to hang the chat box - and
output is truncated.

Toolchains come from PATH, or from the portable copies `scripts/setup_toolchains.py` puts in tools/
(GCC from WinLibs in tools/gcc, Microsoft Build of OpenJDK in tools/jdk) - no installer, no admin rights.
C and C++ are linked statically, so the program needs no compiler DLLs to run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import config
from . import Tool

RUN_TIMEOUT_S = 20
MAX_TIMEOUT_S = 60
COMPILE_TIMEOUT_S = 120          # the first GCC/javac run of the day reads a lot from disk
MAX_STDOUT = 4000
MAX_STDERR = 2500
ALIASES = {"python": "python", "py": "python", "python3": "python", "c": "c", "cpp": "cpp", "c++": "cpp",
           "cxx": "cpp", "cc": "cpp", "java": "java"}
TOOLS_DIR = config.ROOT / "tools"
_ENV_KEEP = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC", "USERPROFILE", "HOMEDRIVE",
             "HOMEPATH", "HOME", "APPDATA", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
             "OS", "LANG", "LC_ALL", "SYSTEMDRIVE", "PROGRAMDATA"}


def find_program(name: str) -> str | None:
    """A compiler/runtime: the portable copy in tools/ first, then PATH."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    for d in (TOOLS_DIR / "gcc" / "bin", TOOLS_DIR / "jdk" / "bin"):
        if (d / exe).is_file():
            return str(d / exe)
    found = shutil.which(name)
    if name in ("java", "javac") and found and "java8path" in found.lower():
        return None                                    # Oracle's Java 8 runtime shim: no javac, too old
    return found


def available() -> dict[str, bool]:
    return {"python": True, "c": bool(find_program("gcc")), "cpp": bool(find_program("g++")),
            "java": bool(find_program("javac") and find_program("java"))}


def _env(bin_dir: str | None = None, **extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP}
    if bin_dir:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def _run(cmd: list[str], cwd: str, env: dict, timeout: float, stdin: str | None = None):
    return subprocess.run(cmd, cwd=cwd, env=env, input=stdin if stdin is not None else "",
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _format(p, seconds: float, notes: list[str]) -> str:
    out, err = (p.stdout or "").rstrip(), (p.stderr or "").rstrip()
    parts = [f"exit code {p.returncode}" + (f" ({seconds:.1f} s)" if seconds >= 1 else "")] + notes
    if out:
        parts.append(out[:MAX_STDOUT] + (" ...[stdout truncated]" if len(out) > MAX_STDOUT else ""))
    if err:
        parts.append("stderr:\n" + (err if len(err) <= MAX_STDERR else "...\n" + err[-MAX_STDERR:]))
    if not out and not err:
        parts.append("no output (print the results)")
    return "\n".join(parts)


def _java_class(code: str) -> str:
    m = (re.search(r"public\s+(?:final\s+|abstract\s+)*class\s+([A-Za-z_]\w*)", code)
         or re.search(r"class\s+([A-Za-z_]\w*)[^{]*\{(?:(?!class\s).)*static\s+void\s+main", code, re.S))
    return m.group(1) if m else "Main"


def run_code(language: str, code: str, stdin: str = "", timeout: int = RUN_TIMEOUT_S,
             python_exe: str | None = None) -> str:
    lang = ALIASES.get(str(language).strip().lower())
    if lang is None:
        return f"error: unsupported language '{language}' (use python, c, cpp or java)"
    timeout = max(1, min(int(timeout or RUN_TIMEOUT_S), MAX_TIMEOUT_S))
    with tempfile.TemporaryDirectory(prefix="alice_run_") as tmp:
        notes: list[str] = []
        phase = "compiling"
        try:
            if lang == "python":
                (Path(tmp) / "main.py").write_text(code, encoding="utf-8")
                cmd = [python_exe or sys.executable, "-I", "-X", "utf8", "main.py"]
                env = _env(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
            elif lang in ("c", "cpp"):
                compiler = find_program("gcc" if lang == "c" else "g++")
                if not compiler:
                    return (f"error: no {'C' if lang == 'c' else 'C++'} compiler found. Run "
                            "`python scripts/setup_toolchains.py` once (portable GCC into ALICE's tools/ folder).")
                src = "main.c" if lang == "c" else "main.cpp"
                (Path(tmp) / src).write_text(code, encoding="utf-8")
                std = ["-std=c17"] if lang == "c" else ["-std=c++20"]
                build = [compiler, *std, "-O2", "-Wall", "-Wextra", "-static", "-o", "prog.exe", src, "-lm"]
                env = _env(str(Path(compiler).parent))
                c = _run(build, tmp, env, COMPILE_TIMEOUT_S)
                if c.returncode != 0:
                    return "compile error:\n" + (c.stderr or c.stdout).strip()[-MAX_STDERR:]
                if c.stderr.strip():
                    notes.append("compiler warnings:\n" + c.stderr.strip()[-1500:])
                cmd = [str(Path(tmp) / "prog.exe")]
            else:                                          # java
                javac, java = find_program("javac"), find_program("java")
                if not (javac and java):
                    return ("error: no Java compiler (JDK) found. Run `python scripts/setup_toolchains.py` once "
                            "(portable OpenJDK into ALICE's tools/ folder).")
                cls = _java_class(code)
                (Path(tmp) / f"{cls}.java").write_text(code, encoding="utf-8")
                env = _env(str(Path(javac).parent), JAVA_HOME=str(Path(javac).parents[1]))
                c = _run([javac, "-encoding", "UTF-8", "-d", ".", f"{cls}.java"], tmp, env, COMPILE_TIMEOUT_S)
                if c.returncode != 0:
                    return "compile error:\n" + (c.stderr or c.stdout).strip()[-MAX_STDERR:]
                if c.stderr.strip():
                    notes.append("compiler warnings:\n" + c.stderr.strip()[-1500:])
                cmd = [java, "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-cp", ".", cls]
            phase = "running"
            t0 = time.monotonic()
            p = _run(cmd, tmp, env, timeout, stdin)
            return _format(p, time.monotonic() - t0, notes)
        except subprocess.TimeoutExpired as e:
            why = (" (an infinite loop, a program waiting for more input than stdin gave it, or just too slow?)"
                   if phase == "running" else "")
            return f"error: timed out after {e.timeout:.0f} s while {phase}{why}"
        except OSError as e:
            return f"error: could not start it: {e}"


def make_tool(python_exe: str | None = None) -> Tool:
    langs = [k for k, ok in available().items() if ok]
    return Tool(
        name="run_code",
        description=("Run a short program and get its output. Languages: " + ", ".join(langs) + ". Use it to "
                     "test or verify code, or for calculations too complex for 'calculate'. Print the results. "
                     "If the program reads input (input(), scanf, cin, Scanner), give that input in 'stdin' - "
                     "it cannot type into a prompt. Fresh temporary folder, 20 s time limit; Python has the "
                     "standard library and numpy. The student must approve each run."),
        parameters={"type": "object",
                    "properties": {"language": {"type": "string", "enum": ["python", "c", "cpp", "java"]},
                                   "code": {"type": "string", "description": "the complete program"},
                                   "stdin": {"type": "string", "description": "text typed into the program, if it reads input"}},
                    "required": ["language", "code"]},
        run=lambda language="python", code="", stdin="": run_code(language, code, stdin, python_exe=python_exe),
        needs_confirmation=True,
    )
