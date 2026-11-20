"""Run model-written code: Python, C, C++ and Java.

NOT a security sandbox: the code runs as the user and can read/write files and use the network.
Mitigations: the student approves every run after seeing the code; it runs in a fresh temporary folder with
a timeout; the environment is cut down to what compilers and runtimes need (no API keys); input comes only
from the `stdin` argument - never the terminal, where a waiting input() used to hang the chat box - and
output is truncated.

The student sees the program run: inside `live_output(callback)` (set by the chat front ends around a turn)
every run reports start, compiler messages, output as it is printed, and the exit status. Python runs
unbuffered and C/C++ programs get unbuffered stdout, so output shows up while the program is still running.
`open_in_console` runs a program in its own console window instead, where the student can type into it.

Toolchains come from PATH, or from the portable copies `scripts/setup_toolchains.py` puts in tools/
(GCC from WinLibs in tools/gcc, Microsoft Build of OpenJDK in tools/jdk) - no installer, no admin rights.
C and C++ are linked statically, so the program needs no compiler DLLs to run.
"""
from __future__ import annotations

import codecs
import contextvars
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import config
from . import Tool

RUN_TIMEOUT_S = 20
MAX_TIMEOUT_S = 60
COMPILE_TIMEOUT_S = 120          # the first GCC/javac run of the day reads a lot from disk
MAX_STDOUT = 4000                # what the model gets back
MAX_STDERR = 2500
MAX_SHOWN = 100_000              # what the student is shown live
FLUSH_EVERY_S = 0.1
ALIASES = {"python": "python", "py": "python", "python3": "python", "c": "c", "cpp": "cpp", "c++": "cpp",
           "cxx": "cpp", "cc": "cpp", "java": "java"}
NAMES = {"python": "Python", "c": "C", "cpp": "C++", "java": "Java"}
TOOLS_DIR = config.ROOT / "tools"
RUNS_DIR = config.ROOT / "data" / "runs"                 # programs opened in a console window
CONSOLE_RUNNER = Path(__file__).with_name("console_runner.py")
_ENV_KEEP = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC", "USERPROFILE", "HOMEDRIVE",
             "HOMEPATH", "HOME", "APPDATA", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
             "OS", "LANG", "LC_ALL", "SYSTEMDRIVE", "PROGRAMDATA"}
# Linked into every C/C++ program: stdout to a pipe is fully buffered, so without this nothing would show
# until the program ends (Windows has no line buffering: _IOLBF means full buffering there).
_UNBUFFERED_C = ("#include <stdio.h>\n"
                 "__attribute__((constructor)) static void alice_unbuffered_stdout(void) "
                 "{ setvbuf(stdout, NULL, _IONBF, 0); }\n")

Emit = Callable[[dict], None]
_live: contextvars.ContextVar[tuple[Emit, threading.Event | None] | None] = contextvars.ContextVar(
    "alice_live_output", default=None)
_run_ids = itertools.count(1)


@contextmanager
def live_output(emit: Emit | None, cancel: threading.Event | None = None):
    """Report the runs made inside this block (in this thread) to `emit`; `cancel` stops a running program."""
    token = _live.set((emit, cancel) if emit else None)
    try:
        yield
    finally:
        _live.reset(token)


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


def console_available() -> bool:
    return sys.platform == "win32"


def _env(bin_dir: str | None = None, **extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP}
    if bin_dir:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def _java_class(code: str) -> str:
    m = (re.search(r"public\s+(?:final\s+|abstract\s+)*class\s+([A-Za-z_]\w*)", code)
         or re.search(r"class\s+([A-Za-z_]\w*)[^{]*\{(?:(?!class\s).)*static\s+void\s+main", code, re.S))
    return m.group(1) if m else "Main"


@dataclass
class Plan:
    """How to build and run one program in `folder` (the source is already written there)."""
    run: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    build: list[str] | None = None
    error: str | None = None                       # no toolchain / unsupported language


def prepare(lang: str, code: str, folder: Path, python_exe: str | None = None) -> Plan:
    if lang == "python":
        (folder / "main.py").write_text(code, encoding="utf-8")
        return Plan([python_exe or sys.executable, "-I", "-u", "-X", "utf8", "main.py"],
                    _env(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg"))
    if lang in ("c", "cpp"):
        compiler = find_program("gcc" if lang == "c" else "g++")
        if not compiler:
            return Plan(error=f"error: no {NAMES[lang]} compiler found. Run `python scripts/setup_toolchains.py` "
                              "once (portable GCC into ALICE's tools/ folder).")
        src = "main.c" if lang == "c" else "main.cpp"
        (folder / src).write_text(code, encoding="utf-8")
        (folder / "alice_unbuffered.c").write_text(_UNBUFFERED_C, encoding="utf-8")
        std = ["-std=c17"] if lang == "c" else ["-std=c++20"]
        exe = str(folder / ("prog.exe" if sys.platform == "win32" else "prog"))
        return Plan([exe], _env(str(Path(compiler).parent)),
                    [compiler, *std, "-O2", "-Wall", "-Wextra", "-static", "-o", exe, src, "alice_unbuffered.c", "-lm"])
    javac, java = find_program("javac"), find_program("java")
    if not (javac and java):
        return Plan(error="error: no Java compiler (JDK) found. Run `python scripts/setup_toolchains.py` once "
                          "(portable OpenJDK into ALICE's tools/ folder).")
    cls = _java_class(code)
    (folder / f"{cls}.java").write_text(code, encoding="utf-8")
    return Plan([java, "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8", "-cp", ".", cls],
                _env(str(Path(javac).parent), JAVA_HOME=str(Path(javac).parents[1])),
                [javac, "-encoding", "UTF-8", "-d", ".", f"{cls}.java"])


class _Shown:
    """Output on its way to the student: batched (one event per ~0.1 s, not per write) and capped."""

    def __init__(self, emit: Emit | None, run: int):
        self.emit, self.run = emit, run
        self.pending: list[tuple[str, str]] = []
        self.sent = 0
        self.lock = threading.Lock()

    def add(self, stream: str, text: str) -> None:
        if self.emit and text:
            with self.lock:
                self.pending.append((stream, text))

    def flush(self) -> None:
        if not self.emit:
            return
        with self.lock:
            pending, self.pending = self.pending, []
        merged: list[list[str]] = []
        for stream, text in pending:                 # neighbouring chunks of one stream become one event
            if merged and merged[-1][0] == stream:
                merged[-1][1] += text
            else:
                merged.append([stream, text])
        for stream, text in merged:
            if self.sent >= MAX_SHOWN:
                return
            if self.sent + len(text) > MAX_SHOWN:
                text = text[:MAX_SHOWN - self.sent] + f"\n[... only the first {MAX_SHOWN:,} characters are shown]\n"
            self.sent += len(text)
            self.emit({"event": "out", "run": self.run, "stream": stream, "text": text})


def _stream(cmd: list[str], cwd: str, env: dict, timeout: float, stdin: str, shown: _Shown,
            cancel: threading.Event | None) -> tuple[int | None, str, str, float, str | None]:
    """Run to completion, passing output on as it comes. (exit code, stdout, stderr, seconds, why it was
    stopped: 'timeout' / 'stopped' / None)."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    collected = {"stdout": [], "stderr": []}

    def read(pipe, name):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        carry = ""                                     # a "\r" whose "\n" may come in the next chunk
        while chunk := pipe.read1(65536):
            text, carry = carry + decoder.decode(chunk), ""
            if text.endswith("\r"):
                text, carry = text[:-1], "\r"
            text = text.replace("\r\n", "\n")          # Windows programs write \r\n
            collected[name].append(text)
            shown.add(name, text)
        rest = (carry + decoder.decode(b"", final=True)).replace("\r\n", "\n")
        collected[name].append(rest)
        shown.add(name, rest)

    def write():
        try:
            p.stdin.write(stdin.encode("utf-8"))
            p.stdin.close()                                # end of input: a program reading more gets EOF
        except OSError:                                    # it exited without reading everything
            pass

    threads = [threading.Thread(target=read, args=(p.stdout, "stdout"), daemon=True),
               threading.Thread(target=read, args=(p.stderr, "stderr"), daemon=True),
               threading.Thread(target=write, daemon=True)]
    for t in threads:
        t.start()
    t0, why = time.monotonic(), None
    while p.poll() is None:
        if time.monotonic() - t0 > timeout:
            why = "timeout"
        elif cancel is not None and cancel.is_set():
            why = "stopped"
        if why:
            p.kill()
            break
        time.sleep(0.05)
        shown.flush()
    p.wait()
    seconds = time.monotonic() - t0
    for t in threads[:2]:
        t.join(timeout=5)
    shown.flush()
    return p.returncode, "".join(collected["stdout"]), "".join(collected["stderr"]), seconds, why


def _result(code: int, out: str, err: str, seconds: float, notes: list[str]) -> str:
    out, err = out.rstrip(), err.rstrip()
    parts = [f"exit code {code}" + (f" ({seconds:.1f} s)" if seconds >= 1 else "")] + notes
    if out:
        parts.append(out[:MAX_STDOUT] + (" ...[stdout truncated]" if len(out) > MAX_STDOUT else ""))
    if err:
        parts.append("stderr:\n" + (err if len(err) <= MAX_STDERR else "...\n" + err[-MAX_STDERR:]))
    if not out and not err:
        parts.append("no output (print the results)")
    return "\n".join(parts)


def run_code(language: str, code: str, stdin: str = "", timeout: int = RUN_TIMEOUT_S,
             python_exe: str | None = None, emit: Emit | None = None,
             cancel: threading.Event | None = None) -> str:
    """Build and run; returns the report for the model. Progress goes to `emit`, or else to the
    `live_output` callback of the current turn."""
    if emit is None and (live := _live.get()):
        emit, cancel = live[0], cancel or live[1]
    lang = ALIASES.get(str(language).strip().lower())
    run = next(_run_ids)
    if emit:
        emit({"event": "start", "run": run, "language": lang or str(language), "title": NAMES.get(lang, str(language)),
              "code": code, "stdin": stdin or ""})

    def end(status: str, ok: bool, result: str) -> str:
        if emit:
            emit({"event": "end", "run": run, "status": status, "ok": ok})
        return result

    if lang is None:
        msg = f"error: unsupported language '{language}' (use python, c, cpp or java)"
        if emit:
            emit({"event": "out", "run": run, "stream": "stderr", "text": msg + "\n"})
        return end("not run", False, msg)
    timeout = max(1, min(int(timeout or RUN_TIMEOUT_S), MAX_TIMEOUT_S))
    shown = _Shown(emit, run)
    with tempfile.TemporaryDirectory(prefix="alice_run_") as tmp:
        plan = prepare(lang, code, Path(tmp), python_exe)
        if plan.error:
            shown.add("stderr", plan.error + "\n")
            shown.flush()
            return end("not run", False, plan.error)
        notes: list[str] = []
        try:
            if plan.build:
                if emit:
                    emit({"event": "phase", "run": run, "text": "compiling"})
                c = subprocess.run(plan.build, cwd=tmp, env=plan.env, stdin=subprocess.DEVNULL, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=COMPILE_TIMEOUT_S,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                messages = (c.stderr or c.stdout).strip()
                if c.returncode != 0:
                    shown.add("stderr", messages + "\n")
                    shown.flush()
                    return end("compile error", False, "compile error:\n" + messages[-MAX_STDERR:])
                if messages:
                    notes.append("compiler warnings:\n" + messages[-1500:])
                    shown.add("stderr", messages + "\n")
            if emit:
                emit({"event": "phase", "run": run, "text": "running"})
            code_, out, err, seconds, why = _stream(plan.run, tmp, plan.env, timeout, stdin or "", shown, cancel)
        except subprocess.TimeoutExpired as e:                 # the compiler
            return end("compile timed out", False, f"error: timed out after {e.timeout:.0f} s while compiling")
        except OSError as e:
            shown.add("stderr", f"could not start it: {e}\n")
            shown.flush()
            return end("not run", False, f"error: could not start it: {e}")
    if why == "timeout":
        return end(f"stopped after {timeout} s (time limit)", False,
                   f"error: timed out after {timeout} s while running (an infinite loop, a program waiting for "
                   f"more input than stdin gave it, or just too slow?)\n" + _result(-1, out, err, seconds, notes))
    if why == "stopped":
        return end("stopped by the student", False, "error: the student stopped the program\n"
                   + _result(-1, out, err, seconds, notes))
    return end(f"exit code {code_} · {seconds:.1f} s", code_ == 0, _result(code_, out, err, seconds, notes))


def open_in_console(language: str, code: str, python_exe: str | None = None) -> str:
    """Build and run the program in a new console window (Windows) where the student can type into it.
    Returns a short message for the chat."""
    lang = ALIASES.get(str(language).strip().lower())
    if lang is None:
        return f"can't run {language} code (Python, C, C++ and Java work)"
    if not console_available():
        return "opening a console window only works on Windows; use Run instead"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for old in RUNS_DIR.iterdir():                     # yesterday's programs
        try:
            if old.is_dir() and time.time() - old.stat().st_mtime > 86400:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    folder = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=RUNS_DIR))
    plan = prepare(lang, code, folder, python_exe)
    if plan.error:
        return plan.error
    (folder / "plan.json").write_text(json.dumps({"title": f"ALICE - {NAMES[lang]} program", "build": plan.build,
                                                  "run": plan.run, "cwd": str(folder)}), encoding="utf-8")
    subprocess.Popen([sys.executable, "-I", str(CONSOLE_RUNNER), str(folder / "plan.json")], cwd=folder, env=plan.env,
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    return f"opened the {NAMES[lang]} program in a console window"


def make_tool(python_exe: str | None = None) -> Tool:
    langs = [k for k, ok in available().items() if ok]
    return Tool(
        name="run_code",
        description=("Run a short program and get its output. Languages: " + ", ".join(langs) + ". Use it to "
                     "test or verify code, or for calculations too complex for 'calculate'. Print the results. "
                     "If the program reads input (input(), scanf, cin, Scanner), give that input in 'stdin' - "
                     "it cannot type into a prompt. Fresh temporary folder, 20 s time limit; Python has the "
                     "standard library and numpy. The student must approve each run and sees its output."),
        parameters={"type": "object",
                    "properties": {"language": {"type": "string", "enum": ["python", "c", "cpp", "java"]},
                                   "code": {"type": "string", "description": "the complete program"},
                                   "stdin": {"type": "string", "description": "text typed into the program, if it reads input"}},
                    "required": ["language", "code"]},
        run=lambda language="python", code="", stdin="": run_code(language, code, stdin, python_exe=python_exe),
        needs_confirmation=True,
    )
