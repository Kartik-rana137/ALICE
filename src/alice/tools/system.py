"""Local information tools: date arithmetic, the computer's setup, and the workspace's git repository.

All read-only and local; nothing leaves the machine (unless the student escalates a turn to Claude, whose
preview shows the tool output first). The current local time itself needs no tool: it travels with every
message in the turn note (persona.turn_note).
"""
from __future__ import annotations

import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import Tool

# ----- dates
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _today() -> datetime.date:                 # patched in tests
    return datetime.date.today()


def _parse_date(text: str | None, today: datetime.date) -> datetime.date:
    s = (text or "today").strip()
    low = s.lower()
    if low in ("today", "now"):
        return today
    if low == "tomorrow":
        return today + datetime.timedelta(days=1)
    if low == "yesterday":
        return today - datetime.timedelta(days=1)
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        pass
    plain = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s, flags=re.I)          # "12th December" -> "12 December"
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.datetime.strptime(plain, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"can't read the date '{s}'; use YYYY-MM-DD, 'today', 'tomorrow' or e.g. '12 December 2026'")


def _describe(d: datetime.date) -> str:
    return f"{d.isoformat()} ({_WEEKDAYS[d.weekday()]}, ISO week {d.isocalendar()[1]})"


def _weekdays_between(a: datetime.date, b: datetime.date) -> int:
    """Mon-Fri days in [a, b) (or (b, a] when b < a, negative)."""
    sign = 1 if b >= a else -1
    lo, hi = (a, b) if sign > 0 else (b, a)
    full_weeks, rest = divmod((hi - lo).days, 7)
    count = full_weeks * 5 + sum(1 for i in range(rest) if (lo + datetime.timedelta(days=full_weeks * 7 + i)).weekday() < 5)
    return sign * count


def date_calc(start: str = "today", end: str | None = None, add_days: int | None = None) -> str:
    try:
        today = _today()
        a = _parse_date(start, today)
        lines = [f"{start or 'today'}: {_describe(a)}"]
        if end:
            b = _parse_date(end, today)
            days = (b - a).days
            weeks, rem = divmod(abs(days), 7)
            lines.append(f"from {a.isoformat()} to {_describe(b)}: {_plural(days, 'day')} "
                         f"({'-' if days < 0 else ''}{_plural(weeks, 'week')} and {_plural(rem, 'day')}), "
                         f"{_weekdays_between(a, b)} of them weekdays (Mon-Fri, counting the start day, not the end)")
        if add_days is not None:
            r = a + datetime.timedelta(days=int(add_days))
            lines.append(f"{a.isoformat()} {'+' if int(add_days) >= 0 else '-'} {_plural(abs(int(add_days)), 'day')} "
                         f"= {_describe(r)}")
        lines.append(f"(today is {_describe(today)})")
        return "\n".join(lines)
    except (ValueError, OverflowError) as e:
        return f"error: {e}"


DATE_TOOL = Tool(
    "date_calc",
    "Date arithmetic and calendar facts: the weekday and ISO week of a date, the number of days (and weekdays) "
    "between two dates, or the date N days before/after a date. Use it for deadlines and schedules instead of "
    "counting days yourself. Dates as YYYY-MM-DD, 'today' or 'tomorrow'.",
    {"type": "object", "properties": {
        "start": {"type": "string", "description": "a date, default 'today'"},
        "end": {"type": "string", "description": "optional second date: days from start to end"},
        "add_days": {"type": "integer", "description": "optional: days to add to start (negative to subtract)"}}},
    date_calc)


# ----- the computer's setup
# `<name> --version` is only ever run for these well-known developer tools (never a program the model names freely).
VERSION_ARGS = {
    "python": ["--version"], "python3": ["--version"], "py": ["--version"], "pip": ["--version"],
    "git": ["--version"], "gcc": ["--version"], "g++": ["--version"], "clang": ["--version"],
    "cmake": ["--version"], "make": ["--version"], "ninja": ["--version"], "gdb": ["--version"],
    "node": ["--version"], "npm": ["--version"], "deno": ["--version"], "tsc": ["--version"],
    "java": ["-version"], "javac": ["-version"], "rustc": ["--version"], "cargo": ["--version"],
    "go": ["version"], "dotnet": ["--version"], "nvcc": ["--version"], "conda": ["--version"],
    "uv": ["--version"], "docker": ["--version"], "julia": ["--version"], "ruby": ["--version"],
}


def _ram() -> tuple[int, int] | None:
    """(total, available) bytes, or None."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemoryStatus()
            st.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys, st.ullAvailPhys
        elif Path("/proc/meminfo").exists():
            info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
            kb = lambda k: int(info[k].strip().split()[0]) * 1024  # noqa: E731
            return kb("MemTotal"), kb("MemAvailable")
        elif sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                       timeout=5).stdout.strip())
            return total, 0
    except Exception:
        return None
    return None


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or "") + (r.stderr or "")
    return out.strip() or None


def _gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit():
            gpus.append(f"{parts[0]}: {int(parts[1]) / 1024:.1f} GB VRAM, {int(parts[2]) / 1024:.1f} GB in use "
                        f"(including ALICE's own model), driver {parts[3]}")
    return gpus


def _cpu_name() -> str:
    """The marketing name ("Intel(R) Core(TM) ...") where the OS has it, else platform.processor()."""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if abs(n) == 1 else 's'}"


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB"


def system_info(programs: list[str] | None = None, workspace: Path | None = None) -> str:
    lines = [f"OS: {platform.system()} {platform.release()} ({platform.version()}), {platform.machine()}",
             f"Python running ALICE: {platform.python_version()}"]
    lines.append(f"CPU: {_cpu_name()}, {os.cpu_count()} logical cores")
    ram = _ram()
    if ram:
        lines.append(f"RAM: {_gb(ram[0])} total" + (f", {_gb(ram[1])} available" if ram[1] else ""))
    gpus = _gpus()
    lines += [f"GPU: {g}" for g in gpus] or ["GPU: no NVIDIA GPU found (nvidia-smi not available)"]
    try:
        du = shutil.disk_usage(workspace or Path.cwd())
        lines.append(f"Disk (workspace drive): {_gb(du.free)} free of {_gb(du.total)}")
    except OSError:
        pass
    for name in programs or []:
        name = str(name).strip().lower()
        if name not in VERSION_ARGS:
            lines.append(f"{name}: not checked (only common developer tools: {', '.join(sorted(VERSION_ARGS))})")
            continue
        exe = shutil.which(name)
        if not exe:
            lines.append(f"{name}: not found on PATH")
            continue
        out = _run([exe, *VERSION_ARGS[name]])
        lines.append(f"{name}: {(out or 'no version output').splitlines()[0][:150]}")
    return "\n".join(lines)


def make_system_tool(workspace: Path | None = None) -> Tool:
    return Tool(
        "system_info",
        "This computer's setup: OS, Python version, CPU, RAM, NVIDIA GPU and VRAM in use, free disk space, and "
        "optionally the installed versions of developer tools (python, pip, git, gcc, g++, clang, cmake, node, "
        "npm, java, javac, rustc, cargo, go, dotnet, nvcc, conda, uv, docker, ...). Use it for questions about the "
        "student's machine or environment (will a model fit in VRAM? which Python/compiler do I have?) instead "
        "of guessing.",
        {"type": "object", "properties": {"programs": {
            "type": "array", "items": {"type": "string"},
            "description": "developer tools whose versions to check, e.g. ['python', 'git', 'nvcc']"}}},
        lambda programs=None: system_info(programs, workspace))


# ----- git (read-only)
_REF = re.compile(r"^[\w./~^@{}-]{1,100}$")


class GitReader:
    """Read-only git commands in the workspace. Repository config cannot run programs: no external diff or
    textconv drivers, no fsmonitor hook, no pager."""

    SAFE = ["-c", "core.fsmonitor=false", "-c", "core.pager=cat", "-c", "color.ui=false"]

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _git(self, *args: str) -> str:
        if not shutil.which("git"):
            return "error: git is not installed (not on PATH)"
        try:
            r = subprocess.run(["git", "-C", str(self.root), *self.SAFE, *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20, stdin=subprocess.DEVNULL,
                               env={**os.environ, "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"})
        except subprocess.TimeoutExpired:
            return "error: git took too long"
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip()
            if "not a git repository" in msg:
                return f"error: the workspace ({self.root.name}) is not a git repository"
            return f"error: {msg[:500]}"
        return r.stdout.strip() or "(no output)"

    def _path(self, path: str | None) -> list[str]:
        if not path:
            return []
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"'{path}' is outside the workspace")
        return ["--", str(p.relative_to(self.root).as_posix() or ".")]

    def run(self, action: str = "status", path: str | None = None, ref: str = "HEAD", n: int = 10) -> str:
        try:
            pathspec = self._path(path)
        except PermissionError as e:
            return f"error: {e}"
        if ref and (ref.startswith("-") or not _REF.match(ref)):
            return f"error: '{ref}' is not a valid commit or branch name"
        n = max(1, min(int(n), 50))
        if action == "status":
            return self._git("status", "--short", "--branch", *pathspec)
        if action == "diff":
            return self._git("diff", "--no-ext-diff", "--no-textconv", *pathspec)
        if action == "diff_staged":
            return self._git("diff", "--staged", "--no-ext-diff", "--no-textconv", *pathspec)
        if action == "log":
            return self._git("log", f"-n{n}", "--date=short", "--format=%h %ad %an: %s", ref, *pathspec)
        if action == "show":
            return self._git("show", "--no-ext-diff", "--no-textconv", "--stat", "--patch", ref, *pathspec)
        return "error: action must be one of status, diff, diff_staged, log, show"


def make_git_tool(root: str | Path) -> Tool:
    reader = GitReader(root)
    return Tool(
        "git",
        f"Read-only git in the student's workspace ({reader.root.name}): 'status' (changed files), 'diff' (unstaged "
        "changes), 'diff_staged', 'log' (recent commits), 'show' (one commit). Use it when the student asks about "
        "their changes, a bug they just introduced, or recent commits. It cannot modify anything.",
        {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["status", "diff", "diff_staged", "log", "show"]},
            "path": {"type": "string", "description": "optional file or folder (relative to the workspace)"},
            "ref": {"type": "string", "description": "commit/branch for log or show, default HEAD"},
            "n": {"type": "integer", "description": "commits for log, 1-50, default 10"}},
         "required": ["action"]},
        reader.run)
