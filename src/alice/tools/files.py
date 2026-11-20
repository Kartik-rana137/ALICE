"""Read-only file tools, confined to one workspace directory."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from . import Tool

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".idea", ".vscode", "dist", "build", ".tox", ".eggs"}
MAX_LIST = 300
MAX_LINES = 400
MAX_FILE_BYTES = 2_000_000
MAX_MATCHES = 80


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"'{rel}' is outside the workspace ({self.root})")
        return p

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    def walk(self, start: Path):
        """Files under `start`, sorted, pruning SKIP_DIRS (without descending into them)."""
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for f in sorted(filenames):
                yield Path(dirpath) / f

    def list_files(self, path: str = ".", pattern: str = "*") -> str:
        try:
            start = self.resolve(path)
        except PermissionError as e:
            return f"error: {e}"
        if not start.is_dir():
            return f"error: '{path}' is not a directory"
        out = []
        for p in self.walk(start):
            if fnmatch.fnmatch(p.name, pattern):
                out.append(f"{self.rel(p)}  ({p.stat().st_size} bytes)")
                if len(out) >= MAX_LIST:
                    out.append(f"... (stopped at {MAX_LIST} files; narrow the path or pattern)")
                    break
        return "\n".join(out) or "no matching files"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        try:
            p = self.resolve(path)
        except PermissionError as e:
            return f"error: {e}"
        if not p.is_file():
            return f"error: '{path}' is not a file"
        if p.stat().st_size > MAX_FILE_BYTES:
            return f"error: file is too large ({p.stat().st_size} bytes)"
        raw = p.read_bytes()
        if b"\x00" in raw[:4096]:
            return "error: binary file"
        lines = raw.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + MAX_LINES - 1, start + MAX_LINES - 1)
        body = "\n".join(f"{i:5}| {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n... ({len(lines) - end} more lines; read again with start_line={end + 1})" if end < len(lines) else ""
        return f"{self.rel(p)} lines {start}-{end} of {len(lines)}\n{body}{more}"

    def search_text(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        try:
            start = self.resolve(path)
            rx = re.compile(pattern)
        except PermissionError as e:
            return f"error: {e}"
        except re.error as e:
            return f"error: bad regex: {e}"
        hits = []
        files = [start] if start.is_file() else self.walk(start)
        for p in files:
            if not fnmatch.fnmatch(p.name, glob) or p.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self.rel(p)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_MATCHES:
                        return "\n".join(hits) + f"\n... (stopped at {MAX_MATCHES} matches)"
        return "\n".join(hits) or "no matches"


    def read_document(self, path: str, start: int = 1, end: int | None = None) -> str:
        """PDF / Word / PowerPoint (and text) files, a page / slide / part range at a time."""
        from ..documents import DocumentError, read
        try:
            doc = read(self.resolve(path))
        except (PermissionError, DocumentError) as e:
            return f"error: {e}"
        n = len(doc.units)
        start = max(1, min(int(start), n or 1))
        if end is None:                                  # as many units as fit in one tool result
            end, size = start, 0
            while end <= n and size + len(doc.units[end - 1].text) <= 5500:
                size += len(doc.units[end - 1].text)
                end += 1
            end = max(start, end - 1)
        end = min(int(end), n)
        more = f"\n... ({n - end} more {doc.unit_name}s; read again with start={end + 1})" if end < n else ""
        return f"{doc.describe()}, {doc.unit_name}s {start}-{end}\n\n{doc.text(start, end)}{more}"


def make_tools(root: str | Path) -> list[Tool]:
    ws = Workspace(root)
    path_prop = {"type": "string", "description": "path relative to the workspace root ('.' for the root)"}
    return [
        Tool("list_files", f"List files in the user's workspace ({ws.root.name}), recursively. Read-only.",
             {"type": "object", "properties": {"path": path_prop,
                                               "pattern": {"type": "string", "description": "filename glob, e.g. '*.py'"}}},
             ws.list_files),
        Tool("read_file", "Read a text file from the workspace with line numbers (up to 400 lines per call).",
             {"type": "object", "properties": {"path": path_prop,
                                               "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
              "required": ["path"]},
             ws.read_file),
        Tool("read_document", "Read a PDF, Word (.docx) or PowerPoint (.pptx) file from the workspace as text, "
             "a range of pages / slides / parts at a time (labelled, so you can say where things are).",
             {"type": "object", "properties": {"path": path_prop,
                                               "start": {"type": "integer", "description": "first page/slide/part, default 1"},
                                               "end": {"type": "integer", "description": "last one (default: as many as fit)"}},
              "required": ["path"]},
             ws.read_document),
        Tool("search_text", "Search workspace files for a regular expression; returns file:line: text.",
             {"type": "object", "properties": {"pattern": {"type": "string"}, "path": path_prop,
                                               "glob": {"type": "string", "description": "filename glob, e.g. '*.py'"}},
              "required": ["pattern"]},
             ws.search_text),
    ]
