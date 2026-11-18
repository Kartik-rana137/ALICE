"""`alice ingest ...` and `alice library ...`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config
from .library import SUPPORTED, Library, LibraryError, LlamaEmbedder
from .llm import LLMError, server_healthy


def _files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED and p.exists():
            out.append(p)
        else:
            print(f"skip (not found or unsupported): {p}")
    return out


def ingest(args) -> int:
    files = _files(args.paths)
    if not files:
        print("nothing to ingest (supported: " + ", ".join(sorted(SUPPORTED)) + ")")
        return 1
    if not config.EMBED_GGUF.exists():
        print(f"embedding model missing: {config.EMBED_GGUF} (run scripts/download_models.sh)")
        return 1
    if args.title and len(files) > 1:
        print("--title only works with a single file")
        return 1
    # GPU if configured, or if it is free (the chat model is not running)
    gpu = config.EMBED_DEVICE == "gpu" or not server_healthy()
    lib = Library()
    emb = LlamaEmbedder(gpu=gpu)
    print(f"Embedding on {'GPU' if gpu else 'CPU (chat model is using the GPU)'} ...")
    t0 = time.time()
    try:
        for f in files:
            t = time.time()
            try:
                status, n, title = lib.add(f, emb, title=args.title, section_headers=not args.no_sections)
            except (LLMError, LibraryError) as e:
                print(f"error: {e}")
                return 1
            except Exception as e:                # unreadable/corrupt file: report and continue
                print(f"  failed  {f.name}: {type(e).__name__}: {e}")
                continue
            print(f"  {status:7} {f.name}: {title!r} ({n} chunks, {time.time() - t:.1f}s)")
    finally:
        emb.stop()
    print(f"done in {time.time() - t0:.0f}s; library has {len(lib.documents())} documents")
    return 0


def library(args) -> int:
    lib = Library()
    if args.remove is not None:
        print("removed" if lib.remove(args.remove) else f"no document with id {args.remove}")
        return 0
    if args.rebuild:
        gpu = config.EMBED_DEVICE == "gpu" or not server_healthy()
        emb = LlamaEmbedder(gpu=gpu)
        print(f"Rebuilding the index on {'GPU' if gpu else 'CPU'} ...")
        try:
            for title, status, n in lib.rebuild(emb):
                print(f"  {status:12} {title} ({n} chunks)")
        finally:
            emb.stop()
        return 0
    if args.retitle:
        doc_id, title = int(args.retitle[0]), " ".join(args.retitle[1:]).strip()
        print("renamed" if title and lib.set_title(doc_id, title) else "usage: alice library --retitle ID NEW TITLE")
        return 0
    docs = lib.documents()
    if not docs:
        print("library is empty; add papers with: alice ingest PATH")
        return 0
    for doc_id, title, pages, added, chunks, path in docs:
        print(f"{doc_id:4}  {title}  ({pages or '-'} pages, {chunks} chunks, added {added})")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="alice")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="add PDFs / .md / .txt files or folders to the library")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--title", help="override the detected title (single file only)")
    pi.add_argument("--no-sections", action="store_true", help="don't attach section headings to chunks "
                    "(ablation)")
    pl = sub.add_parser("library", help="list library documents")
    pl.add_argument("--remove", type=int, metavar="ID", help="remove a document by id")
    pl.add_argument("--retitle", nargs="+", metavar=("ID", "TITLE"), help="rename a document")
    pl.add_argument("--rebuild", action="store_true", help="re-ingest all documents with the current pipeline")
    args = ap.parse_args(argv)
    return ingest(args) if args.cmd == "ingest" else library(args)
