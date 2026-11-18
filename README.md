# ALICE

A locally-run AI mentor with a personality, focused on **coding, academics and research** for a CS/AI student. Bachelor thesis project.

Status: Phases 0-3 (benchmarks, chat CLI, tools, paper library). See [docs/DEVLOG.md](docs/DEVLOG.md) for the running notes, roadmap, benchmarks and decisions.

## Setup (Windows)

1. llama.cpp CUDA build in `tools/llama.cpp/` (see DEVLOG, Phase 0) and the models:
   `bash scripts/download_models.sh` (Qwen3.5-9B Q4_K_M 5.7 GB + Qwen3-Embedding-0.6B 0.6 GB)
2. Python 3.11+:
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -e ".[dev]"
   ```

## Configuration and hardware upgrades

All hardware-specific settings live in `alice.toml` (not committed). Start from the example:
`copy alice.example.toml alice.toml`. The defaults target a constrained consumer GPU setup
(Qwen3.5-9B at 16k context, embedder on the CPU). To upgrade:

1. Download the bigger model into `models/` and add a profile, e.g. `[models.big]` with `gguf`, `ctx`,
   `ngl`, `extra_args`; set `[chat] profile = "big"` (or run `alice --model big`).
2. With spare VRAM / unified memory, set `[embedding] device = "gpu"`.
3. If you change the embedding model: `alice library --rebuild`, then
   `python scripts/calibrate_retrieval.py` and copy the suggested thresholds into `[embedding]`.
   ALICE refuses to mix embedding models in one library and tells you when a rebuild is needed.

Works on Windows, macOS and Linux (llama.cpp's `llama-server`; `.exe` is added automatically on Windows).
The paper library is exact vector search + SQLite FTS5; `scripts/scale_bench.py` measured ~19 ms per
search at 100k chunks (~1,000 papers) and ~185 ms at 1M chunks (~10,000 papers).

## Use

```
.venv\Scripts\alice tui                   # chat box (full-screen terminal UI); starts the model automatically
.venv\Scripts\alice                       # line-based chat (same options)
.venv\Scripts\alice --think               # thinking mode on (slower, best at maths/logic)
.venv\Scripts\alice --workspace C:\code\proj   # let ALICE read files in that folder (default: current dir)
.venv\Scripts\alice --offline             # no internet tools (arXiv)
.venv\Scripts\alice ingest papers\        # add PDFs / .md / .txt to your library
.venv\Scripts\alice library               # list library documents (--remove ID, --retitle ID TITLE)
```

Chat box keys: Enter send · Esc stop a reply · Ctrl+T thinking on/off · Ctrl+R new conversation ·
Ctrl+L clear the screen · Ctrl+Q quit. Use Windows Terminal or the VS Code terminal (the old console
window renders it poorly). In the line-based chat, Ctrl+C stops a reply.
Both understand `/think on|off`, `/tools`, `/reset`, `/help`, `/quit`.

Tools: `calculate`, `run_python` (asks before running), `list_files` / `read_file` / `search_text` (read-only, workspace only), `arxiv_search` / `arxiv_paper` (internet), and `search_library` / `list_library` when your library has documents. Answers from the library cite passages as `[L1]`, `[L2]`; ALICE lists the sources after the reply and warns about any citation that was not actually retrieved.

Questions that name a paper by nickname ("the Transformer paper", "the ResNet paper") are searched within
that paper. With `--auto-arxiv` (or `/arxiv auto on`, or `[arxiv] auto_lookup = true`), questions about
papers/methods that aren't in your library trigger an arXiv search for the name before ALICE answers.

Privacy: everything runs locally except the arXiv tools, which send the search text to arxiv.org. Sessions are saved to `logs/transcripts/` and the library to `data/library/`; both are gitignored.

## Tests and evaluations

```
.venv\Scripts\python -m pytest
.\scripts\bench.ps1 -Model Qwen3.5-9B-Q4_K_M                    # speed (llama-bench)
python scripts\run_prompts.py Qwen3.5-9B-Q4_K_M                   # Phase 0 prompt set
.venv\Scripts\python scripts\phase1_check.py                     # numeric accuracy + opener A/B
set ALICE_LIBRARY=data\eval_library
.venv\Scripts\alice ingest data\eval_papers                       # 5 arXiv papers (download: see DEVLOG)
.venv\Scripts\python scripts\rag_eval.py                         # retrieval + grounded-answer eval
.venv\Scripts\python scripts\scale_bench.py                      # vector search at 10k-1M chunks
.venv\Scripts\python scripts\calibrate_retrieval.py --set 2026   # thresholds for the embedding model
```
