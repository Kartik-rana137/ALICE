# ALICE

A locally-run AI mentor with a personality, focused on **coding, academics and research** for a CS/AI student. Bachelor thesis project.

Status: Phases 0-3 (benchmarks, chat CLI, tools, paper library). See [docs/DEVLOG.md](docs/DEVLOG.md) for the running notes, roadmap, benchmarks and decisions.

## Setup (Windows)

1. llama.cpp CUDA build in `tools/llama.cpp/` (see DEVLOG, Phase 0) and the models:
   `bash scripts/download_models.sh` (Qwen3.5-9B Q4_K_M 5.7 GB + Qwen3-Embedding-0.6B 0.6 GB)
2. Python 3.10+:
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -e ".[dev]"
   ```

## Use

```
.venv\Scripts\alice                       # chat; starts llama-server automatically
.venv\Scripts\alice --think               # thinking mode on (slower, best at maths/logic)
.venv\Scripts\alice --workspace C:\code\proj   # let ALICE read files in that folder (default: current dir)
.venv\Scripts\alice --offline             # no internet tools (arXiv)
.venv\Scripts\alice ingest papers\        # add PDFs / .md / .txt to your library
.venv\Scripts\alice library               # list library documents (--remove ID, --retitle ID TITLE)
```

In the chat: `/think on|off`, `/tools`, `/reset`, `/help`, `/quit`; Ctrl+C stops a reply.

Tools: `calculate`, `run_python` (asks before running), `list_files` / `read_file` / `search_text` (read-only, workspace only), `arxiv_search` / `arxiv_paper` (internet), and `search_library` / `list_library` when your library has documents. Answers from the library cite passages as `[L1]`, `[L2]`; ALICE lists the sources after the reply and warns about any citation that was not actually retrieved.

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
```
