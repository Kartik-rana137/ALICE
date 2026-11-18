# ALICE Development Log

Running notes for the ALICE project (local, coding/academics/research-focused AI companion; intended as bachelor thesis, ~2030).
Newest entries at the bottom. Record decisions with their *why* — future-me writing the thesis will need them.

---

## 2026-11-18 — Project kickoff & feasibility assessment

### Goal
A locally-run conversational AI with a persona (optionally real-time voice, memory and an avatar), specialised for **coding, academic study and research assistance**.

### Hardware (baseline, "Gen 0")
| Part | Spec | Implication |
|---|---|---|
| GPU | Entry-level consumer GPU (small VRAM) | Hard ceiling on model size; shared between LLM, STT, TTS |
| CPU | Consumer multi-core CPU | CPU offload possible but slow |
| RAM | Moderate system RAM | Can hold larger quantized models for CPU/hybrid inference |
| OS | Windows | Prefer tools with Windows builds (llama.cpp, Ollama, faster-whisper, Piper) |

Rough VRAM budget: LLM ~3.5–4.5 GB (Q4, 4–8k ctx) + STT ~0.5 GB + TTS ~0 (CPU). Tight. Anything bigger → partial CPU offload → latency hit.

### What the task consists of
A conversational AI companion like this is not one model; it is a **pipeline**:

1. **LLM core** (reasoning, persona, tool use)
2. **Speech-to-text** (user voice → text)
3. **Text-to-speech** (low-latency, consistent voice)
4. **Orchestrator** (streaming, interruption handling, turn-taking, latency management)
5. **Memory** (short-term context, long-term retrieval/summaries)
6. **Tools/skills** — *this is where ALICE differs*: code execution sandbox, repo/file reading, paper/PDF ingestion (RAG), arXiv/Semantic Scholar search, LaTeX/notes, citation handling
7. **Avatar/presentation** (optional, later): Live2D/VRM, lip-sync
8. **Evaluation** — needed for thesis (latency, accuracy, human eval, ablations)

### Realistic scope per hardware tier
- **Now (small GPU):** 3–8B quantized model (e.g. Qwen2.5/3 class, Llama 3.x 8B, Phi-class) via llama.cpp/Ollama. Text chat + RAG + tool use feasible. Voice feasible with small Whisper + Piper/Kokoro (CPU). Expect ~15–35 tok/s on a 7B Q4 if it fits; 14B+ via CPU offload ≈ 2–5 tok/s (too slow for voice).
- **Upgrade tier (12–24 GB VRAM):** 14–32B coder/reasoning models, better TTS (voice cloning), fine-tuning with LoRA/QLoRA becomes realistic.
- **Thesis-time (4 yrs):** whatever hardware/models exist then; design the system **model-agnostic** so the LLM is a swappable component.

### Key decisions (initial, revisit freely)
- **Model-agnostic architecture**: LLM behind an OpenAI-compatible HTTP API (llama.cpp server / Ollama). *Why:* hardware will change, models improve fast.
- **Python** as main language. *Why:* ecosystem for STT/TTS/RAG.
- **Text-first, voice second, avatar last.** *Why:* core value for coding/research is reasoning + tools; voice/avatar are layers on top and eat VRAM.
- **Local-only by default**; no cloud dependency in the core loop.
- **Log everything measurable from day one** (latency, tokens/s, VRAM) so the thesis has data.

### Proposed roadmap
**Active focus (decided 2026-11-18): Phases 0-3 only.** Phases 4-8 are on the backburner (see "Scope guardrails" at the bottom).

| Phase | Deliverable | Notes |
|---|---|---|
| 0 | Repo, environment, benchmark baseline models on this GPU | Record tok/s, VRAM, quality on a small coding/academic prompt set |
| 1 | Text chat CLI with persona + streaming | Prompt/persona design, conversation memory |
| 2 | Tools: code sandbox, file/repo reader, web/paper search | Function-calling reliability is the main risk with small models |
| 3 | RAG over papers/notes (PDF ingestion, embeddings, citations) | Small embedding model on CPU |
| 4 | Voice in/out with interruption handling | faster-whisper + Piper/Kokoro; watch VRAM |
| 5 | Long-term memory (summaries, vector store, user profile) | |
| 6 | Avatar (optional) | VRM/Live2D + lip-sync |
| 7 | Fine-tuning / personalisation (needs GPU upgrade) | LoRA on coding/academic style, persona |
| 8 | Evaluation + thesis writing | Benchmarks (HumanEval-style, custom research-QA), human eval, ablations |

### Risks
- VRAM is the bottleneck; voice + LLM concurrently may not fit → may need sequential or CPU TTS/STT.
- Small models hallucinate citations/facts → RAG with grounded citations is essential, not optional.
- Tool-calling reliability on ≤8B models.
- Scope creep (avatar/streaming features) vs. thesis-worthy research contribution.

### Candidate thesis angles (decide by ~year 2–3)
- Latency/quality trade-offs of a fully local voice assistant on consumer hardware.
- Grounded research assistant: RAG + citation verification with small local models.
- Persona/consistency + domain specialisation via LoRA vs. prompting.
- Evaluation of local vs. cloud models for student coding/study workflows.

### Next steps
- [ ] Decide Python version/env manager (uv or conda) and set up env
- [ ] Install llama.cpp (CUDA build) or Ollama; benchmark 3–4 candidate models
- [ ] Create `benchmarks/` with a reproducible script + results log
- [ ] Draft persona/system prompt v0

### Open questions
(Persona, public-streaming and academic-domain questions resolved on 2026-11-18, see below.)

---

## 2026-11-18 — Persona & scope decisions

- **Persona:** mentor first, with a slight touch of sass. Teaches and guides rather than just answering; light teasing is fine, but never at the cost of clarity or correctness.
- **Scope:** personal assistant only, for now. No public streaming/performance.
- **Consequences:**
  - No need for chat-moderation, stream-overlay or audience-interaction features; drops a large chunk of the original feature set.
  - Avatar (Phase 6) becomes lower priority and purely optional.
  - Persona work should focus on pedagogy: asking guiding questions, explaining reasoning, admitting uncertainty, flagging unverified citations. Sass is a style layer on top.
  - Thesis angle "persona consistency / mentor-style tutoring vs. plain answering" is now a stronger candidate.
- **Why:** a mentor persona fits the coding/academics/research goal and is easier to evaluate (e.g. learning outcomes, answer quality) than entertainment value.
- **Revisit if:** public streaming becomes a goal (would reopen moderation, safety and latency requirements).

---

## 2026-11-18 — Academic domain decision

- **Domain:** Computer Science Engineering, specialization in AI/ML.
- **Consequences:**
  - Paper RAG (Phase 3) should target arXiv cs.AI / cs.LG / cs.CL / cs.CV, Semantic Scholar and Papers with Code; math-heavy PDFs mean the ingestion pipeline needs good LaTeX/equation handling.
  - Core coursework areas to support: algorithms and data structures, ML/DL, linear algebra/probability/optimisation, OS/networks/DBs.
  - Coding tools: Python-first sandbox with PyTorch/NumPy-aware help (shape errors, training-loop debugging); Jupyter notebook support is worth considering.
  - Evaluation: use code benchmarks (HumanEval/MBPP-style) plus a custom CS/AI-course question set; the thesis can plausibly be about ALICE being a study/research aid for AI students.
  - Bonus: the project itself is an AI-specialisation showcase (local inference, quantization, RAG, possibly LoRA fine-tuning).
- **Why:** focusing on one domain makes retrieval, tools and evaluation concrete instead of generic.

---

## 2026-11-18 — Phase 0 started: tooling + first Qwen benchmarks

### Setup
- llama.cpp **b12498**, prebuilt Windows **CUDA 12.4** x64 build (driver reports CUDA 13.3, so the 13.5 build was avoided). No compiling / CUDA toolkit needed.
- Layout (all gitignored): `tools/llama.cpp/` (binaries), `models/` (GGUFs).
- Models from `unsloth/*-GGUF` on Hugging Face.

### Qwen landscape (as of 2026-11-18)
- **Qwen3.5** (Feb/Mar 2026): dense small sizes 0.8B / 2B / 4B / 9B (+27B, 35B-A3B, 122B-A10B, 397B-A17B). Natively multimodal, Apache 2.0.
- **Qwen3.6** (Apr 2026): 27B, 35B-A3B (MoE, ~3B active params).
- **Qwen3.8** (Aug 2026): 27B and larger; no small dense sizes yet.
- Third-party reports say Qwen3.5 9B/4B are strong on general/agentic tasks but weakest on coding/reasoning relative to larger models — must verify on our own prompts.
- Interesting for this hardware: **35B-A3B MoE** keeps attention on GPU and experts in system RAM, only ~3B params active per token → may be usable speed despite 35B total size. To test.

### Benchmarks (llama-bench, -ngl 99, pp512 / tg128, dev GPU)
| Model | Quant | Size | Prompt t/s | Gen t/s | Fits fully on GPU |
|---|---|---|---|---|---|
| Qwen3.5-2B | Q4_K_M | 1.18 GiB | 718 | 120 | yes |
| Qwen3.5-4B | Q4_K_M | 2.54 GiB | 303 | 65 | yes |
| Qwen3.5-9B | Q4_K_M | 5.28 GiB | 162 | 39 | yes (ngl 99); with ngl 30 gen drops to 25 t/s |

Reproducible script: `scripts/bench.ps1`, results appended to `benchmarks/results/speed.csv`.
**Caveat:** llama-bench uses a tiny context. The 9B fits at ngl 99 here but leaves ~0.7 GiB for KV cache/compute buffers, so it may not hold at real context lengths (4k-8k). Needs a real-context test via llama-server.

Pending: Qwen3.6-35B-A3B IQ4_XS (17.7 GB) with CPU-expert offload.

### Alternatives to Qwen for coding (research, 2026-11-18)
Generic "best local coding LLM" blog results were dated (Mistral 7B, DeepSeek-Coder 6.7B, Phi-4-mini). Checked actual 2026 releases on Hugging Face instead:
- **Gemma 4** (Google, Apr-May 2026): E2B, E4B, 12B, 26B-A4B (MoE), 31B. Vendor LiveCodeBench v6: roughly tied with Qwen3.5 at each size (31B 80.0 vs Qwen3.5-27B 80.7; 26B-A4B 77.1 vs Qwen3.5-35B-A3B 74.6). One local Go test found Gemma 4 26B-A4B more consistent across seeds; Qwen3.5 looks stronger on agentic/repo tasks. -> **Chosen alternative.**
- Others noted, not tested yet: GLM-5.3-Flash (Aug 2026), IBM Granite 4.1 8B (Apr 2026), Ministral 3 8B/14B (Dec 2025), Nemotron-3-Nano-4B, gpt-oss-20b (Aug 2025, MoE).
- Candidates downloaded for comparison: Gemma 4 E4B Q4_K_M (5.0 GB), Gemma 4 26B-A4B UD-IQ4_XS (13.6 GB), Gemma 4 12B IQ4_XS (6.4 GB).
- Source caveat: third-party aggregator numbers disagree on thinking mode/quant; differences under ~1 point are noise. Our own prompt set decides.

### Housekeeping rule
Delete GGUFs that are ruled out after benchmarking; keep this log's table as the record.

### Idea: hybrid local + Claude escalation
Possible and compatible with the local-first principle. Design sketch:
- Local model handles everything by default; escalation to Claude is **opt-in, per-request**, never silent.
- Router decides when to escalate: (a) local model self-reports low confidence, (b) task class known to exceed local ability (long-context code refactors, hard proofs), (c) tool-call/validation failure after N retries, (d) user says "ask the big model".
- Privacy: show/confirm what is sent; strip personal data/paths; escalation log for the thesis.
- Cost: API key + budget cap; log tokens/cost per escalation.
- Thesis value: measure *what fraction of tasks* a small local model can handle and what routing policy minimises cloud calls at equal quality.
- Because the LLM sits behind an OpenAI-compatible interface, Claude is just another backend in the router. Build after Phase 2 (tools) once there is a baseline to route against.

---

## 2026-11-18 — Scope guardrails: focus on Phases 0-3

Decision: to keep the project manageable, the active scope is **Phases 0-3 only**.

**Active core**
- Phase 0: environment + model benchmarks (in progress)
- Phase 1: text chat CLI with mentor/sassy persona + streaming
- Phase 2: tools (code sandbox, file/repo reader, web/paper search)
- Phase 3: RAG over papers/notes with citations

**Backburner (not started, don't work on without revisiting this decision)**
- Phase 4 voice in/out
- Phase 5 long-term memory
- Phase 6 avatar
- Phase 7 fine-tuning / personalisation (needs GPU upgrade)
- Hybrid local + Claude escalation (build after Phase 2, once there is a local baseline)
- Phase 8 evaluation + thesis writing is *not* backburnered in spirit: measurements are collected from Phase 0 onwards.

**Rules**
- Each phase must be usable on its own before the next starts.
- Model benchmarking is capped: pick one default model at the end of Phase 0 and stop; re-evaluate only on a hardware upgrade or a major model release.
- Choose a thesis question by roughly year 2. Candidates: grounded paper RAG with citation checking; local-vs-Claude routing policy; mentor-style tutoring vs plain answering.
- Keep the LLM swappable behind an OpenAI-compatible API.

---

## 2026-11-18 — Phase 0 results: model comparison and default choice

### Setup
6 models, same 12 prompts (`benchmarks/prompts.json`: coding, academic, research-honesty, mentor persona), same v0 mentor system prompt, 8k context, thinking mode off, temperature 0.3, 1000-token cap. Raw replies: `benchmarks/results/quality/*.json`. Speed: `benchmarks/results/speed.csv`.

### Results
| Model | Gen t/s | Where it runs | Fake-paper honesty test | PyTorch shape fix | Replies cut off at cap | Persona ("just give me the code") |
|---|---|---|---|---|---|---|
| Qwen3.5-2B Q4_K_M | 122 | GPU | **fail**: fabricated an arXiv id + summary | ok | 4/12 | **handed over full code** |
| Qwen3.5-4B Q4_K_M | 64 | GPU | **unstable**: correct in run 1, fabricated a full summary in run 2 | **wrong** (32*14*14) | 3/12 | mostly refuses |
| Qwen3.5-9B Q4_K_M | 33 | GPU @8k ctx (fits) | pass | ok | 3/12 | refuses, guides |
| Gemma 4 E4B Q4_K_M | 57 | GPU | hedged ("not in my knowledge base"), no fabrication | ok | 3/12 | refuses, good sass |
| Gemma 4 26B-A4B IQ4_XS | 13 | MoE, 20 expert layers on CPU | pass | ok | 1/12 | refuses, best sass |
| Qwen3.6-35B-A3B IQ4_XS | 14 | MoE, 36 expert layers on CPU | pass | ok | 4/12 | refuses, gives pseudocode skeleton |

Whole 12-prompt suite wall time: 2B 74 s, E4B 113 s, 4B 123 s, 9B 240 s, 35B-A3B 629 s, 26B-A4B 712 s.

### Findings
- **Small models fabricate citations.** Both Qwen3.5 2B and 4B invented details for a paper that doesn't exist; the 4B flipped between runs at the same temperature. Confirms the risk in the plan: grounded RAG with citation checking (Phase 3) is mandatory, not optional.
- Everything >= 9B passed the honesty and shape-debug checks; the 4B-and-below tier is not reliable enough for the default.
- Persona prompt v0 works on every model except the 2B; Gemma 4 has the most natural mentor-with-sass tone.
- MoE models (13-15 t/s) are fine for text chat but ~2.5x slower than the 9B and use 14-18 GB of system RAM, which will compete with RAG embeddings. First request after loading is very slow (model page-in from disk: 70 s prompt eval, 168 s first reply for the 26B).
- Qwen 9B at 8k context fits fully in VRAM at ~33 t/s; llama-bench alone would have hidden whether it fit.

### Caveats (important for the thesis)
- One run per prompt per model, 12 prompts, temperature 0.3: **noisy**. The 4B result flipping proves run-to-run variance matters. A real evaluation needs multiple seeds and a larger set.
- Automated checks were crude regexes (they mislabelled the 26B's LaTeX complexity answer); failures were read by hand, passes were only spot-checked.
- Vendor/third-party claims (Qwen weakest at coding; Gemma ~ tied) were not contradicted, but our sample is too small to confirm them.

### Decision
- **Default model: Qwen3.5-9B Q4_K_M** (fully on GPU, ~33 t/s, passes all checks, whole suite in 4 min, leaves system RAM free).
- **Heavy-reasoning tier: Qwen3.6-35B-A3B IQ4_XS** with `--n-cpu-moe 36` (~14 t/s). Candidate for the local tier of a future local->Claude router. Chosen over Gemma 4 26B-A4B because it passed every check, processes prompts ~2x faster (186 vs 101 t/s), and is newer; it is close to a tie, though, and Gemma's persona and brevity were better.
- **Deleted** (unused): Qwen3.5-2B, Qwen3.5-4B, Gemma 4 E4B, Gemma 4 26B-A4B. All are public; re-download with `scripts/download_models.sh` (edit the file list) from `unsloth/<repo>-GGUF`:
  `Qwen3.5-2B-Q4_K_M`, `Qwen3.5-4B-Q4_K_M`, `gemma-4-E4B-it-Q4_K_M`, `gemma-4-26B-A4B-it-UD-IQ4_XS`.
- Per the guardrails: model benchmarking is now **capped**. Re-evaluate only on a hardware upgrade or a major model release. Gemma 4 26B-A4B remains the first alternative to try then.

### Next (Phase 1)
- [ ] Python project skeleton (uv or venv), config for model path / server flags
- [ ] llama-server launcher + OpenAI-compatible client wrapper
- [ ] Text chat CLI with persona prompt v1 (refine mentor + sass from the v0 results) + streaming
- [ ] Conversation memory within a session

---

## 2026-11-18 — Phase 1 started; 35B heavy tier removed

- **Removed Qwen3.6-35B-A3B** (17.7 GB) and its `heavy` profile (no longer used). Only **Qwen3.5-9B Q4_K_M** remains in `models/`.
  - Supersedes the "heavy-reasoning tier" bullet in the Phase 0 decision above. The 35B remains the first thing to bring back for harder questions or the local tier of a future local->Claude router; re-download line is (commented) in `scripts/download_models.sh`, run with `--n-cpu-moe 36` (~14 t/s).
- **Phase 1 skeleton built** (`src/alice/`, stdlib only, no runtime deps):
  - `config.py` model profiles/paths; `llm.py` llama-server launcher (reuses a running server) + streaming SSE client; `conversation.py` session memory with context trimming; `persona.py` system prompt **v1**; `cli.py` chat REPL (`alice`, `/reset`, `/quit`).
  - Persona v1 changes from v0 (driven by Phase 0 findings): explicit "never invent papers/authors/arXiv ids/URLs", "say when unsure", "correct wrong premises", word-count target (~250) to reduce the cut-off replies, and "drop the sass when the student is discouraged".
  - Tests: `tests/test_alice.py` (SSE parsing, context trimming, server command building).

### Phase 1 smoke test (real server, Qwen3.5-9B, persona v1)
- End-to-end CLI works: server auto-start (~20 s load), streaming output, reply timing, clean server shutdown on exit (no stray processes), 11 unit tests passing.
- Fake-paper prompt: correct (says it doesn't recognise the paper, asks for a link/arXiv id, no fabrication). 8.9 s.
- Homework prompt ("just give me the full code"): the first v1 wording was too permissive and the model pasted the full solution. Tightened persona (explicit "homework without an attempt -> skeleton/pseudocode + a question, no full solution"). Retest: tone and task-setting are right, **but the "skeleton" still leaks the whole solution with hint comments** — a known limitation of the 9B at following "leave blanks". Idea for later: post-check / second pass, or evaluate on a larger model after a hardware upgrade.
- Persona is a style/behaviour layer that needs a proper eval later (mentor vs plain answering is a candidate thesis question), not endless prompt tweaking now.

### Phase 1 TODO
- [x] Python skeleton, launcher, client, CLI, conversation memory, tests
- [ ] Use `/tokenize` for exact context accounting (currently a conservative chars/3 estimate)
- [ ] Save/load conversation transcripts (useful for thesis data and later memory)
- [ ] Decide Phase 1 "done" bar, then move to Phase 2 (tools: code sandbox, file reader, search)

---

## 2026-11-18 — Persona v2: solution first, then step-by-step guidance

- **Decision:** ALICE always gives the complete, correct solution up front (homework included), then walks through it step by step (what each part does and why, pitfalls), then ends with one check-your-understanding question or variation. Concept questions are explained directly. No more skeletons or withheld answers.
- **Supersedes:** the v1 "homework -> skeleton/hints" rule and the Phase 0 reading of "refuses to paste code" as the *good* persona behaviour. Honesty rules (no invented citations, admit uncertainty, correct wrong premises) and the ~250-word conciseness target are unchanged.
- **Why:** seeing the full answer first and learning from it works better than being gated; this also avoids the 9B's inability to write a skeleton without leaking the solution (see Phase 1 smoke test).
- **Smoke test (9B, `persona v2`):** binary search -> code, 5-step walkthrough, follow-up question about unsorted input. Bayes problem -> derivation, walkthrough, follow-up with prevalence 1/100. Format followed on both.
- **Issues seen:**
  - Arithmetic inconsistency: headline said "~1.96%" while its own derivation gave 0.01943 (1.94%). Small models slip on numeric answers -> tool use (code sandbox, Phase 2) should do calculations.
  - Sass was barely present in these two replies (tone reads neutral). Not changed yet; revisit if it matters.
  - The check question for binary search ("time complexity if not sorted?") is a bit vague as a learning prompt.
- **Thesis note:** "solution-first + guided walkthrough" vs. "hint-first (Socratic)" is a concrete, testable pedagogy comparison for the evaluation phase.

---

## 2026-11-18 — Persona v2.1: sass restored

- **Decision:** bring the sass back, present but not over the top.
- **Rule in the prompt:** open every reply with one short, dry, playful remark, then go straight to content; optional second quip near the end, never more; tease the situation not the student's ability; no scolding/lecturing, pet names, emojis or roleplay; no jokes inside code, derivations or numbered steps; drop the sass and be kind when the student is discouraged.
- **Iteration:** the first attempt (v2.1a, "a touch of wit, in the opening line or closing question") gave sass in 2/3 test replies (none on the L1-vs-L2 concept question) and a contradictory scold ("I can't just dump code on you" followed by the code). Making the opener mandatory and banning "can't" scolding fixed it: 3/3 replies had a short opener.
- **Known weakness:** openers are formulaic ("Another day, another ..." appeared in 2 of 3 replies). Small-model style; accept for now, revisit with a larger model or sampling changes.
- Persona tuning is deliberately stopped here; a real pedagogy/tone evaluation belongs to the evaluation phase.

---

## 2026-11-18 — Phase 1 fixes (redo of the weak parts)

Issues from the Phase 1 smoke tests and a code review, and what changed:

| Issue | Fix |
|---|---|
| Arithmetic slips (headline 1.96% vs derived 1.94%) | Persona v3: work the calculation first, bold answer at the END; optional thinking mode; Phase 2 calculator tool |
| Formulaic sass openers ("Another day, another ...") | Qwen-recommended sampling (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5) + persona v3.1: one random concrete opener style per reply (listing banned phrases primed them, see below) |
| Vague follow-up questions | Persona: exactly one concrete question answerable in a sentence or small code change, with examples |
| Token budget was a chars/3 guess | Exact counts via llama-server `/tokenize`, cached per message, char estimate as fallback |
| 8k context, 4 server slots | 16k context, 1 slot (~5.6 GB VRAM; 32k would need ~5.8 GB) |
| Replies silently cut at 1000 tokens | 2048-token answers (4096 with thinking), and the CLI says when a reply hit the limit |
| No transcripts | JSONL per session in `logs/transcripts/` (gitignored), incl. persona version, opener style, timings, tool calls |
| Could silently reuse a server running another model | Checks `/props` model path; refuses on mismatch |
| Ctrl+C at the prompt crashed; HTTP errors were opaque; log handle leaked | Clean exit; server error messages shown; handle closed |
| No reasoning control | `/think on|off`, `--think`, `--show-thinking`; reasoning budget 2048 tokens |

### A/B check (`scripts/phase1_check.py`, 5 samples x 3 numeric questions + 8 opener questions)
Numeric questions with known answers: Bayes test (1.94%), parameter count of a 2-layer MLP (101,770), training batches for 1,250 samples / 80% split / batch 32 (32). Scored by the final bold answer in the solution part (excluding walkthrough and follow-up question); failures read by hand.

| Setup | Correct (of 15) | Mean time | Openers starting "Another ..." |
|---|---|---|---|
| persona v2.1, temp 0.3 (as committed) | 9 | 10.9 s | 3/8 |
| persona v3, Qwen sampling | 13 | 15.5 s | 5/8 |
| persona v3, thinking on | 15 | 77.4 s | 0/8 |

- v2.1 failures were real: wrong headline numbers stated before the calculation (4.7%, 1.96%), forgetting the 80% split (40 batches), an addition error (101,706). Two replies even contained "Wait, I miscalculated".
- v3 fixed most of it; remaining misses: one rounded to "2%", one stray wrong headline (9,466) before a correct derivation.
- Thinking mode was perfect but 5x slower -> **thinking stays opt-in** (`/think`), not default.
- Banning "Another day, another ..." by name made it *more* frequent (5/8): naming a phrase primes small models. Replaced by random per-reply opener styles (v3.1); verified in the Phase 2 run below.
- The first auto-scorer was wrong in both directions (it read the number in the follow-up question, and missed LaTeX `\mathbf{}`); fixed and re-scored. Lesson for the thesis: automated answer extraction needs validation against manual reading.

---

## 2026-11-18 — Phase 2: tools

Implemented with OpenAI-style function calling through llama-server (`--jinja`), streamed; tool-call fragments are reassembled client-side. Agent loop in `agent.py`: stream -> run requested tools -> feed results back -> repeat (max 6 rounds; the last round offers no tools, forcing an answer).

| Tool | What | Safety |
|---|---|---|
| `calculate` | exact arithmetic via an AST whitelist (no eval) | limits on exponent/factorial/result size |
| `run_python` | runs a script in a subprocess, temp dir, 20 s timeout, `-I` | **asks the user every time**; NOT a security sandbox (runs as the user, has network/files) |
| `list_files`, `read_file`, `search_text` | read-only access to one workspace dir (default: cwd) | path-escape blocked; skips .git/.venv etc.; size limits |
| `arxiv_search`, `arxiv_paper` | arXiv export API | only tools that leave the machine: query text goes to arxiv.org; generic User-Agent; 3 s rate limit; shown in the CLI; `--offline` disables |

- Tool guidance in the system prompt is generated only for the tools that are actually available.
- Memory now trims whole turns (user message + tool calls + tool results + answer) so a tool call is never separated from its result.
- Live probe: Qwen3.5-9B called `calculate` correctly and answered 101,770. **Issue:** it first drafted an answer, then called the tool and wrote "My bad ... I hallucinated a different number in the draft". Persona now says: call the tool first, answer once, never mention drafts.

---

## 2026-11-18 — Phase 3: paper library (RAG)

- `alice ingest PATH...` (PDF, .md, .txt), `alice library [--remove ID] [--retitle ID TITLE]`. Library = one SQLite file in `data/library/` (gitignored: papers are personal/copyrighted). `ALICE_LIBRARY` env var selects another library (eval uses `data/eval_library/`).
- Extraction with pypdf (BSD; PyMuPDF avoided because of AGPL). Cleaning: de-hyphenation, whitespace. Chunks: ~900 chars, sentence-aligned, one sentence overlap, per page (so citations have page numbers).
- Embeddings: **Qwen3-Embedding-0.6B Q8_0** (639 MB) in a second llama-server (`--embedding --pooling last`), query instruction prefix as recommended for Qwen3-Embedding. Chunks are embedded with their title prepended.
- Search: hybrid = dense cosine (numpy, brute force) + SQLite FTS5 BM25 (porter stemming), fused with reciprocal-rank fusion (k=60).
- Citations: every passage shown to the model gets a session label `[L1]`, `[L2]`, ...; after each reply the CLI lists the cited sources (title, page) and **warns about any label that was never retrieved** (fabricated citation).
- Ingestion speed (5 arXiv papers, 15+12+15+16+26 pages, 415 chunks): **CPU 752 s** (~1.8 s/chunk, while the chat model ran on the GPU) vs **GPU 84 s**. `alice ingest` uses the GPU automatically when the chat model is not running. At chat time the embedder runs on the CPU (`CUDA_VISIBLE_DEVICES=-1`) so it never competes for VRAM; one query embeds in well under a second.
- Title detection: PDF metadata, else the largest font in the top part of page 1, skipping rotated text. The first version took arXiv's rotated margin stamp ("arXiv:1706.03762v7 [cs.CL] ...") as the title for 4/5 papers, and a figure label for LoRA; small-caps titles come out as letter soup, so broken-looking titles fall back to the first plausible text line. All 5 eval titles now correct (LoRA via the fallback).

---

## 2026-11-18 — Phase 2 + 3 evaluation results and the fixes they triggered

### Calculator tool vs thinking (same 15 numeric answers as the Phase 1 A/B; failures read by hand)
| Setup | Correct | Used the tool | Mean time |
|---|---|---|---|
| persona v3, no tools | 13/15 | - | 15.5 s |
| persona v3, thinking | 15/15 | - | 77.4 s |
| v3.1 + calculator (run 1) | 13/15 | 12/15 | 20.3 s |
| v3.1 + calculator, copy-friendly output (run 2) | **15/15** | 13/15 | **19.4 s** |

- Run-1 failures: one reply skipped the tool and said "~1%"; one called it, got `101770`, and wrote **"10,177"** plus an invented supporting calculation. Fixes: the calculator now also prints `101770 (101,770)` and fractions as percentages, and the CLI runs a **grounding check**: if the last calculator result does not appear in the answer (allowing rounding / % form), it warns the user. The real "10,177" case is a regression test.
- Conclusion: tool use gets thinking-level accuracy at a quarter of the latency, so thinking stays opt-in.

### Opener variety
- Random opener styles alone (v3.1): still 3/8 "Another ...". A hard logit ban on the "Another" token made the model write "AnOther" (it routes around bans), so a soft penalty (-6) is used: 0/16 afterwards.
- It then drifted to "Oh, ..." (5/16), used a pet name once ("Oh, darling"), scolded once, and the "smug fun fact" style produced **invented trivia** ("Luhn patented a hash map in 1953"; it was an internal IBM memo). The fun-fact style was removed (honesty beats variety), "Oh" gets -3 and " darling" -8.
- Final 16 openers: no "Another", no pet names, no trivia; repeated first words only on topic words. Spot check afterwards still found one "Oh, ..." that teased the student's ability. **Known limitation**: tone control on a 9B is statistical, not guaranteed.

### Tool-loop bug (found in the opener run)
- Asked about softmax, the model put Python code (`import math; ...`) into `calculate`, got "invalid syntax", repeated the identical call 5 times, and in the final no-tools round wrote a raw `<tool_call>` block as its answer.
- Fixes: calculate explains it takes one expression and points to run_python; identical failed calls are not re-run (if a whole round is repeats, the next round is the last); the last round gets a not-saved "answer now in plain text" note; leaked tool-call markup is stripped (polite fallback if nothing is left); the CLI printer holds back anything that could be the start of `<tool_call>`. Regression tests replay the exact failure. Live re-run of the prompt: 4/4 clean.

### arXiv search bug (found in the live CLI test)
- "Attention Is All You Need" returned no results: ANDing every word including stopwords matches nothing on arXiv. Now: exact title phrase first (`ti:"..."`), then content keywords, then a 3-keyword fallback; results say which strategy matched ("exact title match" / "keyword match" / "partial keyword match; no exact title match"). For the fake paper this now reads "no exact title match, no keyword match", which is exactly the evidence the model needs.

### Phase 3: retrieval (10 questions with a known source paper, top-5)
| Mode | hit@5 | MRR (first run) | MRR (after ligature fix) |
|---|---|---|---|
| keyword (BM25) | 9/10 | 0.85 | 0.85 |
| dense | 8/10 | 0.67 | 0.70 |
| hybrid (RRF) | 9/10 | 0.78 | 0.82 |

- Bug found: pypdf returns ligatures (`ﬁne-tuning` with U+FB01), so keyword search for "fine-tuning"/"efficient" never matched. Fixed with NFKC normalization (also turns math letters like 𝑊 into W); re-ingested.
- Remaining miss (all modes): "What are BERT's two pre-training tasks?" - the paper says "two unsupervised tasks ... Task #1: Masked LM"; vocabulary mismatch. Next step: a reranker (e.g. Qwen3-Reranker-0.6B).
- Interesting for the thesis: keyword search alone beats dense on this set; the 0.6B embedder blurs exact terms (symbols, numbers, names).

### Phase 3: grounded answers (10 paper questions + 1 negative)
| | model decides when to search | retrieval-first (prefetch) |
|---|---|---|
| library actually used | **0/10** | 10/10 |
| all key facts in answer | 8/10 (from memory) | 10/10 |
| cites >= 1 passage | 0/10 | 10/10 |
| cites the right paper | 0/10 | 10/10 |
| fabricated citation labels | 1 (caught by the checker) | 0 |
| negative question admits "not in library" | yes | yes |

- Left to itself the 9B never searched: it knows these famous papers and answered from memory, once even citing `[L1]` without retrieving anything. Retrieval-first design: before each turn the user's message is searched; if the best passage is similar enough, a synthetic `search_library` call + result is inserted into the turn, so the model only has to cite.
- Threshold calibration (Qwen3-Embedding-0.6B, top-1 cosine): paper questions 0.68-0.83, unrelated CS questions 0.22-0.58 -> 0.65. A paraphrase ("What learning-rate warmup schedule did the *Transformer paper* use?") scored only 0.605 and missed, so messages that explicitly mention papers/notes/library/authors use 0.50. Similarity alone cannot separate "about my papers" from "general question on a topic my papers cover"; intent words are a cheap extra signal.
- **Caveat (important):** these five papers are famous, so the model already knows most answers (8/10 facts without RAG). The eval measures grounding/citation behaviour, not whether RAG adds knowledge. A proper thesis eval needs papers the model cannot know (published after its training, e.g. 2026) and more questions, with several runs.

### Live CLI test (all tools + eval library)
Library question -> searched, cited [L1] (Attention Is All You Need p. 7), correct formula and warmup 4000; Bayes -> calculator, 1.94%, grounding check passed; fake paper -> three arXiv searches, correctly "does not exist"; run_python -> asked for confirmation, ran, correct Fibonacci output. Issues seen: needless arXiv lookups for a paper already in the library, and chatty "let me check..." narration before tool calls.

### Other fixes in this round
- `.gitignore` pattern `tools/` also hid `src/alice/tools/` (the whole tools package would never have been committed). Root-only patterns are now anchored (`/tools/`, `/models/`, `/data/`, `/logs/`).
- Title detection: rotated arXiv stamps and figure labels skipped; small-caps fallback.

### Status
- Phase 0: done. Phase 1: done (chat CLI, persona v3.1, transcripts, thinking toggle). Phase 2: done for the MVP (7 tools + safety guards). Phase 3: working end to end (ingest, hybrid search, retrieval-first answering, citation check).
- Tests: 76 passing (`pytest`), including regression tests for every bug above.

### Next (still within Phases 2-3)
- [ ] Reranker for the vocabulary-mismatch misses
- [ ] Eval set built from papers the model cannot know + more questions, multiple runs (thesis-grade)
- [ ] Skip arXiv lookups when the library already has the paper; trim pre-tool narration
- [ ] Real sandbox for run_python (e.g. a restricted user or container) if it is ever auto-approved

---

## 2026-11-18 — Improvement round 2: retrieval, latency, an honest RAG test

### Retrieval: section-aware chunking (and an ablation that corrected my explanation)
- **Snag:** "What are BERT's two pre-training tasks?" missed in every mode. The evidence chunk ("Task #1: Masked LM ...") ranked 7th-8th. The words "pre-training tasks" only appear in the section heading "3.1 Pre-training BERT", not in the passage.
- **Why not a reranker:** the standard fix, but the 9B leaves ~0.5 GB VRAM, so a reranker would run on the CPU (~150 tok/s measured for the 0.6B embedder) -> ~10-20 s extra per question. Rejected for now.
- **Fix:** detect numbered section headings in PDFs (and `#` headings in Markdown), split pages at headings, carry the current section across page breaks, store it per chunk, index/embed `title | section` with the chunk, and show it in citations ("BERT ..., p. 4 (3.1 Pre-training BERT)"). `alice library --rebuild` re-ingests old libraries; heading detection rejects table rows, equations and sentences (unit-tested).
- **Ablation** (famous-papers set, top-5):

| Pipeline | keyword | dense | hybrid |
|---|---|---|---|
| page chunks (before) | 9/10, MRR 0.85 | 8/10, 0.70 | 9/10, 0.82 |
| section-aligned chunks, no header text | 10/10, 0.88 | 8/10, 0.67 | 10/10, 0.80 |
| section-aligned chunks + header text | 10/10, 0.88 | 8/10, 0.72 | 10/10, 0.80 |

- **Lesson:** the miss was fixed by *splitting at headings* (the "Task #1" passage now starts its own chunk instead of trailing the previous section), not by the header text, which only helps dense search. Without the ablation I would have credited the wrong mechanism.

### Latency
- Measured: CPU embedding server start 4.1 s; one retrieval (embed query + search) 0.2 s. **Fix:** the embedder now warms up in a background thread when the chat starts, so the 4 s overlap with typing (thread-safe start).
- Inserted passages are ~3.3-3.9k characters (~1k tokens); at ~160 tok/s prompt processing that is ~6 s per library-backed turn. Evidence ranks with hybrid search: 1 for 7/10 questions, but 4 and 5 for two of them, so prefetch now inserts 5 passages (+~1.5 s) instead of 4.
- Not done on purpose: keyword search alone ranked every evidence chunk in the top 3 on this set, so weighting BM25 above dense is tempting, but tuning fusion weights on the 10 questions used to evaluate them would be overfitting. Revisit with an independent question set.

### Snag: the eval overwrote its own results
- Running `rag_eval.py --retrieval-only` rewrote `benchmarks/results/rag_eval.json`, which held the committed answer results. Restored from git; output files are now named `rag_<set>_<mode>.json` and never shared between modes.

### A RAG test the model cannot cheat on
- The famous-paper set can't show whether RAG adds knowledge (the model knew 8/10 answers without it). New set: 5 arXiv papers from **2026-11-17** (BQ-LoRA, The Kernel Reality Check, MoE-LoRA multilingual study, Gemma 4 MATLAB QLoRA, Ambient SFT), 13 questions with answers taken from the abstracts (checked against the extracted text) + 1 off-topic negative. `scripts/rag_eval.py --set 2026`.
- New eval mode `--closed-book`: the model with no library at all, to measure what it knows alone.
- Small snag: piping arXiv's Atom XML through Git Bash corrupted the encoding (surrogate characters); parsing a saved file with ALICE's own arXiv parser worked.
- Retrieval on the 2026 set: keyword 12/13 (MRR 0.87), dense 10/13 (0.62), **hybrid 13/13 (0.82)** - the first set where fusion beats both parts. All 5 titles detected automatically.

### The headline result: does the library add knowledge? (2026 papers, single run)
| | closed-book (no library) | retrieval-first (prefetch) |
|---|---|---|
| all key facts correct | **0/13** | **12/13** |
| cites >= 1 passage / right paper | 0/13 | 12/13 |
| fabricated citation labels | 0 | 0 |

- RAG is the only reason ALICE can answer these. Remaining RAG errors: one answer used a passage about a 10-language pilot instead of the abstract's 200 languages; one had correct facts but no citation.

### Honesty when the model does NOT know (closed-book, graded by hand)
- **Snag:** without the library, persona v3.1 confabulated: invented BQ-LoRA "variants" ("Block-Q / Block-KV"), a "50%" storage figure, Ambient-SFT mechanics; it also claimed things don't exist ("there's no such thing as Gemma 4, the latest is Gemma 2") - treating its training cutoff as the present - and belittled the student twice ("another student trying to compare apples and oranges without even realizing it", "let me guess, you didn't read the paper").
- **Fixes (persona v3.2):** today's date in the system prompt + "your training data ends long before today; never claim something doesn't exist because you don't know it"; "say so in your first sentence, don't describe what an unknown paper probably does"; the student is framed as a capable colleague and humour is aimed at the topic (the "teasing remark about the situation" opener style was removed because it kept turning into teasing the person). Plus an **unverified-reference guard**: when a message names specific work (paper/study/authors, or named methods like BQ-LoRA, Gemma 4) and no library passage matched, a note at the end of the system prompt says "if you don't know it, say so; use arxiv_search".
- **Snag in the measurement:** the regex "admits unknown" detector reported 5 -> 9 -> 9 but is too generous (it counts "...by 75%. However, I may not have details" as an admission). Hand grades (C clean admission, D false denial, H hedge-then-guess, F fabrication; `benchmarks/results/manual_grades_2026_closedbook.json`):

| Setup | C | D | H | F |
|---|---|---|---|---|
| v3.1 | 4 | 3 | 0 | 6 |
| v3.2, guard off | 9 | 0 | 0 | 4 |
| v3.2, guard on (no arXiv tool) | 5 | 0 | 3 | 5 |

- The persona rewrite removed the false denials and roughly doubled clean admissions. The guard showed no benefit in this setup (13 answers, temperature 0.7: differences of 1-2 are noise).
- **But that test missed the guard's purpose:** in the real CLI the guard tells the model to *use arxiv_search*, and the closed-book eval had no arXiv tool. Re-run with the arXiv tools available (`rag_eval.py --closed-book --arxiv`):

| | guard off | guard on |
|---|---|---|
| arXiv actually searched | 4/13 | **8/13** |
| all key facts correct | 3/13 | **8/13** |
| condescending replies | 0/14 | 0/14 |

- Decision: **guard stays on.** Its value is triggering lookups, not phrasing.
- Of the 5 guard-on answers that still skipped arXiv, 4 fabricated (e.g. "75%"; "BQ-LoRA stands for Bit-Quality Low-Rank Adaptation") and 1 claimed "I've searched for this ... cannot find it" without calling any tool - a **fabricated tool use**, a new failure type to track.
- Tone across all v3.2 runs: 0 condescending replies by the regex screen (0/14, 0/14, 0/14, 0/14, plus 0/16 openers), all openers aimed at the topic. Regex screening only; occasional off-topic openers remain ("The fashion industry is surprisingly resilient...").

### Tool tidy-ups (same round)
- Library passages now say they are the student's own copies ("no need to look these up elsewhere") and the arXiv guidance says not to search arXiv for papers the library already covers (the live CLI test had redundant arXiv lookups).
- Tool guidance: "do not narrate tool use ('let me check...'); just call the tool."


### Next
- [ ] Opt-in automatic arXiv lookup when the guard fires (like library prefetch). Must be opt-in: it would send the question text to arxiv.org without an explicit request.
- [ ] Detect fabricated tool use ("I searched..." with no tool call) the way citations and calculations are checked.
- [ ] Repeat the honesty/tone evals with several seeds and a larger question set; hand-grade a sample, use regexes only for screening.
- [ ] Hardware note: larger unified-memory and GPU configurations should allow larger default models and reduce small-model failures.

---

## 2026-11-18 — Scalability: ready for a hardware upgrade

Goal: ALICE must keep working, and get better, when the hardware is upgraded.

### What would have broken on an upgrade, and the fixes
| Problem | Fix |
|---|---|
| Model file, context, GPU layers, ports, embedder device hard-coded in `config.py` | `alice.toml` (gitignored; `alice.example.toml` committed). Precedence: env vars > alice.toml > code defaults (= the original dev setup). New profiles via `[models.<name>]` incl. `extra_args` for server flags (e.g. flash attention); typos in keys are rejected with the list of valid keys. `[chat] profile` picks the default. |
| Windows-only server path (`llama-server.exe`) | One config for every OS; `.exe` added on Windows only. |
| Changing the embedding model would silently corrupt search (old and new vectors compared; crash if dimensions differ, plausible garbage if not) | The library records `embed_model` / `embed_dim` in a `meta` table. With a different configured model, search and ingest refuse with "run `alice library --rebuild`", and the chat disables the library with that message. Old libraries are adopted as built with Qwen3-Embedding-0.6B. A missing embedding model file is reported instead of silently skipping the library. |
| Retrieval thresholds tuned to one embedder | Moved to `[embedding]`; `scripts/calibrate_retrieval.py` derives new values from related vs unrelated questions. Re-run for the current model: 0.65 (2026 set) / 0.62 (famous set) vs the configured 0.65, so the hand-calibrated value holds. The explicit-mention threshold it suggests (0.53-0.61) is higher than the 0.50 chosen from a real paraphrase (0.605); kept 0.50. |
| Vectors decoded row by row into one in-memory matrix on every start | `alice/vectors.py`: memory-mapped float32 cache next to the database, built by streaming from SQLite into a preallocated file, searched exactly in 1 GB blocks. SQLite stays the source of truth; the vector store is behind a small interface so an approximate index can replace it later. |

### Snag: Windows and memory-mapped files
- The first cache version overwrote `vectors.npy` on rebuild. A test that kept a second handle open (= a chat session running while `alice ingest` adds papers in another terminal) failed with `OSError: [Errno 22]`: Windows cannot overwrite or delete a file that is memory-mapped.
- Fix: versioned cache files (`vectors-<hash>.npy`), written to a temp name and moved into place atomically, never overwritten; old versions are deleted once nothing maps them. The library keeps a version counter in `meta`, bumped on every add/remove/rebuild, and each search compares it (O(1)), so a running chat picks up papers ingested by another process. Regression test simulates the two processes.

### How far does exact vector search go? (`scripts/scale_bench.py`, 1024-dim, the dev machine's CPU)
| Chunks (~papers) | Cache size | Search median | p95 |
|---|---|---|---|
| 10,000 (~100) | 0.04 GB | 1.7 ms | 1.9 ms |
| 100,000 (~1,000) | 0.41 GB | 18.9 ms | 21.3 ms |
| 300,000 (~3,000) | 1.23 GB | 56.9 ms | 64.6 ms |
| 1,000,000 (~10,000) | 4.10 GB | 184.5 ms | 193.9 ms |

- Cache build from SQLite: 6.6 s per 100k chunks (plus 7.3 s to insert them). Linear scaling throughout.
- Conclusion: exact search is fine far beyond a student's library. **Known limit:** the cache is rebuilt in full after every change (~1 min per ingest at 10,000 papers). Past ~300k chunks, switch to an incremental/approximate index (sqlite-vec keeps the single-file design; FAISS/LanceDB after that) by implementing `search` / `invalidate` / `len`.
- SQLite FTS5 (keyword half) handles millions of rows; an index on `chunks(doc_id)` was added for removals.
- Regression: retrieval results identical before/after on both eval sets.

### Snag found during the live check: citation laundering
- Asked "Which optimizer settings did the Transformer paper use?", ALICE answered partly from memory with invented details ("learning rate 0.0001", "no weight decay", "batch size 64") and cited **real but irrelevant** passages ([L4] = the LoRA introduction). The citation check passed: it only verified that cited labels were retrieved.
- Fix: a **support check**. For every sentence citing [Lk], each number in it (decimals, percentages, integers >= 10) must appear in one of the cited passages; otherwise the CLI warns "these numbers are not in the passages they cite ... may come from the model's memory". Unit test replays the failure.
- Limit seen in two more live runs: invented numbers in *uncited* sentences are not checked, and a misread but present number ("0.1 instead of 0.3") passes. It narrows citation laundering; it doesn't eliminate it.

### Snag: papers referred to by nickname
- Root cause of the above: retrieval. "The Transformer paper" is "Attention Is All You Need" (no "Transformer" in the title) and "the ResNet paper" is "Deep Residual Learning..."; keyword search is pulled to the many LoRA/BERT passages that say "Transformer". New eval set `--set alias` (6 questions, evidence verified in the stored text): hybrid 4/6, MRR 0.58.
- **Tried and reverted: document-level routing** (embed each paper's title + first passages, search within the best-matching papers as a third RRF signal). Result: alias set unchanged (4/6), famous set worse (10/10 -> 9/10, MRR 0.80 -> 0.76), 2026 set MRR 0.82 -> 0.77. Routing found the right paper, but within it the small embedder still didn't rank "5.3 Optimizer" for "optimizer settings ... Transformer paper"; stripping the nickname from the query loses the paper altogether (it then finds the Adam paper). Code removed (no benefit, extra cost), numbers back to baseline.
- Next candidates: paper-scoped search (detect which paper a nickname means, then search only it with the rest of the question) or LLM query rewriting; both better judged after a model upgrade.

### Tests
90 passing (config portability and validation, profile overrides, model-mismatch refusal and rebuild, legacy adoption, vector index vs brute force across blocks, cache reuse/rebuild with a second mapped handle, citation support check).

---

## 2026-11-18 — Chat box (full-screen terminal UI)

New: a full-screen chat box, `alice tui` (same options as `alice`).

### Design
- **Textual 8.2** (pure-Python TUI framework; runs in Windows Terminal, VS Code, macOS/Linux terminals). Added as a dependency.
- Layout: header, scrolling conversation (your messages in one style of bubble, ALICE's replies rendered as **Markdown** - code blocks, bold, lists - streamed and re-rendered at most every 80 ms), tool activity lines (`▸ calculate: 37 * 89` / `◂ ok: ...`), library/sources/warnings lines, a status bar (model, thinking, tools, library size, last-turn speed), an input box, and a key footer.
- run_python permission is a **modal dialog** showing the code with syntax highlighting (y/n/Esc). The agent runs in a worker thread and waits for the dialog through an event.
- Keys: Enter send, Esc stop the reply, Ctrl+T thinking, Ctrl+R new conversation, Ctrl+L clear screen, Ctrl+Q quit; the /commands work too.
- **Refactor:** the chat logic moved from `cli.py` into `session.py` (`ChatSession`: tools, library, automatic retrieval, reference guard, agent loop, citation/support/calculator checks, transcripts) so the line CLI and the chat box share one implementation. The leaked-`<tool_call>` filter became a reusable `StreamFilter`.
- **Stop:** `run_turn` takes a cancel event, checked on every streamed event and between tool rounds; closing the stream closes the HTTP connection so llama-server stops generating. Stopped exchanges are not kept in memory.
- Also fixed while refactoring: the chat embedder was created with `gpu=False`, ignoring the new `[embedding] device` setting.

### Snags found by running it for real (the fake-session tests passed first time)
1. **Handler name collision:** the method `on_ready(started)` is also Textual's handler for its built-in `Ready` event (first paint). Textual called it immediately, enabling the input before the model had loaded; questions then hit an unstarted session ("'NoneType' object has no attribute 'add_user'"). Renamed to `session_ready`; regression test with a slow fake start; `ChatSession.ask()` now fails clearly if called before `start()`.
2. **SQLite and threads:** the library is opened on the UI thread and searched on the worker thread; Python's sqlite3 refuses cross-thread use by default. This SQLite build is "serialized" (`sqlite3.threadsafety == 3`), so the connection is shared with `check_same_thread=False` (only when serialized); chat-time use is read-only; the UI no longer queries the library size on every status refresh. Regression test searches from another thread.
3. **Citation format drift:** the 9B cited "(L1)" instead of "[L1]", so no sources line appeared. The checker now accepts both (only retrieved labels count).

### Verification
- Headless UI tests (Textual's `run_test` pilot, fake session): streaming, tool lines, the dialog (approve and decline), sources, leaked-markup filtering, Ctrl+T/Ctrl+R, Esc stopping a reply, input disabled until the model loads.
- Real end-to-end run driving the chat box headlessly with the 9B and the eval library: paper question -> library passages (incl. 5.3 Optimizer) -> answer with warmup_steps = 4000; "What is 37 * 89?" -> calculator -> 3293; status "last 24.2s · 71 tok · 36 tok/s · tools: calculate".
- Tests: 96 passing.

### Tone decision (2026-11-18)
- Teasing the student is fine. The occasional opener like "...before you try to calculate it in your head and embarrass yourself" is accepted, not a bug. The persona rules stay as they are (they still prevent insults, pet names and scolding), but tone drift toward teasing is no longer a target for further tuning or evaluation.
