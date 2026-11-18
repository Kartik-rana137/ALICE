# ALICE Development Log

Running notes for the ALICE project (local, coding/academics/research-focused AI companion; intended as bachelor thesis, ~2030).
Newest entries at the bottom. Record decisions with their *why* — future-me writing the thesis will need them.

---

## 2026-11-18 — Project kickoff & feasibility assessment

### Goal
A locally-run conversational AI (real-time voice, persona, memory, optional avatar) specialised for **coding, academic study and research assistance** instead of entertainment.

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
  - No need for chat-moderation, stream-overlay or audience-interaction features; drops a chunk of the original feature set.
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
