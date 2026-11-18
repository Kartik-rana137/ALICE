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
