"""Model profiles and paths. The LLM is a swappable backend behind an OpenAI-compatible API."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_EXE = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
LOG_DIR = ROOT / "logs"
TRANSCRIPT_DIR = LOG_DIR / "transcripts"      # gitignored: personal conversations stay local
HOST = "127.0.0.1"
PORT = int(os.environ.get("ALICE_PORT", "8088"))

# Paper library (Phase 3). The library holds personal/copyrighted PDFs: gitignored.
LIBRARY_DIR = Path(os.environ.get("ALICE_LIBRARY", ROOT / "data" / "library"))
EMBED_GGUF = ROOT / "models" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBED_PORT = int(os.environ.get("ALICE_EMBED_PORT", "8091"))


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    presence_penalty: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# Qwen3.5 model-card recommendations (huggingface.co/Qwen/Qwen3.5-9B, checked 2026-11-18).
QWEN_DIRECT = Sampling(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)      # non-thinking, general
QWEN_THINKING = Sampling(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0)   # thinking, precise/coding


@dataclass(frozen=True)
class Profile:
    name: str
    gguf: str                    # file name inside models/
    ctx: int = 16384             # 16k + 1 slot = ~5.6 GB VRAM for the 9B; 32k needs ~5.8 GB
    ngl: int = 99
    n_cpu_moe: int = 0           # MoE models: expert layers kept on the CPU
    max_tokens: int = 2048       # visible answer budget (non-thinking)
    thinking_max_tokens: int = 4096   # thinking + answer share this budget
    reasoning_budget: int = 2048      # cap on thinking tokens (server flag)
    sampling: Sampling = field(default=QWEN_DIRECT)
    thinking_sampling: Sampling = field(default=QWEN_THINKING)
    # Soft penalties on strings (tokenized by the server). The 9B opens ~40% of sassy replies with
    # "Another ..."; a hard ban made it write "AnOther", so this only discourages it. With that gone it
    # drifted to "Oh, ..." (5/16) and once used a pet name despite the persona rules.
    logit_bias: tuple = (("Another", -6.0), (" Another", -6.0), ("Oh", -3.0), (" darling", -8.0))
    note: str = ""

    @property
    def path(self) -> Path:
        return ROOT / "models" / self.gguf


# Chosen in Phase 0 (see docs/DEVLOG.md). Add profiles here when hardware/models change.
# A MoE "heavy" profile (Qwen3.6-35B-A3B, n_cpu_moe=36) was benchmarked and removed; see DEVLOG.
PROFILES: dict[str, Profile] = {
    "default": Profile("default", "Qwen3.5-9B-Q4_K_M.gguf", note="fully on GPU, ~33 tok/s"),
}
