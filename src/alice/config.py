"""Model profiles and paths. The LLM is a swappable backend behind an OpenAI-compatible API."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_EXE = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
LOG_DIR = ROOT / "logs"
HOST = "127.0.0.1"
PORT = int(os.environ.get("ALICE_PORT", "8088"))


@dataclass(frozen=True)
class Profile:
    name: str
    gguf: str                 # file name inside models/
    ctx: int = 8192
    ngl: int = 99
    n_cpu_moe: int = 0        # MoE models: expert layers kept on the CPU
    max_tokens: int = 1000
    temperature: float = 0.3
    note: str = ""

    @property
    def path(self) -> Path:
        return ROOT / "models" / self.gguf


# Chosen in Phase 0 (see docs/DEVLOG.md). Add profiles here when hardware/models change.
# A MoE "heavy" profile (Qwen3.6-35B-A3B, n_cpu_moe=36) was benchmarked and removed; see DEVLOG.
PROFILES: dict[str, Profile] = {
    "default": Profile("default", "Qwen3.5-9B-Q4_K_M.gguf",
                       note="fully on GPU, ~33 tok/s"),
}
