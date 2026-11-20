"""Configuration.

Hardware-specific settings live in `alice.toml` at the repo root (not committed; copy
`alice.example.toml`). Precedence: environment variables > alice.toml > the defaults below, which
target a constrained consumer GPU setup. Upgrading hardware should only ever mean
editing alice.toml (bigger model, longer context, embedder on the GPU), never editing code.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(os.environ.get("ALICE_CONFIG", ROOT / "alice.toml"))


def load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


_CFG = load_toml(CONFIG_FILE)


def _get(section: str, key: str, default):
    return _CFG.get(section, {}).get(key, default)


def _flag(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def server_binary(path: str | Path, platform: str = sys.platform) -> Path:
    """llama-server path; '.exe' is added on Windows so one config works on every OS."""
    p = resolve(path)
    if platform == "win32" and p.suffix.lower() != ".exe":
        p = p.with_name(p.name + ".exe")
    return p


SERVER_EXE = server_binary(os.environ.get("ALICE_LLAMA_SERVER", _get("server", "binary", "tools/llama.cpp/llama-server")))
HOST = _get("server", "host", "127.0.0.1")
PORT = int(os.environ.get("ALICE_PORT", _get("server", "port", 8088)))
MODELS_DIR = resolve(_get("paths", "models", "models"))
# The processed state of the system prompt's fixed part (persona + tool schemas, ~3k tokens = ~19 s on the dev
# GPU) is saved here and restored on the next start in a fraction of a second (llama-server slot save/restore).
PROMPT_CACHE = _flag(os.environ.get("ALICE_PROMPT_CACHE", _get("server", "prompt_cache", True)))
PROMPT_CACHE_DIR = resolve(_get("server", "prompt_cache_dir", "data/prompt_cache"))
LOG_DIR = resolve(_get("paths", "logs", "logs"))
TRANSCRIPT_DIR = LOG_DIR / "transcripts"      # gitignored: personal conversations stay local

# Paper library (Phase 3). Holds personal/copyrighted PDFs: gitignored.
LIBRARY_DIR = Path(os.environ.get("ALICE_LIBRARY", resolve(_get("library", "dir", "data/library"))))
PREFETCH_K = int(_get("library", "prefetch_k", 5))

# Embedding model. A library remembers which model built it; changing this requires
# `alice library --rebuild`, and the thresholds must be recalibrated (scripts/calibrate_retrieval.py).
EMBED_MODEL = _get("embedding", "gguf", "Qwen3-Embedding-0.6B-Q8_0.gguf")
EMBED_GGUF = MODELS_DIR / EMBED_MODEL
EMBED_PORT = int(os.environ.get("ALICE_EMBED_PORT", _get("embedding", "port", 8091)))
EMBED_DEVICE = _get("embedding", "device", "cpu")     # "cpu" keeps embeddings beside the chat model on constrained setups
RETRIEVE_MIN_SIM = float(os.environ.get("ALICE_RETRIEVE_MIN_SIM", _get("embedding", "retrieve_min_sim", 0.65)))
RETRIEVE_MIN_SIM_EXPLICIT = float(_get("embedding", "retrieve_min_sim_explicit", 0.50))


# Automatic arXiv lookup when a question names work that isn't in the library. Off by default: it sends a
# search phrase (the paper/method name) to arxiv.org without being asked each time.
ARXIV_AUTO = _flag(os.environ.get("ALICE_ARXIV_AUTO", _get("arxiv", "auto_lookup", False)))

# Long-term memory (memory.py): notes about the student kept between sessions, on this machine only.
MEMORY_ENABLED = _flag(os.environ.get("ALICE_MEMORY", _get("memory", "enabled", True)))
MEMORY_DIR = Path(os.environ.get("ALICE_MEMORY_DIR", resolve(_get("memory", "dir", "data/memory"))))
MEMORY_SAVE_ON_EXIT = _flag(_get("memory", "save_on_exit", True))
MEMORY_PROMPT_FACTS = int(_get("memory", "prompt_facts", 25))         # newest facts put in the system prompt
MEMORY_PROMPT_SESSIONS = int(_get("memory", "prompt_sessions", 3))    # newest session summaries
# Compaction: when the conversation fills this share of the context budget, the oldest turns are summarised
# (all but the newest `keep_turns`) instead of silently falling out of the window.
COMPACT_AT = float(_get("memory", "compact_at", 0.75))
COMPACT_KEEP_TURNS = int(_get("memory", "keep_turns", 2))


@dataclass(frozen=True)
class ClaudeSettings:
    """Escalation to Claude (claude.py). Off unless enabled; never sends anything without asking."""
    enabled: bool = False
    model: str = "claude-opus-5-5"
    effort: str = "high"                 # low | medium | high | xhigh | max
    max_tokens: int = 16000
    monthly_budget_usd: float = 5.0
    context_turns: int = 4               # earlier turns sent along with the question
    suggest: bool = True                 # offer /claude when the local answer fails a check


CLAUDE = ClaudeSettings(
    enabled=_flag(os.environ.get("ALICE_CLAUDE", _get("claude", "enabled", False))),
    model=_get("claude", "model", ClaudeSettings.model),
    effort=_get("claude", "effort", ClaudeSettings.effort),
    max_tokens=int(_get("claude", "max_tokens", ClaudeSettings.max_tokens)),
    monthly_budget_usd=float(_get("claude", "monthly_budget_usd", ClaudeSettings.monthly_budget_usd)),
    context_turns=int(_get("claude", "context_turns", ClaudeSettings.context_turns)),
    suggest=_flag(_get("claude", "suggest", True)),
)


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
    gguf: str                    # file name inside the models directory
    ctx: int = 16384             # 16k context target for the constrained default setup
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
    extra_args: tuple = ()       # extra llama-server flags, e.g. ("-fa", "on", "--cache-type-k", "q8_0")
    note: str = ""
    # Chat-template override (path relative to the repo root, or absolute; "" = the model's own). The stock
    # Qwen3.5 template renders earlier turns differently from how the model generated them, so every turn
    # missed llama-server's prompt cache from the previous turn on (DEVLOG 2026-11-19, response time).
    chat_template: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.gguf

    @property
    def template_path(self) -> Path | None:
        if not self.chat_template:
            return None
        p = Path(self.chat_template)
        return p if p.is_absolute() else ROOT / p


def profile_from_table(name: str, table: dict) -> Profile:
    """Build a Profile from an alice.toml [models.<name>] table; unknown keys are an error."""
    known = {f.name for f in fields(Profile)} - {"name"}
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"alice.toml [models.{name}]: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
    if "gguf" not in table:
        raise ValueError(f"alice.toml [models.{name}] needs a 'gguf' file name")
    kw = dict(table)
    for key in ("sampling", "thinking_sampling"):
        if key in kw:
            kw[key] = Sampling(**kw[key])
    if "logit_bias" in kw:
        kw["logit_bias"] = tuple((str(s), float(b)) for s, b in kw["logit_bias"])
    if "extra_args" in kw:
        kw["extra_args"] = tuple(str(a) for a in kw["extra_args"])
    return Profile(name=name, **kw)


def build_profiles(cfg: dict) -> dict[str, Profile]:
    profiles = {
        # Chosen in Phase 0 (see docs/DEVLOG.md). A MoE "heavy" profile (Qwen3.6-35B-A3B,
        # n_cpu_moe=36) was benchmarked and removed; re-add it in alice.toml if wanted.
        "default": Profile("default", "Qwen3.5-9B-Q4_K_M.gguf", note="fully on GPU, ~37 tok/s",
                           chat_template="templates/qwen3.5-stable-history.jinja"),
    }
    for name, table in cfg.get("models", {}).items():
        profiles[name] = profile_from_table(name, table)
    return profiles


PROFILES: dict[str, Profile] = build_profiles(_CFG)
DEFAULT_PROFILE = os.environ.get("ALICE_PROFILE", _get("chat", "profile", "default"))
if DEFAULT_PROFILE not in PROFILES:
    raise ValueError(f"default profile '{DEFAULT_PROFILE}' is not defined (have: {', '.join(PROFILES)})")
