#!/usr/bin/env bash
# Sequential, resumable model downloads (parallel downloads from Hugging Face stalled).
# Usage: scripts/download_models.sh   (run from anywhere)
cd "$(dirname "$0")/../models" || exit 1
get() { # repo file
  echo "== $2"
  curl -L -C - --retry 20 --retry-all-errors --retry-delay 5 -sS -o "$2" \
    "https://huggingface.co/$1/resolve/main/$2" && echo "ok $2" || echo "FAILED $2"
}
get unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf          # default model (5.7 GB)

# Benchmarked in Phase 0 and removed; uncomment to bring one back (see docs/DEVLOG.md):
# get unsloth/Qwen3.6-35B-A3B-GGUF       Qwen3.6-35B-A3B-UD-IQ4_XS.gguf      # 17.7 GB, run with --n-cpu-moe 36
# get unsloth/gemma-4-26B-A4B-it-GGUF    gemma-4-26B-A4B-it-UD-IQ4_XS.gguf   # 13.6 GB, run with --n-cpu-moe 20
# get unsloth/gemma-4-E4B-it-GGUF        gemma-4-E4B-it-Q4_K_M.gguf          # 5.0 GB
# get unsloth/Qwen3.5-4B-GGUF            Qwen3.5-4B-Q4_K_M.gguf              # 2.7 GB
