#!/usr/bin/env bash
# Sequential, resumable model downloads (parallel downloads from Hugging Face stalled).
# Usage: scripts/download_models.sh   (run from anywhere)
cd "$(dirname "$0")/../models" || exit 1
get() { # repo file
  echo "== $2"
  curl -L -C - --retry 20 --retry-all-errors --retry-delay 5 -sS -o "$2" \
    "https://huggingface.co/$1/resolve/main/$2" && echo "ok $2" || echo "FAILED $2"
}
get unsloth/gemma-4-E4B-it-GGUF            gemma-4-E4B-it-Q4_K_M.gguf
get unsloth/gemma-4-26B-A4B-it-GGUF        gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
get unsloth/Qwen3.6-35B-A3B-GGUF           Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
