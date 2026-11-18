"""Recalibrate the automatic-retrieval thresholds for the configured embedding model.

The thresholds in alice.toml ([embedding] retrieve_min_sim / retrieve_min_sim_explicit) are specific to
one embedding model. After switching models (and `alice library --rebuild`), run this against an eval
library to get new values:

    set ALICE_LIBRARY=data\\eval_library_2026
    .venv\\Scripts\\python scripts\\calibrate_retrieval.py --set 2026 [--gpu-embed]

"Related" = the eval questions about papers in the library; "unrelated" = everyday CS/ML questions.
Suggested retrieve_min_sim is the midpoint of the gap between them (90th percentile of unrelated,
10th percentile of related); retrieve_min_sim_explicit (used when the message mentions papers/notes)
is set 0.15 below the lowest related score, but never below the median unrelated score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_eval import QUESTION_SETS  # noqa: E402

from alice import config  # noqa: E402
from alice.library import Library, LlamaEmbedder  # noqa: E402

UNRELATED = [
    "What's the difference between a process and a thread?", "How do I reverse a singly linked list in C?",
    "What's the time complexity of quicksort?", "How does a hash table work?",
    "Explain gradient descent in simple terms.", "What is overfitting and how do I detect it?",
    "What does the softmax function do?", "I've been stuck on a segfault in my C program for 3 hours.",
    "How many trainable parameters does nn.Linear(784, 128) have?", "What is batch normalization?",
    "How do I use git rebase?", "Explain the difference between TCP and UDP.",
    "What is a closure in Python?", "How does binary search work?", "What is the CAP theorem?",
    "Why is my PyTorch loss NaN?", "Explain Dijkstra's algorithm.", "What is dropout?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(QUESTION_SETS), default="2026")
    ap.add_argument("--gpu-embed", action="store_true")
    a = ap.parse_args()
    lib = Library()
    if lib.is_empty() or lib.embedding_problem():
        print(lib.embedding_problem() or f"library at {config.LIBRARY_DIR} is empty")
        return 1
    related = [q for _, paper, q, _, _ in QUESTION_SETS[a.set] if paper]
    emb = LlamaEmbedder(gpu=a.gpu_embed)
    try:
        top1 = lambda q: lib.dense_ranking(q, emb, n=1)[0][1]   # noqa: E731
        rel = np.array([top1(q) for q in related])
        unr = np.array([top1(q) for q in UNRELATED])
    finally:
        emb.stop()
    lo_rel, hi_unr = np.percentile(rel, 10), np.percentile(unr, 90)
    suggest = round(float((lo_rel + hi_unr) / 2), 2)
    explicit = round(float(max(rel.min() - 0.15, np.median(unr))), 2)
    print(f"embedding model: {config.EMBED_MODEL}   library: {config.LIBRARY_DIR}")
    print(f"related   (n={len(rel)}): min {rel.min():.3f}  p10 {lo_rel:.3f}  median {np.median(rel):.3f}  max {rel.max():.3f}")
    print(f"unrelated (n={len(unr)}): min {unr.min():.3f}  median {np.median(unr):.3f}  p90 {hi_unr:.3f}  max {unr.max():.3f}")
    if lo_rel <= hi_unr:
        print("WARNING: the distributions overlap; automatic retrieval will either miss or over-fire. "
              "Rely more on the explicit-mention threshold.")
    print(f"\nsuggested alice.toml [embedding]:\n  retrieve_min_sim = {suggest}\n  retrieve_min_sim_explicit = {explicit}")
    print(f"current:  retrieve_min_sim = {config.RETRIEVE_MIN_SIM}, retrieve_min_sim_explicit = {config.RETRIEVE_MIN_SIM_EXPLICIT}")
    out = config.ROOT / "benchmarks" / "results" / f"calibration_{a.set}.json"
    json.dump({"embed_model": config.EMBED_MODEL, "set": a.set, "related": rel.round(4).tolist(),
               "unrelated": unr.round(4).tolist(), "suggested": {"retrieve_min_sim": suggest,
               "retrieve_min_sim_explicit": explicit}}, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
