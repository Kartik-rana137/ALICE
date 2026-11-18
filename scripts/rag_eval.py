"""Phase 3 evaluation: retrieval quality and grounded answering over the eval library.

Setup:   ALICE_LIBRARY=data/eval_library  alice ingest data/eval_papers
Usage:   set ALICE_LIBRARY=data\\eval_library
         .venv\\Scripts\\python scripts\\rag_eval.py [--retrieval-only] [--k 5]

Retrieval: a question counts as hit@k if any of the top-k passages comes from the expected paper and
           matches the evidence regex. Compared for keyword (BM25), dense and hybrid (RRF) search.
Answering: full agent turn with the library + calculator tools; checks the library was searched,
           the answer contains the key facts, cites at least one label, cites no unretrieved labels,
           and that cited passages come from the expected paper.
Writes benchmarks/results/rag_eval.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from alice import config, persona
from alice.agent import run_turn
from alice.library import Library, LibrarySession, LlamaEmbedder
from alice.llm import LlamaServer
from alice.tools import ToolBox, calc

# (id, arXiv id of the expected paper or None, question, evidence regex for retrieval, answer regexes)
QUESTIONS = [
    ("tf-optim", "1706.03762", "Which optimizer and learning-rate warmup did the Transformer paper use?",
     r"warmup_?steps|warmup", [r"4,?000", r"0\.98"]),
    ("tf-heads", "1706.03762", "How many attention heads does the base Transformer model use, and what is d_k?",
     r"h\s*=\s*8|8 parallel attention|heads", [r"\b8\b", r"\b64\b"]),
    ("bert-tasks", "1810.04805", "What are BERT's two pre-training tasks?",
     r"next sentence|masked lm|masked language", [r"mask", r"next sentence"]),
    ("bert-mask", "1810.04805", "What fraction of tokens does BERT mask during pre-training, and how are the "
     "chosen tokens replaced?", r"15%|80%", [r"15\s*%", r"80\s*%", r"10\s*%"]),
    ("resnet-depth", "1512.03385", "How deep was the deepest ResNet on ImageNet, and what top-5 error did the "
     "ensemble achieve?", r"152|3\.57", [r"152", r"3\.57"]),
    ("resnet-why", "1512.03385", "What problem with deeper plain networks motivated residual learning?",
     r"degradation", [r"degradation"]),
    ("adam-defaults", "1412.6980", "What default hyperparameter values does the Adam paper recommend?",
     r"0\.999|10\s*[−-]\s*8|default settings", [r"0\.001", r"0\.9\b", r"0\.999"]),
    ("adam-bias", "1412.6980", "Why does Adam apply bias correction to its moment estimates?",
     r"bias[- ]correct|initiali[sz]ed .*zero|biased towards zero", [r"zero", r"initiali[sz]"]),
    ("lora-savings", "2106.09685", "By how much does LoRA reduce the number of trainable parameters and the GPU "
     "memory requirement compared with fine-tuning GPT-3 175B?", r"10,?000|3 times|three times",
     [r"10,?000", r"\b3\b|three"]),
    ("lora-matrices", "2106.09685", "Which attention weight matrices does LoRA adapt in its main experiments?",
     r"W_?q|Wq|query|W_?v", [r"W_?\{?q|query", r"W_?\{?v|value"]),
    ("negative", None, "What do the papers in my library say about diffusion models for image generation?",
     None, []),
]


def doc_arxiv_ids(library: Library) -> dict[int, str]:
    return {doc_id: (re.search(r"(\d{4}\.\d{4,5})", path) or [None, "?"])[1]
            for doc_id, _, _, _, _, path in library.documents()}


def retrieval_eval(library: Library, emb, k: int) -> dict:
    ids = doc_arxiv_ids(library)
    out = {}
    for mode in ("keyword", "dense", "hybrid"):
        rows = []
        for qid, paper, q, evidence, _ in QUESTIONS:
            if paper is None:
                continue
            hits = library.search(q, emb, k=k, mode=mode)
            ranks = [i for i, h in enumerate(hits, 1)
                     if ids.get(h.doc_id) == paper and re.search(evidence, h.text, re.I)]
            rows.append({"id": qid, "hit": bool(ranks), "rank": ranks[0] if ranks else None,
                         "right_paper_in_topk": any(ids.get(h.doc_id) == paper for h in hits)})
        hit = sum(r["hit"] for r in rows)
        mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
        out[mode] = {"hit_at_k": f"{hit}/{len(rows)}", "mrr": round(mrr, 3), "rows": rows}
        print(f"{mode:8} hit@{k} {hit}/{len(rows)}  MRR {mrr:.2f}  misses: "
              + (", ".join(r["id"] for r in rows if not r["hit"]) or "none"), flush=True)
    return out


def answer_eval(library: Library, emb, prefetch: bool = True) -> list[dict]:
    ids = doc_arxiv_ids(library)
    profile = config.PROFILES["default"]
    server = LlamaServer(profile)
    server.start()
    rows = []
    try:
        for qid, paper, q, _, answer_rx in QUESTIONS:
            session = LibrarySession(library, emb)
            box = ToolBox().add(calc.TOOL, *session.tools())
            system = persona.system_prompt(persona.pick_opener_style(), tools=list(box.tools))
            t0 = time.time()
            pre, _ = session.prefetch(q) if prefetch else ([], [])
            res = run_turn([{"role": "system", "content": system}, {"role": "user", "content": q}, *pre],
                           profile, box)
            rep = session.check(res.reply)
            searched = bool(pre) or any(t.name == "search_library" for t in res.tool_results)
            facts = [bool(re.search(rx, res.reply, re.I)) for rx in answer_rx]
            cited_papers = {ids.get(h.doc_id) for _, h in rep.cited}
            row = {"id": qid, "prefetched": bool(pre), "searched": searched, "facts": f"{sum(facts)}/{len(facts)}",
                   "all_facts": all(facts), "n_cited": len(rep.cited), "unknown_labels": rep.unknown,
                   "cited_right_paper": (paper in cited_papers) if paper else None,
                   "cited_papers": sorted(p for p in cited_papers if p), "seconds": round(time.time() - t0, 1),
                   "rounds": res.rounds, "tools": [t.name for t in res.tool_results], "reply": res.reply}
            if paper is None:                         # negative: should admit the library doesn't cover it
                row["admits_not_covered"] = bool(re.search(
                    r"(not|n't|no)\b.{0,60}\b(library|passages|papers|cover|mention|discuss|contain|find|found)",
                    res.reply, re.I))
            rows.append(row)
            print(f"{qid:14} searched={searched} facts={row['facts']} cited={len(rep.cited)} "
                  f"unknown={rep.unknown} right_paper={row['cited_right_paper']} "
                  f"{row.get('admits_not_covered', '')} {row['seconds']}s", flush=True)
    finally:
        server.stop()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--gpu-embed", action="store_true", help="embed queries on the GPU (no chat model running)")
    ap.add_argument("--no-prefetch", action="store_true", help="leave searching entirely to the model")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    library = Library()
    if library.is_empty():
        print(f"library at {config.LIBRARY_DIR} is empty; ingest data/eval_papers first")
        return 1
    emb = LlamaEmbedder(gpu=a.gpu_embed)
    try:
        results = {"library": str(config.LIBRARY_DIR), "k": a.k,
                   "documents": [d[1] for d in library.documents()],
                   "retrieval": retrieval_eval(library, emb, a.k)}
        if not a.retrieval_only:
            results["prefetch"] = not a.no_prefetch
            results["answers"] = answer_eval(library, emb, prefetch=not a.no_prefetch)
            ans = [r for r in results["answers"] if r["id"] != "negative"]
            neg = [r for r in results["answers"] if r["id"] == "negative"]
            print(f"\nanswers (prefetch={'on' if not a.no_prefetch else 'off'}): "
                  f"prefetched {sum(r['prefetched'] for r in ans)}/{len(ans)} (negative: {[r['prefetched'] for r in neg]}), "
                  f"searched {sum(r['searched'] for r in ans)}/{len(ans)}, "
                  f"all facts {sum(r['all_facts'] for r in ans)}/{len(ans)}, "
                  f"cited >=1 label {sum(r['n_cited'] > 0 for r in ans)}/{len(ans)}, "
                  f"cited right paper {sum(bool(r['cited_right_paper']) for r in ans)}/{len(ans)}, "
                  f"fabricated labels {sum(len(r['unknown_labels']) for r in results['answers'])}, "
                  f"negative admits not covered: {[r.get('admits_not_covered') for r in neg]}")
    finally:
        emb.stop()
    out = config.ROOT / "benchmarks" / "results" / ("rag_eval_noprefetch.json" if a.no_prefetch else "rag_eval.json")
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
