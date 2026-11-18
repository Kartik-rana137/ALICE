"""How far does exact vector search scale? Synthetic benchmark for alice.vectors.VectorIndex.

Usage: .venv\\Scripts\\python scripts\\scale_bench.py [--sizes 10000 100000 1000000] [--build-rows 100000]

Part A (search): random unit vectors in a memory-mapped float32 file (1024 dims, like
Qwen3-Embedding-0.6B), 30 queries each after one warm-up; reports median/p95 latency and file size.
Part B (cache build): inserts N embedding BLOBs into SQLite and times building the cache from it.
Rule of thumb for the library: one ~10-page paper is ~60-130 chunks, so 100k chunks ~ 1,000 papers.
Writes benchmarks/results/scale_bench.json. Temporary files go to data/scale_bench (deleted after).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import alice.vectors as vectors  # noqa: E402
from alice import config  # noqa: E402

DIM = 1024


def _random_unit(rng, n, dim=DIM):
    m = rng.standard_normal((n, dim), dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def bench_search(work: Path, n: int, rng) -> dict:
    d = work / f"search_{n}"
    d.mkdir(parents=True)
    stamp = f"bench-{n}"
    vec_path, id_path = vectors.VectorIndex(d, None, None, None)._paths(stamp)
    mat = np.lib.format.open_memmap(vec_path, mode="w+", dtype=np.float32, shape=(n, DIM))
    for s in range(0, n, 100_000):
        mat[s:s + 100_000] = _random_unit(rng, min(100_000, n - s))
    mat.flush()
    del mat
    np.save(id_path, np.arange(n, dtype=np.int64))
    idx = vectors.VectorIndex(d, rows=lambda: [], count=lambda: n, stamp=lambda: stamp)
    queries = _random_unit(rng, 31)
    idx.search(queries[0], 30)                             # warm-up: maps the file, pages it in
    times = []
    for q in queries[1:]:
        t = time.perf_counter()
        idx.search(q, 30)
        times.append((time.perf_counter() - t) * 1000)
    size_gb = vec_path.stat().st_size / 1e9
    idx.invalidate()
    times.sort()
    return {"chunks": n, "approx_papers": n // 100, "file_gb": round(size_gb, 2),
            "median_ms": round(statistics.median(times), 1), "p95_ms": round(times[int(0.95 * len(times)) - 1], 1)}


def bench_build(work: Path, n: int, rng) -> dict:
    d = work / f"build_{n}"
    d.mkdir(parents=True)
    db = sqlite3.connect(d / "lib.db")
    db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, embedding BLOB)")
    t = time.perf_counter()
    for s in range(0, n, 50_000):
        block = _random_unit(rng, min(50_000, n - s))
        db.executemany("INSERT INTO chunks(embedding) VALUES (?)", ((v.tobytes(),) for v in block))
    db.commit()
    insert_s = time.perf_counter() - t
    idx = vectors.VectorIndex(d, rows=lambda: db.execute("SELECT id, embedding FROM chunks ORDER BY id"),
                              count=lambda: n, stamp=lambda: "v1")
    t = time.perf_counter()
    assert len(idx) == n
    build_s = time.perf_counter() - t
    idx.invalidate()
    db.close()
    return {"chunks": n, "sqlite_insert_s": round(insert_s, 1), "cache_build_s": round(build_s, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 300_000, 1_000_000])
    ap.add_argument("--build-rows", type=int, default=100_000)
    a = ap.parse_args()
    work = config.ROOT / "data" / "scale_bench"
    shutil.rmtree(work, ignore_errors=True)
    rng = np.random.default_rng(0)
    out = {"dim": DIM, "block_rows": vectors.BLOCK_ROWS, "search": [], "build": None}
    try:
        for n in a.sizes:
            r = bench_search(work, n, rng)
            out["search"].append(r)
            print(f"search {n:>9,} chunks (~{r['approx_papers']:,} papers): {r['file_gb']:5.2f} GB, "
                  f"median {r['median_ms']:7.1f} ms, p95 {r['p95_ms']:7.1f} ms", flush=True)
        out["build"] = bench_build(work, a.build_rows, rng)
        print(f"build  {a.build_rows:>9,} chunks from SQLite: insert {out['build']['sqlite_insert_s']} s, "
              f"cache build {out['build']['cache_build_s']} s", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    json.dump(out, open(config.ROOT / "benchmarks" / "results" / "scale_bench.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
