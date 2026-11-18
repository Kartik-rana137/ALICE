"""Vector search for the library.

SQLite stays the source of truth (one float32 embedding BLOB per chunk). For search, the vectors are
cached next to the database as a memory-mapped float32 matrix, rebuilt automatically when the library
changes. Startup is then a file map instead of decoding every row, and the OS page cache keeps the
matrix in RAM between queries.

Cache files are versioned (`vectors-<hash>.npy`): a new version is written to a temp file and moved
into place, never overwritten. On Windows a memory-mapped file cannot be overwritten or deleted, so a
chat session that has the cache mapped must not block `alice ingest` running in another terminal.
Each search compares the library's version stamp (O(1)) and reloads if another process changed it.

Search is exact (brute-force cosine on unit vectors), done in blocks so memory stays bounded.
`scripts/scale_bench.py` measures how far this goes; beyond that, implement the same interface
(`search`, `invalidate`, `len`) with an approximate index (sqlite-vec, FAISS, LanceDB) - nothing else
in ALICE depends on how vectors are searched.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

BLOCK_ROWS = 262_144          # 1 GB of float32 at 1024 dims per block


class VectorIndex:
    def __init__(self, directory: Path, rows: Callable[[], Iterable[tuple[int, bytes]]],
                 count: Callable[[], int], stamp: Callable[[], str]):
        """rows(): (chunk_id, float32 bytes) in id order; count(): number of rows; stamp(): a cheap string
        that changes whenever the stored vectors change (a version counter plus the embedding model)."""
        self.dir = directory
        self._rows, self._count, self._stamp = rows, count, stamp
        self._ids: np.ndarray | None = None
        self._vecs: np.ndarray | None = None
        self._loaded_stamp: str | None = None

    def _paths(self, stamp: str) -> tuple[Path, Path]:
        h = hashlib.sha1(stamp.encode()).hexdigest()[:12]
        return self.dir / f"vectors-{h}.npy", self.dir / f"vector-ids-{h}.npy"

    def invalidate(self) -> None:
        """Drop this process's view; the next search reloads (the stamp has changed)."""
        self._ids = self._vecs = None
        self._loaded_stamp = None

    def _ensure(self) -> None:
        stamp = self._stamp()
        if self._vecs is not None and stamp == self._loaded_stamp:
            return
        self._ids = self._vecs = None                     # release any old mapping first
        vec_path, id_path = self._paths(stamp)
        if not (vec_path.exists() and id_path.exists()):
            self._build(vec_path, id_path)
        self._ids = np.load(id_path)
        self._vecs = np.load(vec_path, mmap_mode="r")
        self._loaded_stamp = stamp
        self._cleanup(keep={vec_path.name, id_path.name})

    def _build(self, vec_path: Path, id_path: Path) -> None:
        """Stream rows from the database into a preallocated on-disk matrix (no list of arrays)."""
        n = self._count()
        tmp_vec = vec_path.with_name(vec_path.stem + f".tmp{os.getpid()}.npy")
        tmp_ids = id_path.with_name(id_path.stem + f".tmp{os.getpid()}.npy")
        rows = iter(self._rows())
        first = next(rows, None)
        if first is None:
            np.save(tmp_ids, np.zeros(0, dtype=np.int64))
            np.save(tmp_vec, np.zeros((0, 1), dtype=np.float32))
        else:
            dim = len(first[1]) // 4
            ids = np.empty(n, dtype=np.int64)
            mat = np.lib.format.open_memmap(tmp_vec, mode="w+", dtype=np.float32, shape=(n, dim))
            i = 0
            for cid, blob in _chain(first, rows):
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape[0] != dim:
                    del mat
                    tmp_vec.unlink(missing_ok=True)
                    raise ValueError(f"chunk {cid} has a {v.shape[0]}-dim embedding, expected {dim}: the "
                                     "library mixes embedding models; run `alice library --rebuild`")
                ids[i] = cid
                mat[i] = v
                i += 1
            mat.flush()
            del mat
            np.save(tmp_ids, ids[:i])
        os.replace(tmp_ids, id_path)                      # ids first: vectors present => complete
        os.replace(tmp_vec, vec_path)

    def _cleanup(self, keep: set[str]) -> None:
        for p in list(self.dir.glob("vectors-*.npy")) + list(self.dir.glob("vector-ids-*.npy")):
            if p.name not in keep:
                try:
                    p.unlink()
                except OSError:
                    pass                                  # still mapped by another process: next time

    @property
    def dim(self) -> int | None:
        self._ensure()
        return self._vecs.shape[1] if len(self._ids) else None

    def __len__(self) -> int:
        self._ensure()
        return len(self._ids)

    def search(self, query: np.ndarray, n: int) -> list[tuple[int, float]]:
        """[(chunk_id, cosine similarity)] best first. `query` must be unit length."""
        self._ensure()
        if len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != self._vecs.shape[1]:
            raise ValueError(f"query has {q.shape[0]} dims but the library has {self._vecs.shape[1]}: the "
                             "embedding model changed; run `alice library --rebuild`")
        n = min(n, len(self._ids))
        best_idx = np.empty(0, dtype=np.int64)
        best_sim = np.empty(0, dtype=np.float32)
        for start in range(0, len(self._ids), BLOCK_ROWS):
            sims = self._vecs[start:start + BLOCK_ROWS] @ q
            k = min(n, len(sims))
            top = np.argpartition(-sims, k - 1)[:k]
            best_idx = np.concatenate([best_idx, top + start])
            best_sim = np.concatenate([best_sim, sims[top]])
        order = np.argsort(-best_sim)[:n]
        return [(int(self._ids[best_idx[i]]), float(best_sim[i])) for i in order]


def _chain(first, rest):
    yield first
    yield from rest
