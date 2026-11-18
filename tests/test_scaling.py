"""Config portability, embedding-model safety, and the vector index."""
import sqlite3

import numpy as np
import pytest

from alice import config
from alice.library import Library, LibraryError
from alice.llm import LlamaServer
from alice.vectors import VectorIndex
from test_library import FakeEmbedder


# ---------- config ----------

def test_server_binary_is_portable():
    assert config.server_binary("tools/llama.cpp/llama-server", "win32").name == "llama-server.exe"
    assert config.server_binary("tools/llama.cpp/llama-server.exe", "win32").name == "llama-server.exe"
    assert config.server_binary("tools/llama.cpp/llama-server", "darwin").name == "llama-server"
    assert config.server_binary("/opt/llama/llama-server", "linux").is_absolute()


def test_profiles_from_toml_override_and_add(tmp_path):
    f = tmp_path / "alice.toml"
    f.write_text("""
[models.default]
gguf = "Qwen3.8-27B-Q4_K_M.gguf"
ctx = 32768
extra_args = ["-fa", "on"]
sampling = {temperature = 0.6, top_p = 0.9, top_k = 20}

[models.moe]
gguf = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
n_cpu_moe = 36
logit_bias = [["Another", -4]]
""", encoding="utf-8")
    profiles = config.build_profiles(config.load_toml(f))
    big = profiles["default"]
    assert big.gguf.startswith("Qwen3.8") and big.ctx == 32768 and big.sampling.temperature == 0.6
    cmd = LlamaServer(big).command()
    assert cmd[-2:] == ["-fa", "on"] and cmd[cmd.index("-c") + 1] == "32768"
    assert profiles["moe"].n_cpu_moe == 36 and profiles["moe"].logit_bias == (("Another", -4.0),)


def test_profile_table_rejects_typos():
    with pytest.raises(ValueError, match="unknown keys"):
        config.profile_from_table("x", {"gguf": "a.gguf", "context": 8192})
    with pytest.raises(ValueError, match="needs a 'gguf'"):
        config.profile_from_table("x", {"ctx": 8192})


def test_missing_config_file_means_defaults(tmp_path):
    assert config.load_toml(tmp_path / "nope.toml") == {}
    assert "default" in config.build_profiles({})


# ---------- vector index ----------

def _index_over(tmp_path, vecs):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE c(id INTEGER PRIMARY KEY, e BLOB)")
    for i, v in enumerate(vecs, 1):
        db.execute("INSERT INTO c VALUES (?, ?)", (i * 10, v.astype(np.float32).tobytes()))
    calls = {"rows": 0}

    def rows():
        calls["rows"] += 1
        return db.execute("SELECT id, e FROM c ORDER BY id")

    def stamp():
        return str(db.execute("SELECT COUNT(*), MAX(id) FROM c").fetchone())

    idx = VectorIndex(tmp_path, rows=rows, count=lambda: db.execute("SELECT COUNT(*) FROM c").fetchone()[0],
                      stamp=stamp)
    return idx, db, calls, stamp


def _unit(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_vector_index_matches_brute_force_and_blocks(tmp_path, monkeypatch):
    import alice.vectors as vectors
    monkeypatch.setattr(vectors, "BLOCK_ROWS", 7)                      # force several blocks
    rng = np.random.default_rng(0)
    vecs = _unit(rng.normal(size=(50, 16)))
    idx, _, _, _ = _index_over(tmp_path, vecs)
    q = _unit(rng.normal(size=(1, 16)))[0]
    got = idx.search(q, 5)
    want = np.argsort(-(vecs @ q))[:5]
    assert [cid for cid, _ in got] == [(i + 1) * 10 for i in want]
    assert got[0][1] == pytest.approx(float(vecs[want[0]] @ q), rel=1e-5)


def test_vector_cache_reused_and_rebuilt_while_another_handle_has_it_mapped(tmp_path):
    """Windows can't overwrite or delete a memory-mapped file: a chat session holding the cache must not
    break `alice ingest` in another process, and must notice the change before its next search."""
    rng = np.random.default_rng(1)
    idx, db, calls, stamp = _index_over(tmp_path, _unit(rng.normal(size=(5, 8))))
    assert len(idx) == 5 and calls["rows"] == 1
    other = VectorIndex(tmp_path, rows=lambda: 1 / 0, count=lambda: 5, stamp=stamp)   # "chat process"
    assert len(other) == 5                                             # served from the cache file
    db.execute("INSERT INTO c VALUES (999, ?)", (_unit(rng.normal(size=(1, 8)))[0].astype(np.float32).tobytes(),))
    assert len(idx) == 6 and calls["rows"] == 2                        # rebuilt despite `other`'s mapping
    other._rows = idx._rows
    other._count = idx._count
    assert len(other) == 6                                             # the other handle notices and reloads
    assert len(list(tmp_path.glob("vectors-*.npy"))) == 1              # old version cleaned up once unmapped


def test_vector_index_rejects_wrong_dimension(tmp_path):
    idx, _, _, _ = _index_over(tmp_path, _unit(np.ones((3, 8))))
    with pytest.raises(ValueError, match="embedding model changed"):
        idx.search(np.ones(16, dtype=np.float32) / 4, 2)


# ---------- library remembers its embedding model ----------

def test_library_refuses_to_mix_embedding_models(tmp_path):
    note = tmp_path / "n.md"
    note.write_text("# Notes\n\nAttention heads compute weighted sums of values. " * 4, encoding="utf-8")
    emb = FakeEmbedder()
    lib = Library(tmp_path / "lib", embed_model="small-embedder.gguf")
    lib.add(note, emb)
    assert lib.meta("embed_model") == "small-embedder.gguf" and lib.meta("embed_dim") == "256"
    assert lib.embedding_problem() is None

    upgraded = Library(tmp_path / "lib", embed_model="big-embedder.gguf")
    assert "rebuild" in upgraded.embedding_problem()
    with pytest.raises(LibraryError):
        upgraded.search("attention heads", emb, k=1)
    with pytest.raises(LibraryError):
        upgraded.add(tmp_path / "n.md", emb)
    upgraded.rebuild(emb)
    assert upgraded.embedding_problem() is None and upgraded.meta("embed_model") == "big-embedder.gguf"
    assert upgraded.search("attention heads", emb, k=1)


def test_legacy_library_without_metadata_is_adopted(tmp_path):
    note = tmp_path / "n.md"
    note.write_text("# Notes\n\nResidual connections help very deep networks train. " * 4, encoding="utf-8")
    lib = Library(tmp_path / "lib")
    lib.add(note, FakeEmbedder())
    lib.db.execute("DELETE FROM meta")
    lib.db.commit()
    again = Library(tmp_path / "lib")
    assert again.meta("embed_model") == "Qwen3-Embedding-0.6B-Q8_0.gguf"
