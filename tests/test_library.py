import hashlib

import numpy as np
import pytest

from alice import persona
from alice.library import (Library, LibrarySession, chunk_text, clean_text, fts_query, rrf,
                           split_sentences)


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic and offline, good enough to rank by overlap."""

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.strip(".,").encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_clean_text_dehyphenates_and_joins_lines():
    assert clean_text("trans-\nformer models\nare  great") == "transformer models are great"


def test_clean_text_normalizes_pdf_ligatures_and_math_letters():
    assert clean_text("eﬃcient ﬁne-tuning of 𝑊") == "efficient fine-tuning of W"


def test_split_sentences_and_chunk_overlap():
    text = " ".join(f"Sentence number {i} talks about attention heads and layers." for i in range(60))
    assert len(split_sentences(text)) == 60
    chunks = chunk_text(text, target=300)
    assert len(chunks) > 5 and all(len(c) <= 450 for c in chunks)
    first_last = split_sentences(chunks[0])[-1]
    assert chunks[1].startswith(first_last)                      # one sentence of overlap


def test_chunk_text_splits_giant_sentence():
    assert len(chunk_text("x" * 5000, target=900)) >= 5


def test_fts_query_sanitizes():
    assert fts_query('What is the "dropout" rate (p)?') == '"dropout" OR "rate"'
    assert fts_query("the of and") is None


def test_rrf_prefers_items_ranked_high_in_both():
    s = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(s, key=s.get) == 1 and s[3] > s[2]


@pytest.fixture
def lib(tmp_path):
    (tmp_path / "attn.md").write_text(
        "# Attention notes\n\nThe Transformer uses eight attention heads. Each head has dimension 64. "
        "Training used the Adam optimizer with warmup of 4000 steps. " * 3, encoding="utf-8")
    (tmp_path / "resnet.md").write_text(
        "# ResNet notes\n\nResidual connections add the input to the output of a block. "
        "The deepest network had 152 layers and won ImageNet 2015. " * 3, encoding="utf-8")
    library = Library(tmp_path / "db")
    emb = FakeEmbedder()
    assert library.add(tmp_path / "attn.md", emb)[0] == "added"
    assert library.add(tmp_path / "resnet.md", emb)[0] == "added"
    assert library.add(tmp_path / "attn.md", emb)[0] == "skipped"          # same content hash
    return library, emb


def test_library_search_hybrid_finds_right_doc(lib):
    library, emb = lib
    assert [d[1] for d in library.documents()] == ["Attention notes", "ResNet notes"]
    top = library.search("how many attention heads", emb, k=2)[0]
    assert top.title == "Attention notes" and top.page is None
    assert library.search("152 layers", emb, k=1, mode="keyword")[0].title == "ResNet notes"


def test_library_remove(lib):
    library, emb = lib
    doc_id = library.documents()[0][0]
    assert library.remove(doc_id)
    assert all(h.title != "Attention notes" for h in library.search("attention heads", emb, k=5))


def test_session_labels_and_citation_check(lib):
    library, emb = lib
    s = LibrarySession(library, emb)
    out = s.search_library("attention heads", k=2)
    assert "[L1]" in out and "[L2]" in out
    again = s.search_library("attention heads", k=2)
    assert "[L1]" in again and "[L3]" not in again                       # same chunks keep their labels
    rep = s.check("Eight heads [L1], and also [L1] and [L7].")
    assert [l for l, _ in rep.cited] == ["L1"] and rep.unknown == ["L7"]
    assert "Attention notes" in s.list_library()


def test_persona_tool_guidance_only_for_available_tools():
    p = persona.system_prompt(tools=["calculate"])
    assert "calculate:" in p and "search_library" not in p and "arxiv_search" not in p
    assert persona.system_prompt(tools=[]) == persona.SYSTEM_PROMPT


def test_prefetch_inserts_tool_exchange_only_when_relevant(lib):
    library, emb = lib
    s = LibrarySession(library, emb)
    msgs, hits = s.prefetch("how many attention heads and dimension 64", min_similarity=0.3)
    assert [m["role"] for m in msgs] == ["assistant", "tool"]
    assert msgs[0]["tool_calls"][0]["id"] == msgs[1]["tool_call_id"]
    assert "[L1]" in msgs[1]["content"] and hits
    assert s.check("Eight heads [L1].").unknown == []                  # prefetched labels count as retrieved
    assert s.prefetch("completely unrelated zebra question", min_similarity=0.99) == ([], [])


def test_explicit_paper_references_lower_the_retrieval_bar():
    from alice.library import REFERS_TO_LIBRARY
    assert REFERS_TO_LIBRARY.search("What warmup did the Transformer paper use?")
    assert REFERS_TO_LIBRARY.search("according to my notes, what is LoRA?")
    assert not REFERS_TO_LIBRARY.search("What does the softmax function do?")
