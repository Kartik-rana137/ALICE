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
    assert "Tools (" not in persona.system_prompt(tools=[])


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


def test_heading_detection_pdf_and_markdown():
    from alice.library import heading_of
    assert heading_of("3.1 Pre-training BERT") == "3.1 Pre-training BERT"
    assert heading_of("5 Experiments") == "5 Experiments"
    assert heading_of("A.2 Hyperparameter Details") == "A.2 Hyperparameter Details"
    for not_heading in ["12 34.5 56.1 78.0", "1 We train the model for three epochs and report results.",
                        "2 Dropout rate = 0.1", "A Transformer uses attention", "Task #1: Masked LM"]:
        assert heading_of(not_heading) is None, not_heading
    assert heading_of("## Attention heads", markdown=True) == "Attention heads"
    assert heading_of("#hashtag", markdown=True) is None


def test_split_sections_carries_heading_across_pages():
    from alice.library import split_sections
    p1 = "intro text\n3.1 Pre-training BERT\nWe pre-train BERT using two tasks."
    parts, sec = split_sections(p1, None)
    assert parts == [(None, "intro text"), ("3.1 Pre-training BERT", "We pre-train BERT using two tasks.")]
    parts2, sec2 = split_sections("Task #1: Masked LM is described here.", sec)
    assert parts2 == [("3.1 Pre-training BERT", "Task #1: Masked LM is described here.")]


def test_section_headers_make_heading_words_searchable(tmp_path):
    (tmp_path / "bert.md").write_text("# BERT notes\n\n## Pre-training tasks\n\n"
                                      "Task one is the Masked LM objective. Task two predicts the next sentence.\n\n"
                                      "## Fine-tuning\n\nWe fine-tune on GLUE with a small learning rate.\n",
                                      encoding="utf-8")
    emb = FakeEmbedder()
    with_sec = Library(tmp_path / "a"); with_sec.add(tmp_path / "bert.md", emb)
    without = Library(tmp_path / "b"); without.add(tmp_path / "bert.md", emb, section_headers=False)
    top = with_sec.search("pre-training", emb, k=1, mode="keyword")
    assert top and "Masked LM" in top[0].text and top[0].section == "Pre-training tasks"
    assert "(Pre-training tasks)" in top[0].where
    assert not without.search("pre-training", emb, k=1, mode="keyword")


def test_reference_guard_triggers_on_named_work_not_on_everyday_questions():
    for q in ["What are the two variants of BQ-LoRA called?", "What serves as the noisy teacher in Ambient SFT?",
              "How did the E4B pass rate change?", "What did the Kernel Reality Check paper find?",
              "Is Gemma 4 better than Gemma 3?"]:
        assert persona.needs_reference_guard(q), q
    for q in ["How does a hash table work?", "Why is my CUDA kernel slow on the GPU?", "Explain SGD with momentum.",
              "What's the time complexity of quicksort?", "How do I parse JSON in Python?"]:
        assert not persona.needs_reference_guard(q), q
    p = persona.system_prompt(tools=["arxiv_search"], guard=True)
    assert "do not actually know it" in p and "arxiv_search to look it up" in p
    assert "do not actually know it" not in persona.system_prompt(tools=["arxiv_search"])


def test_citation_support_check_catches_numbers_from_memory():
    """The live failure: Adam settings 'cited' to the LoRA introduction."""
    from alice.library import unsupported_numbers
    passages = {"L4": "LoRA freezes the pretrained weights and injects trainable rank decomposition matrices.",
                "L7": "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10^-9, "
                      "with warmup_steps = 4000."}
    reply = ("Adam with a learning rate of **0.0001**, beta2 = 0.98 [L4]. "
             "They used beta2 = 0.98 and 4,000 warmup steps [L7]. Base models trained for 12 hours.")
    flagged = unsupported_numbers(reply, passages)
    assert ("0.0001", "[L4]") in flagged and ("0.98", "[L4]") in flagged
    assert not any(labels == "[L7]" for _, labels in flagged)             # 0.98 and 4,000 are in L7
    assert not any(num == "12" for num, _ in flagged)                    # uncited sentence: not checked
    assert unsupported_numbers("Two tasks [L4].", passages) == []         # small integers are skipped


def test_library_can_be_searched_from_another_thread(tmp_path):
    """Regression: the chat box opens the library on the UI thread and searches it on a worker thread."""
    import threading
    (tmp_path / "n.md").write_text("# Notes\n\nThe warmup lasts 4000 steps. " * 5, encoding="utf-8")
    lib = Library(tmp_path / "lib")
    lib.add(tmp_path / "n.md", FakeEmbedder())
    found, errors = [], []

    def worker():
        try:
            found.extend(lib.search("warmup steps", FakeEmbedder(), k=1))
        except Exception as e:
            errors.append(e)
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not errors and found


def test_citations_in_parentheses_are_recognised(lib):
    library, emb = lib
    s = LibrarySession(library, emb)
    s.search_library("attention heads", k=1)
    rep = s.check("From your copy of the paper (L1), there are eight heads.")
    assert [l for l, _ in rep.cited] == ["L1"] and rep.unknown == []


def test_paper_references_and_scoped_search(tmp_path):
    """'the Transformer paper' style nicknames: shortest phrase first, scoped search inside that paper."""
    from alice.library import paper_references
    phrases = [p for p, _ in paper_references("How deep was the deepest network the ResNet paper trained?")]
    assert phrases[0] == "ResNet"
    assert [p for p, _ in paper_references('What does "Attention Is All You Need" propose?')][0] == \
        "Attention Is All You Need"
    assert paper_references("How does a hash table work?") == []

    # Realistic shape: the paper behind the nickname uses it throughout, others mention it now and then.
    (tmp_path / "attn.md").write_text(
        "# Attention Is All You Need\n\n## Model\n\nThe Transformer uses stacked self-attention layers.\n\n"
        "## Encoder\n\nThe Transformer encoder has six identical layers.\n\n"
        "## Results\n\nThe Transformer beats recurrent models on translation.\n\n"
        "## Optimizer\n\nWe used Adam with beta2 0.98 and a warmup of 4000 steps.\n", encoding="utf-8")
    (tmp_path / "bert.md").write_text(
        "# BERT\n\n## Model\n\nBERT is a bidirectional Transformer encoder.\n\n## Pre-training\n\nWe mask "
        "fifteen percent of the input tokens at random.\n\n## Data\n\nWe use BooksCorpus and English Wikipedia "
        "text.\n\n## Optimizer\n\nWe used Adam with a learning rate of 1e-4 and the usual optimizer settings.\n",
        encoding="utf-8")
    lib = Library(tmp_path / "lib")
    emb = FakeEmbedder()
    lib.add(tmp_path / "attn.md", emb)
    lib.add(tmp_path / "bert.md", emb)
    scope = lib.resolve_reference("Which optimizer settings did the Transformer paper use?")
    assert scope and scope.title == "Attention Is All You Need" and scope.how == "density"
    assert scope.rest == "Which optimizer settings did use?"
    hits, sc = lib.search_with_scope("Which optimizer settings did the Transformer paper use?", emb, k=3)
    assert sc and all(h.title == "Attention Is All You Need" for h in hits)
    assert lib.resolve_reference("What is a Transformer?") is None              # no paper named: global
