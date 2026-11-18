"""Post-checks in ChatSession (no model needed)."""
import pytest

from alice.agent import TurnResult
from alice.session import ChatSession
from alice.tools import ToolResult


@pytest.mark.parametrize("reply,tools,prefetched,flagged", [
    # real fabricated cases from the 2026 eval (no tool was called)
    ("I've searched for this specific metric and result combination. I cannot find any benchmarks.", [], False, True),
    ("It doesn't appear in my training data or arXiv search results, and there's no record of it.", [], False, True),
    ("I ran the code and it prints 42.", [], False, True),
    # genuine: a search tool or library prefetch really happened
    ("The search results are mostly about Transformers and CNNs.", ["search_library"], False, False),
    ("My search found the warmup schedule in section 5.3.", [], True, False),
    # not a claim of tool use
    ("Let me check the formula: the gradient is 2x.", [], False, False),
    ("I checked the derivation by hand and it holds.", [], False, False),
    ("Binary search runs in O(log n).", [], False, False),
])
def test_claimed_tool_use_check(reply, tools, prefetched, flagged):
    result = TurnResult(reply=reply, messages=[],
                        tool_results=[ToolResult(t, {}, "ok", True) for t in tools])
    notices = []
    assert ChatSession._check_claimed_tools(result, prefetched, notices) is flagged
    assert bool(notices) is flagged


@pytest.mark.parametrize("question,query", [
    ("What are the two variants of BQ-LoRA called?", "BQ-LoRA"),
    ("In the paper on fine-tuning Gemma 4 for MATLAB signal-processing code, how did the pass rate change?",
     "fine-tuning Gemma 4 for MATLAB signal-processing code"),
    ("What did the Kernel Reality Check paper find?", "Kernel Reality Check"),
    ("How does a hash table work?", None),                        # nothing named: no lookup at all
])
def test_arxiv_lookup_sends_only_the_name(question, query):
    from alice.session import arxiv_lookup_query
    assert arxiv_lookup_query(question) == query


def test_auto_arxiv_is_off_by_default_and_toggled_by_command(monkeypatch):
    from alice import config
    from alice.session import ChatSession
    assert config.ARXIV_AUTO is False or "ALICE_ARXIV_AUTO" in __import__("os").environ
    s = ChatSession.__new__(ChatSession)            # no tools/server needed for the command logic
    s.auto_arxiv, s.transcript, s.toolbox = False, None, None
    assert "arXiv tools are off" in s.command("/arxiv auto on").message
    from alice.tools import ToolBox, arxiv
    s.toolbox = ToolBox().add(arxiv.SEARCH)
    assert "on" in s.command("/arxiv auto on").message and s.auto_arxiv
    assert "off" in s.command("/arxiv auto off").message and not s.auto_arxiv


@pytest.mark.parametrize("question,query", [
    ("According to the Kernel Reality Check paper, in which setting is FlashAttention slower?", "Kernel Reality Check"),
    ("What serves as the noisy teacher in Ambient SFT?", "Ambient SFT"),
    ("Explain SGD with momentum.", None),
])
def test_arxiv_lookup_query_cleanup(question, query):
    """Snags from the first auto-arXiv eval: 'the ... paper' wrapper sent along; 'SFT' sent without 'Ambient'."""
    from alice.session import arxiv_lookup_query
    assert arxiv_lookup_query(question) == query
