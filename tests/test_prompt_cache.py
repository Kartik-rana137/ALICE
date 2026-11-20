"""Saving / restoring the processed fixed part of the prompt (llm.prime_prefix) against a fake server."""
import os
import time

from alice.config import Profile
from alice.llm import LLMError, prime_prefix

PROFILE = Profile("default", "fake.gguf")
MARKER = "Today's date:"


class FakeServer:
    def __init__(self, cache_dir, prompt="<sys>persona and tools\n\nToday's date: 2026-11-20.<user>hi",
                 restore_count=None):
        self.cache_dir, self.prompt, self.calls = cache_dir, prompt, []
        self.restore_count = restore_count

    def __call__(self, path, payload, timeout=600):
        self.calls.append(path.split("?")[0] + (("?" + path.split("?")[1]) if "?" in path else ""))
        if path == "/apply-template":
            return {"prompt": self.prompt}
        if path == "/tokenize":
            return {"tokens": list(range(len(payload["content"])))}
        if path == "/completion":
            assert payload["n_predict"] == 0 and payload["cache_prompt"] is True
            return {"timings": {"prompt_n": len(payload["prompt"])}}
        if path == "/slots/0?action=save":
            (self.cache_dir / payload["filename"]).write_bytes(b"state")
            return {"n_saved": 1}
        if path == "/slots/0?action=restore":
            if self.restore_count == "error":
                raise LLMError("server returned 400: bad file")
            n = self.restore_count if self.restore_count is not None else self.prompt.index(MARKER)
            return {"n_restored": n}
        raise AssertionError(path)


def test_first_start_computes_and_saves_then_later_starts_restore(tmp_path):
    srv = FakeServer(tmp_path)
    first = prime_prefix([], PROFILE, None, MARKER, tmp_path, post=srv)
    assert first["status"] == "saved" and first["tokens"] == srv.prompt.index(MARKER)
    assert "/completion" in srv.calls and "/slots/0?action=save" in srv.calls
    srv.calls.clear()
    second = prime_prefix([], PROFILE, None, MARKER, tmp_path, post=srv)
    assert second["status"] == "restored" and "/completion" not in srv.calls


def test_a_changed_prompt_gets_its_own_file_and_old_files_are_pruned(tmp_path):
    for i in range(4):
        srv = FakeServer(tmp_path, prompt=f"<sys>persona v{i} " + "x" * i + "\n\nToday's date: 2026-11-20.")
        assert prime_prefix([], PROFILE, None, MARKER, tmp_path, post=srv)["status"] == "saved"
        time.sleep(0.02)
    assert len(list(tmp_path.glob("prefix-*.bin"))) == 2                 # only the newest two kept


def test_bad_or_partial_restore_falls_back_to_computing(tmp_path):
    prime_prefix([], PROFILE, None, MARKER, tmp_path, post=FakeServer(tmp_path))
    for bad in ("error", 5):
        srv = FakeServer(tmp_path, restore_count=bad)
        assert prime_prefix([], PROFILE, None, MARKER, tmp_path, post=srv)["status"] == "saved"
        assert "/completion" in srv.calls


def test_no_marker_means_nothing_is_cached(tmp_path):
    srv = FakeServer(tmp_path, prompt="<sys>a prompt without the marker")
    assert prime_prefix([], PROFILE, None, MARKER, tmp_path, post=srv) == {"status": "no-marker"}
    assert not os.listdir(tmp_path)
