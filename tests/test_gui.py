"""The GUI back end (no model, no window): token and origin checks, a streamed turn, the run-confirmation round
trip, commands, uploads and running a code block, over the real WebSocket protocol."""
import time

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from alice.agent import TurnResult  # noqa: E402
from alice.config import Profile  # noqa: E402
from alice.gui import server  # noqa: E402
from alice.llm import Event  # noqa: E402
from alice.session import CommandResult, Notice, TurnOutcome  # noqa: E402
from alice.tools import Tool  # noqa: E402

TOKEN = "test-token"
ORIGIN = {"origin": "http://testserver"}


class FakeSession:
    def __init__(self):
        self.profile = Profile("default", "fake-model.gguf")
        self.thinking, self.tool_names, self.library, self.library_size = False, ["calculate", "run_code"], None, 0
        self.memory, self.claude, self.document, self.background = None, None, None, None
        self.asked, self.confirmed = [], None

    def start(self):
        return True

    def command(self, text):
        if not text.startswith("/"):
            return None
        if text.startswith("/read"):
            return CommandResult("(read notes.pdf (2 pages, PDF); the whole text goes along with your question)",
                                 ask="Give me a short overview of this document.")
        return CommandResult("help text" if text == "/help" else "(conversation cleared)")

    def ask(self, text, *, on_event, on_tool_start, on_tool, confirm, on_library, on_arxiv, cancel):
        self.asked.append(text)
        on_event(Event("content", "Let me **check**."))
        if "run" in text:
            on_tool_start("run_code", '{"language": "python", "code": "print(6*7)"}')
            self.confirmed = confirm(Tool("run_code", "", {}, lambda **k: "", needs_confirmation=True),
                                     {"language": "python", "code": "print(6*7)"})
        on_event(Event("content", " It is 42."))
        on_event(Event("finish", "stop", {}))
        result = TurnResult(reply="Let me **check**. It is 42.", messages=[], finish="stop", seconds=1.5,
                            gen_tokens=8, timings={})
        return TurnOutcome(result=result, notices=[Notice("sources", "Sources: [L1] Fake, p. 1")])


@pytest.fixture
def gui(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    session = FakeSession()
    ctl = server.Controller(session)
    for _ in range(100):
        if ctl.ready:
            break
        time.sleep(0.02)
    return TestClient(server.create_app(ctl, TOKEN)), session


def until(ws, kind, limit=60):
    """Receive messages until one of `kind`; returns (that message, everything received)."""
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == kind:
            return m, seen
    raise AssertionError(f"no {kind} message; got {[m['type'] for m in seen]}")


def test_the_page_and_socket_need_the_token_and_our_origin(gui):
    client, _ = gui
    assert client.get("/").status_code == 403
    page = client.get(f"/?token={TOKEN}")
    assert page.status_code == 200 and "<title>ALICE</title>" in page.text
    assert client.post("/upload?token=wrong", files={"file": ("a.txt", b"x")}).status_code == 403
    from starlette.websockets import WebSocketDisconnect
    for url, headers in ((f"/ws?token=wrong", ORIGIN), (f"/ws?token={TOKEN}", {"origin": "http://evil.example"})):
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(url, headers=headers):
                pass
        assert closed.value.code == 4403


def test_a_streamed_turn(gui):
    client, session = gui
    with client.websocket_connect(f"/ws?token={TOKEN}", headers=ORIGIN) as ws:
        ws.send_json({"type": "hello"})
        status, _ = until(ws, "status")
        assert status["ready"] and status["model"] == "fake-model" and "Commands" in status["help"]
        ws.send_json({"type": "send", "text": "what is 6*7?"})
        end, seen = until(ws, "turn_end")
        text = "".join(m["text"] for m in seen if m["type"] == "delta")
        assert text == "Let me **check**. It is 42." and session.asked == ["what is 6*7?"]
        assert {"type": "user", "text": "what is 6*7?"} in seen
        assert end["kept"] and end["notices"] == [{"kind": "sources", "text": "Sources: [L1] Fake, p. 1"}]
        assert "1.5s" in end["stats"]


def test_running_code_asks_the_window_first(gui):
    client, session = gui
    with client.websocket_connect(f"/ws?token={TOKEN}", headers=ORIGIN) as ws:
        ws.send_json({"type": "send", "text": "please run it"})
        ask, seen = until(ws, "confirm")
        assert ask["language"] == "python" and ask["code"] == "print(6*7)"
        assert any(m["type"] == "tool_start" and m["name"] == "run_code" for m in seen)
        ws.send_json({"type": "confirm_reply", "id": ask["id"], "ok": True})
        until(ws, "turn_end")
        assert session.confirmed is True


def test_commands_and_reading_an_uploaded_file(gui, tmp_path):
    client, session = gui
    with client.websocket_connect(f"/ws?token={TOKEN}", headers=ORIGIN) as ws:
        ws.send_json({"type": "send", "text": "/help"})
        info, _ = until(ws, "info")
        assert info["text"] == "help text"
        r = client.post(f"/upload?token={TOKEN}", files={"file": ("my notes.pdf", b"%PDF-1.4 test")})
        path = r.json()["path"]
        assert r.status_code == 200 and path.endswith("_my notes.pdf") and (tmp_path / "uploads").exists()
        ws.send_json({"type": "send", "text": f'/read "{path}"'})
        info, _ = until(ws, "info")
        assert "whole text goes along" in info["text"]
        until(ws, "turn_end")
        assert session.asked == ["Give me a short overview of this document."]


def test_a_code_block_can_be_run_from_the_window(gui):
    client, _ = gui
    with client.websocket_connect(f"/ws?token={TOKEN}", headers=ORIGIN) as ws:
        ws.send_json({"type": "run", "block": "b1", "language": "python", "code": "print(input()[::-1])",
                      "stdin": "ECILA\n"})
        result, _ = until(ws, "run_result")
        assert result["block"] == "b1" and result["output"].endswith("ALICE")
