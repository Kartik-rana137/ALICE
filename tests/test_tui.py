"""Headless test of the chat box (no model): streaming, tool lines, the run_python dialog, notices,
leaked-markup filtering, commands and the Stop key."""
import asyncio
import threading

from textual.widgets import Input, Markdown, Static

from alice.agent import TurnResult
from alice.config import Profile
from alice.llm import Event
from alice.session import CommandResult, Notice, TurnOutcome
from alice.tools import Tool, ToolResult
from alice.tui import AliceApp, ConfirmRun


class FakeSession:
    def __init__(self, block: threading.Event | None = None):
        self.profile = Profile("default", "fake-model.gguf", note="fake")
        self.thinking = False
        self.tool_names = ["calculate", "run_python"]
        self.library = None
        self.library_size = 0
        self.startup_notes = []
        self.transcript = None
        self.asked, self.confirmed, self.stopped = [], None, False
        self.block = block

    def start(self):
        return True

    def stop(self):
        self.stopped = True

    def command(self, text):
        if not text.startswith("/"):
            return None
        if text == "/quit":
            return CommandResult(quit=True)
        if text.startswith("/think"):
            self.thinking = text.endswith("on")
            return CommandResult(f"(thinking {'on' if self.thinking else 'off'})")
        return CommandResult("(conversation cleared)", reset=True) if text == "/reset" else CommandResult("help text")

    def ask(self, text, *, on_event, on_tool_start, on_tool, confirm, on_library, cancel):
        self.asked.append(text)
        if self.block is not None:                       # simulate a long reply until Stop is pressed
            on_event(Event("content", "Thinking about it"))
            while not cancel.is_set():
                cancel.wait(0.05)
            return TurnOutcome(interrupted=True, notices=[Notice("info", "(stopped; this exchange was not kept)")])
        on_library(["Fake Paper, p. 1"])
        on_event(Event("content", "Let me check **6 x 7**."))
        on_tool_start("run_python", '{"code": "print(6*7)"}')
        self.confirmed = confirm(Tool("run_python", "", {}, lambda code: "", needs_confirmation=True),
                                 {"code": "print(6*7)"})
        on_tool(ToolResult("run_python", {"code": "print(6*7)"}, "exit code 0\n42", True))
        on_event(Event("content", "It is **42** [L1]. <tool"))
        on_event(Event("content", "_call>junk</tool_call>"))
        on_event(Event("finish", "stop", {"predicted_n": 9, "predicted_per_second": 30.0}))
        result = TurnResult(reply="It is **42** [L1].", messages=[], finish="stop", seconds=1.2, gen_tokens=9,
                            timings={"predicted_per_second": 30.0})
        return TurnOutcome(result=result, notices=[Notice("sources", "Sources: [L1] Fake Paper, p. 1")])


async def _wait(pilot, cond, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        if cond():
            return True
        await pilot.pause(0.05)
    return cond()


def _texts(app):
    out = []
    for w in app.query_one("#log").children:
        out.append(w.source if isinstance(w, Markdown) else str(w.content))
    return out


def test_chat_box_full_turn_with_dialog_and_notices():
    async def scenario():
        session = FakeSession()
        app = AliceApp(session)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: app.ready)
            assert not app.query_one("#prompt", Input).disabled
            app.query_one("#prompt", Input).value = "What is 6*7?"
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, ConfirmRun))
            await pilot.press("y")                                    # approve the code run
            assert await _wait(pilot, lambda: not app.busy)
            await pilot.pause(0.1)
            texts = _texts(app)
            assert session.asked == ["What is 6*7?"] and session.confirmed is True
            assert "What is 6*7?" in texts                              # user bubble
            assert any(t.startswith("library: Fake Paper") for t in texts)
            assert "Let me check **6 x 7**." in texts                    # first round, before the tool
            assert any(t.startswith("▸ run_python") for t in texts) and any("◂ ok" in t for t in texts)
            assert "It is **42** [L1]. " in texts                        # leaked <tool_call> markup filtered
            assert not any("tool_call" in t for t in texts)
            assert "Sources: [L1] Fake Paper, p. 1" in texts
            assert "last 1.2s" in str(app.query_one("#status", Static).content)
    asyncio.run(scenario())


def test_chat_box_commands_and_declining_code():
    async def scenario():
        session = FakeSession()
        app = AliceApp(session)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: app.ready)
            await pilot.press("ctrl+t")
            assert session.thinking and "thinking on" in str(app.query_one("#status", Static).content)
            app.query_one("#prompt", Input).value = "run something"
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, ConfirmRun))
            await pilot.press("n")
            assert await _wait(pilot, lambda: not app.busy)
            assert session.confirmed is False
            await pilot.press("ctrl+r")
            await pilot.pause(0.1)
            assert _texts(app) == ["(conversation cleared)"]
    asyncio.run(scenario())


def test_input_stays_disabled_until_the_model_has_loaded():
    """Regression: a handler named on_ready() was also called by Textual's own Ready event (first paint),
    enabling the input before the model loaded; questions then hit an unstarted session."""
    release = threading.Event()

    class SlowStart(FakeSession):
        def start(self):
            release.wait(5)
            return True

    async def scenario():
        app = AliceApp(SlowStart())
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.5)                          # well past the first paint
            assert not app.ready and app.query_one("#prompt", Input).disabled
            release.set()
            assert await _wait(pilot, lambda: app.ready)
            assert not app.query_one("#prompt", Input).disabled
    asyncio.run(scenario())


def test_chat_box_stop_key_interrupts_a_reply():
    async def scenario():
        session = FakeSession(block=threading.Event())
        app = AliceApp(session)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _wait(pilot, lambda: app.ready)
            app.query_one("#prompt", Input).value = "long question"
            await pilot.press("enter")
            assert await _wait(pilot, lambda: app.busy)
            await pilot.pause(0.2)
            await pilot.press("escape")
            assert await _wait(pilot, lambda: not app.busy)
            assert "(stopped; this exchange was not kept)" in _texts(app)
    asyncio.run(scenario())
