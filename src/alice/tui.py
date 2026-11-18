"""ALICE chat box: a full-screen terminal UI (Textual).

    alice tui [--model NAME] [--think] [--workspace DIR] [--no-tools] [--offline] [--no-transcript]

Scrolling conversation with Markdown-rendered replies (code blocks, bold, lists) streamed as they
arrive, an input box, tool activity / sources / warnings inline, and a dialog for run_python
permission. Keys: Enter send, Esc stop the reply, Ctrl+T thinking on/off, Ctrl+R new conversation,
Ctrl+L clear the screen, Ctrl+Q quit. The same /commands as the line-based CLI work in the input box.
"""
from __future__ import annotations

import json
import threading
import time

from rich.syntax import Syntax
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown, Static

from .cli import _short, chat_arguments, session_from_args, tool_summary, turn_stats
from .llm import Event, LLMError
from .session import StreamFilter
from .tools import Tool, ToolResult

UPDATE_EVERY_S = 0.08          # throttle Markdown re-renders while streaming


class ConfirmRun(ModalScreen[bool]):
    """Ask before running model-written code."""

    BINDINGS = [Binding("y", "answer(True)", "Run"), Binding("n", "answer(False)", "Don't run"),
                Binding("escape", "answer(False)", "Don't run", show=False)]

    def __init__(self, tool_name: str, code: str):
        super().__init__()
        self.tool_name, self.code = tool_name, code

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"ALICE wants to run [b]{self.tool_name}[/b]. It runs as you, on this computer.")
            with VerticalScroll(id="code"):
                yield Static(Syntax(self.code, "python", line_numbers=True, word_wrap=True))
            with Horizontal(id="buttons"):
                yield Button("Run it (y)", variant="success", id="yes")
                yield Button("Don't run (n)", variant="error", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, run: bool) -> None:
        self.dismiss(run)


class AliceApp(App):
    TITLE = "ALICE"
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    .user { margin: 1 0 0 10; padding: 0 1; border: round $accent; }
    .alice { margin: 1 10 0 0; padding: 0 1; border: round $success; }
    .tool { color: $text-muted; margin: 0 0 0 2; }
    .sources { color: $text-muted; margin: 0 0 0 2; }
    .warning { color: $warning; margin: 0 0 0 2; }
    .info { color: $text-muted; text-style: italic; margin: 1 0 0 0; }
    .thinking { color: $text-muted; text-style: italic; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #prompt { margin: 0 0; }
    ConfirmRun { align: center middle; }
    #dialog { width: 90%; max-width: 110; height: auto; max-height: 85%; padding: 1 2;
              border: thick $warning; background: $surface; }
    #code { height: auto; max-height: 24; margin: 1 0; }
    #buttons { height: auto; align-horizontal: right; }
    #buttons Button { margin-left: 2; }
    """
    BINDINGS = [
        Binding("escape", "stop", "Stop reply"),
        Binding("ctrl+t", "toggle_thinking", "Thinking on/off"),
        Binding("ctrl+r", "new_conversation", "New conversation"),
        Binding("ctrl+l", "clear_screen", "Clear screen"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.ready = False
        self.busy = False
        self.cancel = threading.Event()
        self.closing = False
        self.last_stats = ""

    # ----- layout
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Static("", id="status", markup=False)
        yield Input(placeholder="Loading the model...", id="prompt", disabled=True)
        yield Footer()

    @property
    def log_view(self) -> VerticalScroll:
        return self.query_one("#log", VerticalScroll)

    def on_mount(self) -> None:
        self.sub_title = self.session.profile.name
        for note in self.session.startup_notes:
            self.add_line(note, "warning")
        p = self.session.profile
        self.add_line(f"Loading {p.gguf} ({p.note}) ... the first start takes ~20 s.", "info")
        self.refresh_status()
        self.start_session()

    def refresh_status(self) -> None:
        s = self.session
        parts = [s.profile.gguf.removesuffix(".gguf"), f"thinking {'on' if s.thinking else 'off'}",
                 f"{len(s.tool_names)} tools" if s.tool_names else "tools off"]
        if s.library:
            parts.append(f"library {s.library_size} docs")
        if self.busy:
            parts.append("answering... (Esc to stop)")
        elif self.last_stats:
            parts.append("last " + self.last_stats.strip("[]"))
        self.query_one("#status", Static).update("  ·  ".join(parts))

    # ----- adding things to the conversation (main thread)
    def add_line(self, text: str, kind: str) -> Static:
        w = Static(text, classes=kind, markup=False)
        self.log_view.mount(w)
        self.log_view.scroll_end(animate=False)
        return w

    def add_user(self, text: str) -> None:
        w = Static(text, classes="user", markup=False)
        w.border_title = "you"
        self.log_view.mount(w)
        self.log_view.scroll_end(animate=False)

    async def new_reply(self) -> Markdown:
        md = Markdown("", classes="alice")
        md.border_title = "ALICE"
        await self.log_view.mount(md)
        self.log_view.scroll_end(animate=False)
        return md

    async def set_reply(self, md: Markdown, text: str) -> None:
        await md.update(text)
        self.log_view.scroll_end(animate=False)

    # ----- startup
    @work(thread=True, exclusive=True, group="startup")
    def start_session(self) -> None:
        try:
            started = self.session.start()
        except (LLMError, ValueError) as e:
            self.call_from_thread(self.add_line, f"error: {e}", "warning")
            self.call_from_thread(self.add_line, "Fix the problem and restart (Ctrl+Q quits).", "info")
            return
        self.call_from_thread(self.session_ready, started)

    # NB: not "on_ready" - Textual calls on_ready() itself for its built-in Ready event (first paint),
    # which enabled the input before the model had loaded (found in the first live run).
    def session_ready(self, started: bool) -> None:
        self.ready = True
        s = self.session
        msg = "ALICE is ready." + ("" if started else " (reusing a running server)")
        msg += f"  Tools: {', '.join(s.tool_names) or 'off'}."
        if s.library:
            msg += f"  Library: {s.library_size} documents (searched automatically)."
        self.add_line(msg, "info")
        self.add_line("Enter sends · Esc stops a reply · Ctrl+T thinking · Ctrl+R new conversation · "
                      "/help for commands", "info")
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.placeholder = "Ask ALICE anything...  (/help for commands)"
        prompt.focus()
        self.refresh_status()

    # ----- input
    @on(Input.Submitted, "#prompt")
    def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self.ready:
            return
        if self.busy:
            self.notify("ALICE is still answering. Press Esc to stop her first.", severity="warning")
            return
        event.input.value = ""
        if text == "/clear":
            self.action_clear_screen()
            return
        cmd = self.session.command(text)
        if cmd is not None:
            if cmd.quit:
                self.action_quit()
                return
            self.add_line(cmd.message, "info")
            self.refresh_status()
            return
        self.add_user(text)
        self.busy = True
        self.cancel.clear()
        self.refresh_status()
        self.answer(text)

    # ----- one turn (worker thread)
    @work(thread=True, exclusive=True, group="turn")
    def answer(self, text: str) -> None:
        ui = _TurnRenderer(self)
        try:
            outcome = self.session.ask(text, on_event=ui.event, on_tool_start=ui.tool_start,
                                       on_tool=ui.tool_done, confirm=self.confirm_from_thread,
                                       on_library=ui.library, cancel=self.cancel)
            ui.finish(outcome)
        except Exception as e:                            # never leave the UI stuck in "busy"
            ui._call(self.add_line, f"error: {type(e).__name__}: {e}", "warning")
        finally:
            ui._call(self.turn_done)

    def turn_done(self) -> None:
        self.busy = False
        self.refresh_status()

    def confirm_from_thread(self, tool: Tool, args: dict) -> bool:
        """Called by the agent (worker thread) before running code: show the dialog and wait."""
        answer: dict = {}
        done = threading.Event()
        code = str(args.get("code", json.dumps(args, indent=2)))

        def show() -> None:
            self.push_screen(ConfirmRun(tool.name, code), callback=lambda ok: (answer.update(ok=bool(ok)), done.set()))
        self.call_from_thread(show)
        while not done.wait(0.2):
            if self.closing or self.cancel.is_set():
                return False
        return answer.get("ok", False)

    # ----- actions
    def action_stop(self) -> None:
        if self.busy:
            self.cancel.set()
            self.notify("Stopping...")

    def action_toggle_thinking(self) -> None:
        cmd = self.session.command("/think off" if self.session.thinking else "/think on")
        self.add_line(cmd.message, "info")
        self.refresh_status()

    def action_new_conversation(self) -> None:
        if self.busy:
            self.notify("Stop the current reply first (Esc).", severity="warning")
            return
        if self.ready:
            self.action_clear_screen()
            self.add_line(self.session.command("/reset").message, "info")

    def action_clear_screen(self) -> None:
        self.log_view.remove_children()

    async def action_quit(self) -> None:
        self.closing = True
        self.cancel.set()
        self.exit()


class _TurnRenderer:
    """Turns agent callbacks (called on the worker thread) into widgets on the UI thread."""

    def __init__(self, app: AliceApp):
        self.app = app
        self.md: Markdown | None = None
        self.text = ""
        self.filter = StreamFilter()
        self.last_update = 0.0
        self.thinking: Static | None = None
        self.reasoning_chars = 0

    def _call(self, fn, *args):
        if self.app.closing:
            return None
        try:
            return self.app.call_from_thread(fn, *args)
        except RuntimeError:                              # the app quit while ALICE was answering
            return None

    def library(self, places: list[str]) -> None:
        self._call(self.app.add_line, "library: " + "; ".join(places), "sources")

    def event(self, ev: Event) -> None:
        if ev.kind == "reasoning":
            self.reasoning_chars += len(ev.text)
            dots = "." * min(40, 1 + self.reasoning_chars // 300)
            if self.thinking is None:
                self.thinking = self._call(self.app.add_line, "thinking" + dots, "thinking")
            elif time.monotonic() - self.last_update > 0.3:
                self.last_update = time.monotonic()
                self._call(self.thinking.update, "thinking" + dots)
        elif ev.kind == "content":
            visible = self.filter.feed(ev.text)
            if not visible:
                return
            if self.md is None:
                self.md = self._call(self.app.new_reply)
            self.text += visible
            if time.monotonic() - self.last_update > UPDATE_EVERY_S:
                self.last_update = time.monotonic()
                self._call(self.app.set_reply, self.md, self.text)
        elif ev.kind in ("tool_calls", "finish"):
            self._flush()

    def _flush(self) -> None:
        rest = self.filter.flush()
        if rest and self.md is None:
            self.md = self._call(self.app.new_reply)
        self.text += rest
        if self.md is not None:
            self._call(self.app.set_reply, self.md, self.text)

    def tool_start(self, name: str, raw_args: str) -> None:
        self._flush()
        self.md, self.text, self.filter = None, "", StreamFilter()      # next round gets a new bubble
        self._call(self.app.add_line, f"▸ {name}: {tool_summary(name, raw_args)}", "tool")

    def tool_done(self, tr: ToolResult) -> None:
        mark = "declined" if tr.declined else ("ok" if tr.ok else "failed")
        self._call(self.app.add_line, f"  ◂ {mark}: {_short(tr.output, 200)}", "tool")

    def finish(self, outcome) -> None:
        self._flush()
        if outcome.error:
            self._call(self.app.add_line, f"error: {outcome.error}", "warning")
            return
        result = outcome.result
        if result and result.finish == "tool_loop" and self.md is None:
            md = self._call(self.app.new_reply)
            self._call(self.app.set_reply, md, result.reply)
        for n in outcome.notices:
            self._call(self.app.add_line, n.text, n.kind)
        if result and outcome.kept:
            self.app.last_stats = turn_stats(result)


def main(argv: list[str] | None = None) -> int:
    args = chat_arguments("alice tui").parse_args(argv)
    session = session_from_args(args)
    app = AliceApp(session)
    try:
        app.run()
    finally:
        session.stop()
    if session.transcript:
        print(f"(transcript saved to {session.transcript.path})")
    return 0
