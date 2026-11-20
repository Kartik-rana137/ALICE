"""ALICE chat box: a full-screen terminal UI (Textual).

    alice tui [--model NAME] [--think] [--workspace DIR] [--no-tools] [--offline] [--no-memory] [--claude]

Scrolling conversation with Markdown-rendered replies streamed as they arrive; a status line that says what
ALICE is doing right now (reading your library, thinking, using a tool, writing) and for how long; tool
activity, sources and warnings inline; dialogs before code runs and before anything is sent to Claude.

Keys: Enter send · Shift/Ctrl+Enter new line · Up/Down earlier messages · Esc stop the reply ·
Ctrl+O copy the last answer · Ctrl+G ask Claude · Ctrl+T thinking on/off · Ctrl+R new conversation ·
Ctrl+L clear the screen · PgUp/PgDn scroll · Ctrl+Q quit. The same /commands as the line-based chat work,
plus /copy [code] and /clear.
"""
from __future__ import annotations

import json
import re
import threading
import time

from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, Markdown, Static, TextArea

from .cli import _short, chat_arguments, save_memory_with_messages, session_from_args, tool_summary, turn_stats
from .llm import Event, LLMError
from .render import ascii_emoji, latex_to_text
from .session import StreamFilter
from .tools import Tool, ToolResult

UPDATE_EVERY_S = 0.08          # throttle Markdown re-renders while streaming
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
COMMANDS = {                   # shown as hints while typing a /command
    "/think": "on|off - step-by-step reasoning (slower, better at maths)",
    "/claude": "[question] - ask Claude: re-check the last answer, or a new question",
    "/read": "PATH [question] - read a PDF, Word or PowerPoint file (or drag it into the chat)",
    "/memory": "what ALICE remembers about you",
    "/remember": "TEXT - add a note to long-term memory",
    "/forget": "ID|all - delete memory notes",
    "/arxiv": "auto on|off - look up named papers on arXiv automatically",
    "/copy": "[code] - copy the last answer (or its last code block)",
    "/tools": "list the tools",
    "/reset": "start a new conversation",
    "/clear": "clear the screen",
    "/help": "all commands and keys",
    "/quit": "leave",
}
KEYS_HELP = ("Keys: Enter send · Shift+Enter new line · Up/Down earlier messages · Esc stop · Ctrl+O copy "
             "answer · Ctrl+G ask Claude · Ctrl+T thinking · Ctrl+R new conversation · Ctrl+L clear · Ctrl+Q quit")
PROGRAM_LINES = 200            # a program's box shows its last 200 lines


def program_text(parts: list[tuple[str, str]], max_lines: int = PROGRAM_LINES) -> Text:
    """A program's output so far (stderr in red), the last `max_lines` lines of it."""
    lines: list[list[str]] = []                        # [stream, line]
    for stream, chunk in parts:
        for piece in chunk.splitlines(keepends=True):
            if lines and lines[-1][0] == stream and not lines[-1][1].endswith("\n"):
                lines[-1][1] += piece                  # the rest of a line printed in two goes
            else:
                lines.append([stream, piece])
    text = Text()
    if len(lines) > max_lines:
        text.append(f"... {len(lines) - max_lines} earlier lines\n", style="dim")
        lines = lines[-max_lines:]
    for stream, line in lines:
        text.append(line, style="red" if stream == "stderr" else "")
    text.rstrip()
    return text


class ConfirmRun(ModalScreen[bool]):
    """Ask before running model-written code."""

    BINDINGS = [Binding("y", "answer(True)", "Run"), Binding("n", "answer(False)", "Don't run"),
                Binding("escape", "answer(False)", "Don't run", show=False)]

    LEXERS = {"python": "python", "c": "c", "cpp": "cpp", "c++": "cpp", "java": "java"}

    def __init__(self, tool_name: str, code: str, language: str = "python", stdin: str = ""):
        super().__init__()
        self.tool_name, self.code, self.language, self.stdin = tool_name, code, language or "python", stdin

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"ALICE wants to run [b]{self.language}[/b] code. It runs as you, on this computer.")
            with VerticalScroll(id="code"):
                yield Static(Syntax(self.code, self.LEXERS.get(self.language.lower(), "text"), line_numbers=True,
                                    word_wrap=True))
            if self.stdin:
                yield Static("Input it will get:\n" + self.stdin, markup=False, classes="summary")
            with Horizontal(id="buttons"):
                yield Button("Run it (y)", variant="success", id="yes")
                yield Button("Don't run (n)", variant="error", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, run: bool) -> None:
        self.dismiss(run)


class ConfirmEscalation(ModalScreen[bool]):
    """Show exactly what would be sent to Claude, and what it may cost, before sending."""

    BINDINGS = [Binding("y", "answer(True)", "Send"), Binding("n", "answer(False)", "Cancel"),
                Binding("escape", "answer(False)", "Cancel", show=False)]

    def __init__(self, summary: str, month_line: str, preview: str):
        super().__init__()
        self.summary, self.month_line, self.preview = summary, month_line, preview

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="escalation"):
            yield Label("[b]Ask Claude?[/b]  This leaves your computer and goes to Anthropic's API.")
            yield Static(self.summary, markup=False, classes="summary")
            yield Static(self.month_line, markup=False, classes="summary")
            yield Label("Exactly what is sent:", classes="muted")
            with VerticalScroll(id="code"):
                yield Static(self.preview, markup=False)
            with Horizontal(id="buttons"):
                yield Button("Send (y)", variant="primary", id="yes")
                yield Button("Cancel (n)", variant="default", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, send: bool) -> None:
        self.dismiss(send)


class PromptArea(TextArea):
    """The input box: Up/Down on the first/last line walk through earlier messages, like a shell."""

    BINDINGS = [Binding("up", "history(-1)", show=False), Binding("down", "history(1)", show=False)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list[str] = []
        self._pos: int | None = None
        self._draft = ""

    def remember(self, text: str) -> None:
        if text and (not self.sent or self.sent[-1] != text):
            self.sent.append(text)
        self._pos = None

    def action_history(self, step: int) -> None:
        row = self.cursor_location[0]
        inside = row > 0 if step < 0 else row < self.document.line_count - 1
        if inside or not self.sent or (self._pos is None and step > 0):
            (self.action_cursor_up if step < 0 else self.action_cursor_down)()
            return
        if self._pos is None:
            self._draft, self._pos = self.text, len(self.sent)
        new = self._pos + step
        if new < 0:
            return
        if new >= len(self.sent):
            self._pos, self.text = None, self._draft
        else:
            self._pos, self.text = new, self.sent[new]
        self.move_cursor(self.document.end)


class AliceApp(App):
    TITLE = "ALICE"
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    .welcome { margin: 1 0 0 0; padding: 0 1; border: round $primary; color: $text; }
    .user { margin: 1 0 0 10; padding: 0 1; border: round $accent; }
    .alice { margin: 1 10 0 0; padding: 0 1; border: round $success; }
    .claude { margin: 1 10 0 0; padding: 0 1; border: round $warning; }
    .tool { color: $text-muted; margin: 0 0 0 2; }
    .program { margin: 0 0 0 2; padding: 0 1; border: round $secondary; border-subtitle-color: $text-muted; }
    .program.ok { border-subtitle-color: $success; }
    .program.failed { border-subtitle-color: $error; }
    .sources { color: $text-muted; margin: 0 0 0 2; }
    .warning { color: $warning; margin: 0 0 0 2; }
    .escalate { color: $accent; margin: 0 0 0 2; }
    .memory { color: $text-muted; text-style: italic; margin: 0 0 0 2; }
    .info { color: $text-muted; text-style: italic; margin: 1 0 0 0; }
    .thinking { color: $text-muted; text-style: italic; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #hint { height: auto; max-height: 13; padding: 0 1; color: $text-muted; display: none; }
    #prompt { height: auto; min-height: 3; max-height: 10; }
    ConfirmRun, ConfirmEscalation { align: center middle; }
    #dialog { width: 90%; max-width: 110; height: auto; max-height: 85%; padding: 1 2;
              border: thick $warning; background: $surface; }
    #dialog.escalation { border: thick $primary; }
    #dialog .summary { margin: 1 0 0 0; }
    #dialog .muted { color: $text-muted; margin: 1 0 0 0; }
    #code { height: auto; max-height: 24; margin: 1 0; }
    #buttons { height: auto; align-horizontal: right; }
    #buttons Button { margin-left: 2; }
    """
    BINDINGS = [
        Binding("enter", "submit_prompt", "Send", priority=True, show=False),
        Binding("ctrl+enter", "insert_prompt_newline", "New line", priority=True, show=False),
        Binding("shift+enter", "insert_prompt_newline", "New line", priority=True, show=False),
        Binding("escape", "stop", "Stop reply"),
        Binding("ctrl+o", "copy_answer", "Copy answer", priority=True),
        Binding("ctrl+g", "ask_claude", "Ask Claude", priority=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking"),
        Binding("ctrl+r", "new_conversation", "New chat"),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
        Binding("pageup", "scroll_log(-1)", "Scroll up", priority=True, show=False),
        Binding("pagedown", "scroll_log(1)", "Scroll down", priority=True, show=False),
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
        self.last_answer = ""
        self.phase = ""                   # what the current turn is doing, for the status line
        self.turn_started = 0.0

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "ask_claude" and not getattr(self.session, "claude", None):
            return False                                  # hidden from the footer when escalation is off
        return True

    # ----- layout
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Static("", id="status", markup=False)
        yield Static("", id="hint", markup=False)
        yield PromptArea("", placeholder="Loading the model...", id="prompt", disabled=True,
                         show_line_numbers=False, soft_wrap=True)
        yield Footer()

    @property
    def log_view(self) -> VerticalScroll:
        return self.query_one("#log", VerticalScroll)

    @property
    def prompt(self) -> PromptArea:
        return self.query_one("#prompt", PromptArea)

    def on_mount(self) -> None:
        self.sub_title = self.session.profile.name
        for note in self.session.startup_notes:
            self.add_line(note, "warning")
        p = self.session.profile
        self.add_line(f"Loading {p.gguf.removesuffix('.gguf')} ({p.note}) ...", "info")
        self.turn_started = time.monotonic()
        self.refresh_status()
        self.set_interval(0.25, self.refresh_status)
        self.start_session()

    def refresh_status(self) -> None:
        s = self.session
        if not self.ready:
            waited = time.monotonic() - self.turn_started
            text = f"loading the model... {waited:.0f} s"
            if waited > 15:                       # first start after a reboot: the model comes off the disk
                text += "  (the first start after a reboot reads the whole model from disk; later starts are faster)"
        elif self.busy:
            text = f"{self.phase or 'working'}... {time.monotonic() - self.turn_started:.0f} s  ·  Esc stops"
        else:
            parts = [s.profile.gguf.removesuffix(".gguf"), f"thinking {'on' if s.thinking else 'off'}",
                     f"{len(s.tool_names)} tools" if s.tool_names else "tools off"]
            if s.library:
                parts.append(f"library {s.library_size} docs")
            if getattr(s, "memory", None):
                parts.append(f"memory {s.memory_size} notes")
            if getattr(s, "claude", None):
                parts.append("Claude on call")
            if getattr(s, "document", None) is not None:
                parts.append(f"open: {s.document.title[:40]}")
            background = getattr(s, "background", None)
            if background:
                parts.append(f"{background}...")
            elif self.last_stats:
                parts.append("last " + self.last_stats.strip("[]"))
            text = "  ·  ".join(parts)
        try:
            status = self.query_one("#status", Static)
        except NoMatches:                                 # the timer fired while the app was closing
            return
        if str(status.content) != text:
            status.update(text)

    # ----- adding things to the conversation (main thread)
    def _following(self) -> bool:
        """Is the log scrolled to the bottom? New output only scrolls the view if it was (so reading
        further up isn't interrupted)."""
        log = self.log_view
        return log.max_scroll_y - log.scroll_y <= 2

    def _mount(self, widget, force_scroll: bool = False):
        follow = force_scroll or self._following()
        result = self.log_view.mount(widget)
        if follow:
            self.log_view.scroll_end(animate=False)
        return result

    def add_line(self, text: str, kind: str) -> Static:
        w = Static(text, classes=kind, markup=False)
        self._mount(w)
        return w

    def add_program(self, title: str) -> Static:
        """A program ALICE runs: a box that fills with its output while it runs."""
        w = Static("", classes="program", markup=False)
        w.border_title = f"▶ {title} program"
        w.border_subtitle = "starting..."
        self._mount(w)
        return w

    def update_program(self, w: Static, parts: list[tuple[str, str]], status: str | None = None,
                       ok: bool | None = None) -> None:
        follow = self._following()
        w.update(program_text(parts) if parts or ok is None else Text("(no output)", style="dim"))
        if status:
            w.border_subtitle = status
        if ok is not None:
            w.add_class("ok" if ok else "failed")
        if follow:
            self.log_view.scroll_end(animate=False)

    def add_user(self, text: str) -> None:
        w = Static(text, classes="user", markup=False)
        w.border_title = "you"
        self._mount(w, force_scroll=True)

    async def new_reply(self, kind: str = "alice", title: str = "ALICE") -> Markdown:
        md = Markdown("", classes=kind)
        md.border_title = title
        follow = self._following()
        await self.log_view.mount(md)
        if follow:
            self.log_view.scroll_end(animate=False)
        return md

    async def set_reply(self, md: Markdown, text: str) -> None:
        follow = self._following()
        await md.update(ascii_emoji(latex_to_text(text)))   # display only; history and checks keep the original
        if follow:
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
        lines = ["ALICE is ready" + ("" if started else " (reusing a running model server)") + "."]
        have = [f"tools: {', '.join(s.tool_names)}" if s.tool_names else "tools off"]
        if s.library:
            have.append(f"your library: {s.library_size} documents, searched automatically")
        if getattr(s, "memory", None):
            n = s.memory_size
            have.append(f"memory: {n} note{'s' * (n != 1)} about you (/memory)" if n else "memory: on (/memory)")
        if getattr(s, "claude", None):
            have.append("Claude on call for hard questions (/claude, asks before sending)")
        lines.append(" · ".join(have))
        lines.append("Ask about code, coursework or papers. /help lists commands.")
        lines.append(KEYS_HELP if getattr(s, "claude", None) else KEYS_HELP.replace(" · Ctrl+G ask Claude", ""))
        self.add_line("\n".join(lines), "welcome")
        prompt = self.prompt
        prompt.disabled = False
        prompt.placeholder = "Ask ALICE anything...  (/ for commands, Shift+Enter for a new line)"
        prompt.focus()
        self.refresh_status()

    # ----- input
    @on(TextArea.Changed, "#prompt")
    def show_command_hints(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        hint = self.query_one("#hint", Static)
        word = text.split(" ", 1)[0] if text.startswith("/") and "\n" not in text else ""
        matches = [f"{c}  {d}" for c, d in COMMANDS.items() if word and c.startswith(word)]
        hint.display = bool(matches)
        if hint.display:
            hint.update("\n".join(matches))

    def action_submit_prompt(self) -> None:
        prompt = self.prompt
        if not prompt.has_focus:
            return
        text = prompt.text.strip()
        if not text or not self.ready:
            return
        if self.busy:
            self.notify("ALICE is still answering. Press Esc to stop her first.", severity="warning")
            return
        prompt.remember(text)
        prompt.text = ""
        if text == "/clear":
            self.action_clear_screen()
            return
        if text.split()[0] == "/copy":
            self.copy_answer(code_only="code" in text.split()[1:])
            return
        cmd = self.session.command(text)
        if cmd is not None:
            if cmd.quit:
                self.action_quit()
                return
            if cmd.escalate:
                self.prepare_escalation(cmd.escalate_question)
                return
            self.add_line(cmd.message, "info")
            if cmd.ask:                                   # /read (or a dragged file): ask about it right away
                self.add_user(cmd.ask)
                self._begin_turn("reading the document")
                self.answer(cmd.ask)
                return
            self.refresh_status()
            return
        self.add_user(text)
        self._begin_turn("reading your question")
        self.answer(text)

    def _begin_turn(self, phase: str) -> None:
        self.busy = True
        self.cancel.clear()
        self.phase = phase
        self.turn_started = time.monotonic()
        self.refresh_status()

    def action_insert_prompt_newline(self) -> None:
        if self.prompt.has_focus:
            self.prompt.insert("\n")

    # ----- one turn (worker thread)
    @work(thread=True, exclusive=True, group="turn")
    def answer(self, text: str) -> None:
        ui = _TurnRenderer(self)
        try:
            if getattr(self.session, "background", None):
                ui.set_phase(f"finishing {self.session.background}")
            outcome = self.session.ask(text, on_event=ui.event, on_tool_start=ui.tool_start,
                                       on_tool=ui.tool_done, confirm=self.confirm_from_thread,
                                       on_library=ui.library, on_arxiv=ui.arxiv, on_program=ui.program,
                                       cancel=self.cancel)
            ui.finish(outcome)
        except Exception as e:                            # never leave the UI stuck in "busy"
            ui._call(self.add_line, f"error: {type(e).__name__}: {e}", "warning")
        finally:
            ui._call(self.turn_done)

    def turn_done(self) -> None:
        self.busy = False
        self.phase = ""
        self.refresh_status()

    def confirm_from_thread(self, tool: Tool, args: dict) -> bool:
        """Called by the agent (worker thread) before running code: show the dialog and wait."""
        answer: dict = {}
        done = threading.Event()
        code = str(args.get("code", json.dumps(args, indent=2)))
        language, stdin = str(args.get("language", "python")), str(args.get("stdin", "") or "")

        def show() -> None:
            self.push_screen(ConfirmRun(tool.name, code, language, stdin),
                             callback=lambda ok: (answer.update(ok=bool(ok)), done.set()))
        self.call_from_thread(show)
        while not done.wait(0.2):
            if self.closing or self.cancel.is_set():
                return False
        return answer.get("ok", False)

    # ----- escalation to Claude
    def action_ask_claude(self) -> None:
        if not self.ready or self.busy:
            return
        if not getattr(self.session, "claude", None):
            self.add_line(self.session.command("/claude").message, "info")
            return
        self.prepare_escalation(None)

    @work(thread=True, exclusive=True, group="turn")
    def prepare_escalation(self, question: str | None) -> None:
        from .claude import ClaudeError
        self.call_from_thread(self._begin_turn, "preparing the request for Claude")
        try:
            req = self.session.prepare_escalation(question)
        except Exception as e:                            # an unhandled worker error would close the app
            msg = str(e) if isinstance(e, ClaudeError) else f"{type(e).__name__}: {e}"
            self.call_from_thread(self.add_line, f"(claude: {msg})", "warning")
            self.call_from_thread(self.turn_done)
            return
        c = self.session.claude
        month = (f"This month so far: ${c.ledger.month_total():.2f} of ${c.settings.monthly_budget_usd:.2f} "
                 f"(model {c.settings.model}, effort {c.settings.effort}).")
        self.call_from_thread(self.turn_done)

        def decided(send: bool) -> None:
            if send:
                if question:
                    self.add_user(question)
                self._begin_turn("waiting for Claude")
                self.escalate(req)
            else:
                self.add_line("(not sent to Claude)", "info")
        self.call_from_thread(lambda: self.push_screen(ConfirmEscalation(req.summary(), month, req.preview()),
                                                       callback=decided))

    @work(thread=True, exclusive=True, group="turn")
    def escalate(self, req) -> None:
        ui = _TurnRenderer(self, kind="claude", title=f"Claude ({self.session.claude.settings.model}) for ALICE")
        try:
            out = self.session.escalate(req, on_text=ui.claude_text, on_status=ui.claude_status, cancel=self.cancel)
            ui._flush()
            if out.error:
                ui._call(self.add_line, f"(claude: {out.error})", "warning")
            for n in out.notices:
                ui._call(self.add_line, n.text, n.kind)
            if out.reply and not out.reply.interrupted and out.reply.text:
                self.last_answer = out.reply.text
        except Exception as e:
            ui._call(self.add_line, f"error: {type(e).__name__}: {e}", "warning")
        finally:
            ui._call(self.turn_done)

    # ----- actions
    def action_stop(self) -> None:
        if self.busy:
            self.cancel.set()
            self.notify("Stopping...")

    def action_copy_answer(self) -> None:
        self.copy_answer()

    def copy_answer(self, code_only: bool = False) -> None:
        text = self.last_answer
        if code_only:
            blocks = _CODE_BLOCK.findall(text)
            text = blocks[-1] if blocks else ""
        if not text:
            self.notify("Nothing to copy yet." if not code_only else "The last answer has no code block.",
                        severity="warning")
            return
        self.copy_to_clipboard(text)
        self.notify("Copied the last " + ("code block" if code_only else "answer") + " to the clipboard.")

    def action_scroll_log(self, direction: int) -> None:
        (self.log_view.scroll_page_up if direction < 0 else self.log_view.scroll_page_down)(animate=False)

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

    def __init__(self, app: AliceApp, kind: str = "alice", title: str = "ALICE"):
        self.app = app
        self.kind, self.title = kind, title
        self.md: Markdown | None = None
        self.text = ""
        self.filter = StreamFilter()
        self.last_update = 0.0
        self.thinking: Static | None = None
        self.reasoning_chars = 0
        self.box: Static | None = None                    # the running program's box
        self.box_parts: list[tuple[str, str]] = []
        self.box_title = ""
        self.shown_program = False                        # the current tool call has a box (it shows the result)

    def _call(self, fn, *args):
        if self.app.closing:
            return None
        try:
            return self.app.call_from_thread(fn, *args)
        except RuntimeError:                              # the app quit while ALICE was answering
            return None

    def set_phase(self, phase: str) -> None:
        self.app.phase = phase                            # read by the status timer; a plain attribute

    def library(self, places: list[str]) -> None:
        self.set_phase(f"reading {len(places)} passage{'s' * (len(places) != 1)} from your library")
        self._call(self.app.add_line, "library: " + "; ".join(places), "sources")

    def arxiv(self, query: str) -> None:
        self.set_phase("searching arXiv")
        self._call(self.app.add_line, f"arXiv: searching for '{query}' (sent to arxiv.org)", "sources")

    def event(self, ev: Event) -> None:
        if ev.kind == "reasoning":
            self.set_phase("thinking")
            self.reasoning_chars += len(ev.text)
            dots = "." * min(40, 1 + self.reasoning_chars // 300)
            if self.thinking is None:
                self.thinking = self._call(self.app.add_line, "thinking" + dots, "thinking")
            elif time.monotonic() - self.last_update > 0.3:
                self.last_update = time.monotonic()
                self._call(self.thinking.update, "thinking" + dots)
        elif ev.kind == "content":
            self.set_phase("writing")
            self._show(self.filter.feed(ev.text))
        elif ev.kind in ("tool_calls", "finish"):
            self._flush()

    def _show(self, visible: str) -> None:
        if not visible:
            return
        if self.md is None:
            self.md = self._call(self.app.new_reply, self.kind, self.title)
        self.text += visible
        if time.monotonic() - self.last_update > UPDATE_EVERY_S:
            self.last_update = time.monotonic()
            self._call(self.app.set_reply, self.md, self.text)

    def _flush(self) -> None:
        rest = self.filter.flush()
        if rest and self.md is None:
            self.md = self._call(self.app.new_reply, self.kind, self.title)
        self.text += rest
        if self.md is not None:
            self._call(self.app.set_reply, self.md, self.text)

    def tool_start(self, name: str, raw_args: str) -> None:
        self._flush()
        self.set_phase(f"using {name}")
        self.md, self.text, self.filter = None, "", StreamFilter()      # next round gets a new bubble
        self.shown_program = False
        self._call(self.app.add_line, f"▸ {name}: {tool_summary(name, raw_args)}", "tool")

    def tool_done(self, tr: ToolResult) -> None:
        mark = "declined" if tr.declined else ("ok" if tr.ok else "failed")
        self.set_phase("reading the result")
        if not self.shown_program:
            self._call(self.app.add_line, f"  ◂ {mark}: {_short(tr.output, 200)}", "tool")

    def program(self, e: dict) -> None:
        """Live events of a program ALICE runs (coderun batches the output, ~10 updates a second)."""
        if e["event"] == "start":
            self.shown_program, self.box_parts, self.box_title = True, [], e["title"]
            self.set_phase(f"running the {e['title']} program")
            self.box = self._call(self.app.add_program, e["title"])
        elif self.box is None:
            return
        elif e["event"] == "phase":
            self.set_phase(f"{e['text']} the {self.box_title} program")
            self._call(self.app.update_program, self.box, list(self.box_parts), f"{e['text']}...")
        elif e["event"] == "out":
            self.box_parts.append((e["stream"], e["text"]))
            self._call(self.app.update_program, self.box, list(self.box_parts), "running...")
        elif e["event"] == "end":
            self._call(self.app.update_program, self.box, list(self.box_parts), e["status"], e["ok"])
            self.box = None

    def claude_status(self, what: str) -> None:
        self.set_phase("Claude is thinking" if what == "thinking" else what)

    def claude_text(self, text: str) -> None:
        self.set_phase("Claude is writing")
        self._show(text)

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
            self.app.last_answer = result.reply


def main(argv: list[str] | None = None) -> int:
    args = chat_arguments("alice tui").parse_args(argv)
    session = session_from_args(args)
    app = AliceApp(session)
    try:
        app.run()
        if session.convo is not None:
            save_memory_with_messages(session)
    finally:
        session.stop()
    if session.transcript:
        print(f"(transcript saved to {session.transcript.path})")
    return 0
