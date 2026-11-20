"""`alice gui`: ALICE in her own desktop window.

    alice gui [--browser] [same options as the chat: --model, --think, --workspace, --offline, --no-memory, --claude]

Starts the GUI back end (server.py) on a free port on 127.0.0.1 with a random token, then shows it in a
desktop window (pywebview, which uses Windows' built-in WebView2). --browser opens your default browser
instead. Closing the window saves memory notes (a few seconds) and stops the model.
"""
from __future__ import annotations

import secrets
import socket
import sys
import threading
import time
import webbrowser

from ..cli import chat_arguments, save_memory_with_messages, session_from_args


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    ap = chat_arguments("alice gui")
    ap.add_argument("--browser", action="store_true", help="open in your default browser instead of a window")
    args = ap.parse_args(argv)
    try:
        import uvicorn
        from .server import Controller, create_app
    except ImportError as e:
        print(f"the GUI needs a few extra packages ({e.name}): pip install -e \".[gui]\"")
        return 1
    session = session_from_args(args)
    controller = Controller(session)
    token = secrets.token_urlsafe(24)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(controller, token), host="127.0.0.1", port=port,
                                           log_level="warning", ws="websockets"))
    threading.Thread(target=server.run, name="alice-gui-server", daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}/?token={token}"
    try:
        if args.browser:
            webbrowser.open(url)
            print(f"ALICE is open in your browser ({url.split('?')[0]}). Press Ctrl+C here to quit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            try:
                import webview
            except ImportError:
                print("pywebview is not installed (pip install -e \".[gui]\"); opening your browser instead.")
                webbrowser.open(url)
                while True:
                    time.sleep(1)
            webview.create_window("ALICE", url, width=1180, height=820, min_size=(720, 520),
                                  background_color="#0f1115")
            webview.start()                                  # returns when the window is closed
    finally:
        controller.cancel.set()
        if session.convo is not None:
            save_memory_with_messages(session)
        server.should_exit = True
        session.stop()
        if session.transcript:
            print(f"(transcript saved to {session.transcript.path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
