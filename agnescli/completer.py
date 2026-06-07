"""Slash command autocompletion for Agnescli."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style

COMMANDS = [
    ("/plan", "task", "cmd.plan"),
    ("/image", "prompt", "cmd.image"),
    ("/video", "prompt", "cmd.video"),
    ("/thinking", "", "cmd.thinking"),
    ("/auto", "", "cmd.auto"),
    ("/compact", "", "cmd.compact"),
    ("/new", "", "cmd.new"),
    ("/resume", "[id]", "cmd.resume"),
    ("/sessions", "", "cmd.sessions"),
    ("/config", "", "cmd.config"),
    ("/status", "", "cmd.status"),
    ("/save", "[file]", "cmd.save"),
    ("/clear", "", "cmd.clear"),
    ("/lang", "[code]", "cmd.lang"),
    ("/help", "", "cmd.help"),
    ("/exit", "", "cmd.exit"),
]

STYLE = Style.from_dict(
    {
        "prompt": "bold green",
        "completion-menu": "bg:#2d2d2d #cccccc",
        "completion-menu.completion": "bg:#2d2d2d #cccccc",
        "completion-menu.completion.current": "bg:#005f87 #ffffff",
        "completion-menu.meta.completion": "bg:#3d3d3d #888888",
        "completion-menu.meta.completion.current": "bg:#005f87 #aaaaaa",
    }
)


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        from .i18n import t

        text = document.text_before_cursor.lstrip()

        if text and not text.startswith("/"):
            return

        word = text.split()[0] if text else "/"

        for cmd, args, desc_key in COMMANDS:
            if cmd.startswith(word):
                display = f"{cmd}  {args}" if args else cmd
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=display,
                    display_meta=t(desc_key),
                )


_session: PromptSession | None = None


def create_session() -> PromptSession:
    global _session
    _session = PromptSession(style=STYLE)
    return _session


def _get_session() -> PromptSession:
    global _session
    if _session is None:
        _session = PromptSession(style=STYLE)
    return _session


def prompt_slash(session: PromptSession) -> str:
    return session.prompt(
        "> ",
        completer=SlashCompleter(),
        complete_while_typing=True,
    )


def prompt_confirm(message: str = "Run? [Y/n] ") -> bool:
    """Prompt for y/n confirmation using prompt_toolkit."""
    session = _get_session()
    try:
        answer = session.prompt(f"[dim]{message}[/]").strip().lower()
        return answer in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
