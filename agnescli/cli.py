"""Agnescli - Codex-style Agent CLI for Agnes AI."""

from __future__ import annotations

from typing import Any

import click

from . import __version__
from .config import resolve_api_key, save_api_key
from .ui import console


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="agnescli")
@click.option("--api-key", envvar="AGNES_API_KEY", default=None, help="Agnes AI API key.")
@click.option("--thinking", is_flag=True, default=False, help="Enable thinking mode.")
@click.option("-y", "--auto-confirm", is_flag=True, default=False, help="Skip tool execution confirmation.")
@click.option("-p", "--prompt", default=None, help="Non-interactive mode: run prompt and exit.")
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text", help="Output format for -p mode.")
@click.option(
    "-c", "--continue-session", "continue_session", is_flag=True, default=False, help="Continue the last session."
)
@click.option("--lang", default=None, help="Language (en, zh). Auto-detected if not set.")
@click.pass_context
def main(
    ctx: click.Context,
    api_key: str | None,
    thinking: bool,
    auto_confirm: bool,
    prompt: str | None,
    output_format: str,
    continue_session: bool,
    lang: str | None,
) -> None:
    """Agnescli - Autonomous Agent CLI for Agnes AI."""
    from .i18n import init as i18n_init

    i18n_init(lang=lang)

    ctx.ensure_object(dict)
    ctx.obj["api_key_raw"] = api_key
    ctx.obj["thinking"] = thinking
    ctx.obj["auto_confirm"] = auto_confirm
    if lang:
        ctx.obj["lang"] = lang

    # If no subcommand given, launch the agent directly
    if ctx.invoked_subcommand is None:
        from .agent import run_agent
        from .client import AgnesClient

        key = resolve_api_key(api_key)
        client = AgnesClient(key)

        if prompt is not None:
            _run_prompt(client, prompt, thinking=thinking, output_format=output_format)
        else:
            run_agent(
                client,
                thinking=thinking,
                auto_confirm=auto_confirm,
                continue_session=continue_session,
            )


@main.command()
@click.argument("key")
def setup(key: str) -> None:
    """Save API key to local config (~/.agnescli/config.json)."""
    from .i18n import t

    save_api_key(key)
    console.print(f"[green]{t('cli.api_key_saved')}[/]")


@main.command()
def models() -> None:
    """List available Agnes AI models."""
    from rich.table import Table

    from .i18n import t

    MODELS = [
        ("agnes-2.0-flash", "Chat / Language", "Chat, coding, reasoning, agent workflows", "256K"),
        ("agnes-image-2.1-flash", "Image Generation", "Text-to-image and image-to-image", "-"),
        ("agnes-video-v2.0", "Video Generation", "Text-to-video, image-to-video, keyframes", "441 frames"),
    ]

    table = Table(title=t("cli.models_title"), show_lines=True)
    table.add_column(t("cli.model_name"), style="cyan bold")
    table.add_column(t("cli.model_type"), style="green")
    table.add_column(t("cli.model_desc"))
    table.add_column(t("cli.model_ctx"), justify="right")

    for name, typ, desc, ctx_val in MODELS:
        table.add_row(name, typ, desc, ctx_val)

    console.print(table)
    console.print()


@main.command()
@click.argument("session_id", required=False)
def resume(session_id: str | None) -> None:
    """Resume a previous session. Lists sessions if no ID given."""
    from .i18n import t
    from .session import list_sessions

    if session_id is None:
        sessions = list_sessions(limit=10)
        if not sessions:
            console.print(f"[dim]{t('cli.no_sessions')}[/]")
            return

        from datetime import datetime

        from rich.table import Table

        table = Table(title=t("cli.recent_sessions"), show_lines=False)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("ID", style="cyan")
        table.add_column("Messages", justify="right")
        table.add_column(t("cli.session_preview"))
        table.add_column(t("cli.session_time"))

        for i, s in enumerate(sessions, 1):
            ts = datetime.fromtimestamp(s["modified"]).strftime("%m-%d %H:%M")
            table.add_row(str(i), s["id"], str(s["messages"]), s["preview"][:60], ts)

        console.print(table)
        console.print(f"\n[dim]{t('cli.resume_usage')}[/]")
        return

    # Resume specific session
    from .agent import run_agent
    from .client import AgnesClient

    key = resolve_api_key()
    client = AgnesClient(key)
    run_agent(client, resume_session=session_id)


def _run_prompt(client: Any, prompt: str, *, thinking: bool, output_format: str) -> None:
    """Non-interactive mode: send prompt, print result, exit."""
    import json as json_mod

    from .agent import _agent_loop
    from .config import get_config
    from .session import new_session_id

    cfg = get_config()
    system = (
        "You are an autonomous AI agent running inside Agnescli. Execute the user's request and provide a clear result."
    )
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": prompt})

    session_id = new_session_id()

    try:
        _agent_loop(
            client,
            messages,
            thinking=thinking,
            auto_confirm=True,
            max_iterations=20,
            session_id=session_id,
            model=cfg.get("model", "agnes-2.0-flash"),
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=cfg.get("temperature", 0.7),
        )
    except KeyboardInterrupt:
        pass

    # Extract final assistant message for -p output
    final = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            final = m["content"]
            break

    if output_format == "json":
        result: dict[str, Any] = {
            "session_id": session_id,
            "prompt": prompt,
            "response": final,
            "messages": len(messages),
        }
        print(json_mod.dumps(result, ensure_ascii=False, indent=2))
    else:
        if final:
            print(final)


if __name__ == "__main__":
    main()
