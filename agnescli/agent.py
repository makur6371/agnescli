"""Unified autonomous agent - handles tasks, image, video in one session."""

from __future__ import annotations

import json
import re
import time
import webbrowser
from datetime import datetime
from typing import Any

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .client import AgnesAPIError, AgnesClient
from .completer import _get_session as _get_pt_session
from .completer import create_session, prompt_confirm, prompt_slash
from .config import get_config
from .i18n import available_langs, get_lang, lang_display_name, set_lang, t
from .session import (
    get_last_session_id,
    list_sessions,
    load_session,
    new_session_id,
    save_message,
)
from .tools import TOOLS, execute_tool
from .ui import console

SYSTEM_PROMPT = (
    "You are an autonomous AI agent running inside Agnescli. "
    "You can execute shell commands, read/write files, run Python code, "
    "search for files and content, generate images, and generate videos "
    "to help the user.\n\n"
    "When given a task:\n"
    "1. Break it into steps\n"
    "2. Execute tools to accomplish each step\n"
    "3. Verify results before moving on\n"
    "4. Report the final outcome\n\n"
    "For image/video requests, use the generate_image / generate_video tools directly.\n"
    "Be concise. Focus on getting the task done."
)

PLANNER_PROMPT = (
    "You are a task planner. Given a user request, analyze it and produce a clear, "
    "executable plan with numbered steps.\n\n"
    "Rules:\n"
    "- Each step must be a concrete, actionable item\n"
    "- Order steps logically - dependencies first\n"
    "- For simple tasks (single action, quick question, image/video generation), "
    "respond with: SKIP_PLAN\n"
    "- For multi-step tasks, output ONLY a numbered list, nothing else\n\n"
    "Format:\n"
    "1. First step description\n"
    "2. Second step description\n"
    "3. Third step description\n\n"
    "Keep steps concise (one line each). Max 10 steps."
)

EXECUTOR_PROMPT_TEMPLATE = (
    "You are executing a plan. Here is the plan:\n\n{plan}\n\n"
    'Focus ONLY on step {current_step}: "{current_desc}"\n'
    "Execute this step using your tools. When done, report the result briefly.\n"
    "Do not skip ahead to other steps."
)

COMPACT_PROMPT = (
    "Summarize the following conversation history concisely, preserving key facts, "
    "decisions, and context. Keep tool results that may be referenced later. "
    "Output only the summary, no preamble."
)

MAX_MESSAGES_BEFORE_COMPACT = 40
MAX_TOKENS_BEFORE_COMPACT = 80000


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content)
        for tc in m.get("tool_calls", []):
            total += len(json.dumps(tc))
    return total // 4


def run_agent(
    client: AgnesClient,
    *,
    system: str = SYSTEM_PROMPT,
    thinking: bool = False,
    auto_confirm: bool = False,
    max_iterations: int = 50,
    continue_session: bool = False,
    resume_session: str | None = None,
) -> None:
    cfg = get_config()

    # Initialize i18n
    from .i18n import init as i18n_init

    i18n_init(lang=cfg.get("lang"))

    # Resume or create session
    if resume_session:
        loaded = load_session(resume_session)
        if not loaded:
            console.print(f"[red]{t('session.not_found', id=resume_session)}[/]")
            return
        messages = loaded
        session_id = resume_session
        console.print(f"[dim]{t('session.resumed', id=session_id, count=len(messages))}[/]")
    elif continue_session:
        last_id = get_last_session_id()
        if last_id:
            loaded = load_session(last_id)
            if loaded:
                messages = loaded
                session_id = last_id
                console.print(f"[dim]{t('session.continuing', id=session_id, count=len(messages))}[/]")
            else:
                session_id = new_session_id()
                messages = [{"role": "system", "content": system}]
        else:
            session_id = new_session_id()
            messages = [{"role": "system", "content": system}]
    else:
        session_id = new_session_id()
        messages = [{"role": "system", "content": system}]

    state = {
        "thinking": thinking,
        "auto_confirm": auto_confirm,
        "max_iterations": max_iterations,
        "session_id": session_id,
        "model": cfg.get("model", "agnes-2.0-flash"),
        "max_tokens": cfg.get("max_tokens", 4096),
        "temperature": cfg.get("temperature", 0.7),
        "total_tokens": 0,
    }

    # Show full banner on first run, compact status on subsequent runs
    sessions = list_sessions(limit=1)
    if sessions:
        _show_status_line(state)
    else:
        _show_banner(state)

    pt_session = create_session()

    while True:
        try:
            user_input = prompt_slash(pt_session).strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n{t('agent.bye')}")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()
        action = cmd[0]

        # ── Slash commands ─────────────────────────────────────────
        if action == "/exit":
            console.print(t("agent.bye"))
            break

        if action == "/clear":
            messages = [{"role": "system", "content": system}]
            session_id = new_session_id()
            state["session_id"] = session_id
            console.print(f"[dim]{t('session.cleared')}[/]")
            continue

        if action == "/new":
            messages = [{"role": "system", "content": system}]
            session_id = new_session_id()
            state["session_id"] = session_id
            console.print(f"[dim]{t('session.new', id=session_id)}[/]")
            continue

        if action == "/save":
            path = user_input[5:].strip() or "history.md"
            _save_history(messages, path)
            continue

        if action == "/sessions":
            _show_sessions()
            continue

        if action == "/resume":
            sid = user_input[7:].strip()
            if not sid:
                sessions = list_sessions(limit=10)
                if not sessions:
                    console.print(f"[dim]{t('session.no_saved')}[/]")
                    continue
                _show_sessions()
                try:
                    pick = _get_pt_session().prompt(f"[dim]{t('session.prompt_id')}[/]").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if not pick:
                    continue
                # Support picking by number
                if pick.isdigit() and 1 <= int(pick) <= len(sessions):
                    sid = sessions[int(pick) - 1]["id"]
                else:
                    sid = pick
            loaded = load_session(sid)
            if not loaded:
                console.print(f"[red]{t('session.not_found', id=sid)}[/]")
                continue
            messages = loaded
            session_id = sid
            state["session_id"] = session_id
            console.print(f"[dim]{t('session.resumed', id=sid, count=len(messages))}[/]")
            continue

        if action == "/compact":
            _compact(client, messages, system)
            continue

        if action == "/config":
            _show_config(state, cfg)
            continue

        if action == "/thinking":
            state["thinking"] = not state["thinking"]
            status = t("config.on") if state["thinking"] else t("config.off")
            console.print(f"[cyan]{t('cmd.thinking', status=status)}[/]")
            continue

        if action == "/auto":
            state["auto_confirm"] = not state["auto_confirm"]
            status = t("config.on") if state["auto_confirm"] else t("config.off")
            console.print(f"[cyan]{t('cmd.auto', status=status)}[/]")
            continue

        if action == "/help":
            _show_banner(state)
            continue

        if action == "/status":
            _show_status(state, messages, session_id)
            continue

        if action == "/lang":
            lang_arg = cmd[1] if len(cmd) > 1 else ""
            if not lang_arg:
                # Show available languages
                langs = available_langs()
                current = get_lang()
                parts = []
                for lang_code in langs:
                    name = lang_display_name(lang_code)
                    marker = " *" if lang_code == current else ""
                    parts.append(f"  [cyan]{lang_code}[/] ({name}){marker}")
                console.print("\n".join(parts))
                continue
            if set_lang(lang_arg):
                cfg["lang"] = lang_arg
                from .config import save_config_value

                save_config_value("lang", lang_arg)
                console.print(f"[green]{t('lang.switched', lang=lang_display_name(lang_arg))}[/]")
                _show_banner(state)
            else:
                console.print(f"[red]Unknown language: {lang_arg}[/]")
            continue

        if action == "/image":
            prompt = user_input[6:].strip()
            if not prompt:
                console.print(f"[yellow]{t('image.usage')}[/]")
                continue
            messages.append({"role": "user", "content": user_input})
            save_message(session_id, {"role": "user", "content": user_input})
            result = _handle_image(client, prompt)
            messages.append({"role": "assistant", "content": result})
            save_message(session_id, {"role": "assistant", "content": result})
            continue

        if action == "/video":
            prompt = user_input[7:].strip()
            if not prompt:
                console.print(f"[yellow]{t('video.usage')}[/]")
                continue
            messages.append({"role": "user", "content": user_input})
            save_message(session_id, {"role": "user", "content": user_input})
            result = _handle_video(client, prompt)
            messages.append({"role": "assistant", "content": result})
            save_message(session_id, {"role": "assistant", "content": result})
            continue

        if action == "/plan":
            task = user_input[5:].strip()
            if not task:
                console.print("[yellow]Usage: /plan <task description>[/]")
                continue
            messages.append({"role": "user", "content": task})
            save_message(session_id, {"role": "user", "content": task})
            try:
                _plan_and_execute(
                    client,
                    messages,
                    system,
                    thinking=state["thinking"],
                    auto_confirm=state["auto_confirm"],
                    max_iterations=state["max_iterations"],
                    session_id=session_id,
                    model=state["model"],
                    max_tokens=state["max_tokens"],
                    temperature=state["temperature"],
                )
            except KeyboardInterrupt:
                console.print(f"\n[dim]{t('agent.interrupted')}[/]")
            except AgnesAPIError as exc:
                console.print(f"\n[red]{t('agent.api_error', code=exc.status_code, msg=exc.message)}[/]")
            except Exception as exc:
                console.print(f"\n[red]{t('agent.error', msg=exc)}[/]")
            continue

        # ── Agent task ─────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})
        save_message(session_id, {"role": "user", "content": user_input})

        # Auto-compact check
        est_tokens = _estimate_tokens(messages)
        if est_tokens > MAX_TOKENS_BEFORE_COMPACT:
            console.print(f"[yellow]{t('compact.suggest_tokens', n=est_tokens)}[/]")
        elif len(messages) > MAX_MESSAGES_BEFORE_COMPACT:
            console.print(f"[yellow]{t('compact.suggest_msgs', n=len(messages))}[/]")

        try:
            tokens_used = _agent_loop(
                client,
                messages,
                thinking=state["thinking"],
                auto_confirm=state["auto_confirm"],
                max_iterations=state["max_iterations"],
                session_id=session_id,
                model=state["model"],
                max_tokens=state["max_tokens"],
                temperature=state["temperature"],
            )
            state["total_tokens"] += tokens_used
        except KeyboardInterrupt:
            console.print(f"\n[dim]{t('agent.interrupted')}[/]")
        except AgnesAPIError as exc:
            console.print(f"\n[red]{t('agent.api_error', code=exc.status_code, msg=exc.message)}[/]")
            messages.pop()
        except Exception as exc:
            console.print(f"\n[red]{t('agent.error', msg=exc)}[/]")
            messages.pop()


def _show_status_line(state: dict) -> None:
    th = "T" if state["thinking"] else ""
    au = "A" if state["auto_confirm"] else ""
    flags = " ".join(filter(None, [th, au]))
    flag_str = f" [{flags}]" if flags else ""
    console.print(f"[bold cyan]{t('agent.title')}[/]{flag_str} - {t('agent.type_or_help')}")


def _show_banner(state: dict) -> None:
    on = t("config.on")
    off = t("config.off")
    th = "[green]" + on + "[/]" if state["thinking"] else "[dim]" + off + "[/]"
    au = "[green]" + on + "[/]" if state["auto_confirm"] else "[dim]" + off + "[/]"
    console.print(
        Panel(
            f"[bold cyan]{t('agent.title')}[/] - {t('agent.subtitle')}\n"
            f"{t('agent.type_task')}\n\n"
            f"[bold]/plan[/] task       - {t('cmd.plan')}\n"
            f"[bold]/image[/] prompt     - {t('cmd.image')}\n"
            f"[bold]/video[/] prompt     - {t('cmd.video')}\n"
            f"[bold]/thinking[/]        - {t('cmd.thinking', status=th)}\n"
            f"[bold]/auto[/]            - {t('cmd.auto', status=au)}\n"
            f"[bold]/compact[/]         - {t('cmd.compact')}\n"
            f"[bold]/new[/]             - {t('cmd.new')}\n"
            f"[bold]/resume[/] [id]     - {t('cmd.resume')}\n"
            f"[bold]/sessions[/]        - {t('cmd.sessions')}\n"
            f"[bold]/config[/]          - {t('cmd.config')}\n"
            f"[bold]/status[/]          - {t('cmd.status')}\n"
            f"[bold]/clear[/]           - {t('cmd.clear')}\n"
            f"[bold]/save[/] [file]     - {t('cmd.save')}\n"
            f"[bold]/lang[/] [code]     - {t('cmd.lang')}\n"
            f"[bold]/help[/]            - {t('cmd.help')}\n"
            f"[bold]/exit[/]            - {t('cmd.exit')}",
            border_style="cyan",
        )
    )


# ── Plan & Execute ─────────────────────────────────────────────────


def _plan_task(client: AgnesClient, task: str) -> list[str] | None:
    """Ask the model to produce an execution plan. Returns list of steps or None."""
    plan_messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": task},
    ]

    with console.status(f"[bold cyan]{t('plan.planning')}", spinner="dots"):
        resp = client.chat(plan_messages, stream=False, max_tokens=1024)
    content = resp["choices"][0]["message"]["content"].strip()

    if "SKIP_PLAN" in content:
        return None

    # Parse numbered steps: "1. Do something" or "1) Do something"
    steps = []
    for line in content.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m:
            steps.append(m.group(1).strip())

    return steps if steps else None


def _display_plan(steps: list[str]) -> None:
    """Display plan as a numbered list in a panel."""
    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"  [cyan]{i}.[/] {step}")
    plan_text = "\n".join(lines)
    console.print(Panel(plan_text, title=f"[bold]{t('plan.title')}", border_style="cyan", expand=False))


def _confirm_plan(steps: list[str]) -> str:
    """Ask user to approve the plan. Returns 'y', 'n', or 'edit'."""
    try:
        session = _get_pt_session()
        answer = session.prompt(f"\n[dim]{t('plan.execute_prompt')}[/]").strip().lower()
        if answer in ("", "y", "yes"):
            return "y"
        if answer in ("n", "no"):
            return "n"
        if answer in ("e", "edit"):
            return "edit"
        return "y"
    except (EOFError, KeyboardInterrupt):
        return "n"


def _edit_plan(steps: list[str]) -> list[str] | None:
    """Let user edit the plan interactively."""
    console.print(f"[dim]{t('plan.edit_prompt')}[/]")
    session = _get_pt_session()
    new_steps = []
    while True:
        try:
            line = session.prompt(f"  [cyan]{len(new_steps) + 1}.[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not line:
            break
        new_steps.append(line)
    return new_steps if new_steps else None


def _plan_and_execute(
    client: AgnesClient,
    messages: list[dict[str, Any]],
    system: str,
    *,
    thinking: bool,
    auto_confirm: bool,
    max_iterations: int,
    session_id: str,
    model: str = "agnes-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> None:
    """Plan a task, get user approval, then execute step by step."""
    task = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            task = m["content"]
            break
    if not task:
        return

    steps = _plan_task(client, task)

    if steps is None:
        console.print(f"[dim]{t('plan.simple_task')}[/]")
        _agent_loop(
            client,
            messages,
            thinking=thinking,
            auto_confirm=auto_confirm,
            max_iterations=max_iterations,
            session_id=session_id,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return

    _display_plan(steps)

    action = _confirm_plan(steps)
    if action == "n":
        console.print(f"[dim]{t('plan.cancelled')}[/]")
        messages.append({"role": "assistant", "content": "Plan cancelled by user."})
        save_message(session_id, {"role": "assistant", "content": "Plan cancelled by user."})
        return
    if action == "edit":
        edited = _edit_plan(steps)
        if edited is None:
            console.print(f"[dim]{t('plan.cancelled')}[/]")
            return
        steps = edited
        _display_plan(steps)

    # Execute plan step by step
    console.print()
    _execute_plan(
        client,
        messages,
        system,
        steps,
        thinking=thinking,
        auto_confirm=auto_confirm,
        max_iterations=max_iterations,
        session_id=session_id,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _execute_plan(
    client: AgnesClient,
    messages: list[dict[str, Any]],
    system: str,
    steps: list[str],
    *,
    thinking: bool,
    auto_confirm: bool,
    max_iterations: int,
    session_id: str,
    model: str = "agnes-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> None:
    """Execute a plan step by step, showing progress."""
    total = len(steps)
    completed = 0
    results: list[str] = []

    plan_summary = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

    for i, step in enumerate(steps):
        _show_plan_progress(i, steps, results)

        step_system = (
            system
            + "\n\n"
            + EXECUTOR_PROMPT_TEMPLATE.format(
                plan=plan_summary,
                current_step=i + 1,
                current_desc=step,
            )
        )
        step_messages = [
            {"role": "system", "content": step_system},
        ]
        for m in messages[-6:]:
            if m.get("role") != "system":
                step_messages.append(m)
        step_messages.append({"role": "user", "content": f"Execute step {i + 1}: {step}"})

        try:
            _agent_loop(
                client,
                step_messages,
                thinking=thinking,
                auto_confirm=auto_confirm,
                max_iterations=max(10, max_iterations // total),
                session_id=session_id,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except KeyboardInterrupt:
            console.print(f"\n[yellow]{t('plan.stopped_at', n=i + 1, total=total)}[/]")
            break
        except Exception as exc:
            console.print(f"\n[red]{t('plan.step_failed', n=i + 1, error=exc)}[/]")
            results.append(f"FAILED: {exc}")
            continue

        step_result = "Done"
        for m in reversed(step_messages):
            if m.get("role") == "assistant" and m.get("content"):
                step_result = m["content"][:200]
                break
        results.append(step_result)
        completed += 1

        existing_keys = {json.dumps(m, sort_keys=True, ensure_ascii=False, default=str) for m in messages}
        for m in step_messages[1:]:
            key = json.dumps(m, sort_keys=True, ensure_ascii=False, default=str)
            if key not in existing_keys:
                messages.append(m)
                save_message(session_id, m)
                existing_keys.add(key)

    # Final summary
    console.print()
    _show_plan_progress(total, steps, results)
    status = (
        f"[green]{t('plan.complete')}[/]"
        if completed == total
        else f"[yellow]{t('plan.completed_n', n=completed, total=total)}[/]"
    )
    console.print(
        Panel(
            f"{status}\n{t('plan.steps_executed', n=completed, total=total)}",
            title=f"[bold]{t('plan.result_title')}",
            border_style="green" if completed == total else "yellow",
            expand=False,
        )
    )


def _show_plan_progress(current: int, steps: list[str], results: list[str]) -> None:
    """Show plan with progress indicators."""
    lines = []
    for i, step in enumerate(steps):
        if i < current:
            icon = "[green][OK][/]"
            result_preview = ""
            if i < len(results):
                r = results[i]
                if r.startswith("FAILED"):
                    icon = "[red][X][/]"
                    result_preview = f" [red]({r})[/]"
                else:
                    result_preview = f" [dim]({r[:60]}{'...' if len(r) > 60 else ''})[/]"
            lines.append(f"  {icon} {i + 1}. {step}{result_preview}")
        elif i == current:
            lines.append(f"  [cyan bold]>>[/] {i + 1}. [bold]{step}[/]")
        else:
            lines.append(f"  [dim]  {i + 1}. {step}[/]")

    console.print(Panel("\n".join(lines), title=f"[bold]{t('plan.title')}", border_style="dim", expand=False))


# ── Agent loop ──────────────────────────────────────────────────────


def _agent_loop(
    client: AgnesClient,
    messages: list[dict[str, Any]],
    *,
    thinking: bool,
    auto_confirm: bool,
    max_iterations: int,
    session_id: str,
    model: str = "agnes-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> int:
    total_tokens = 0

    for _ in range(max_iterations):
        response = client.chat(
            messages,
            stream=True,
            thinking=thinking,
            tools=TOOLS,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}

        console.print()
        with Live(console=console, refresh_per_second=12) as live:
            for chunk in response:
                choices = chunk.get("choices", [])
                if not choices:
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    continue
                delta = choices[0].get("delta", {})

                content = delta.get("content") or ""
                if content:
                    full_content += content
                    live.update(Markdown(full_content))

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if "id" in tc:
                        tool_calls[idx]["id"] = tc["id"]
                    if "function" in tc:
                        if "name" in tc["function"]:
                            tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

        # Show token usage
        if usage:
            prompt_t = usage.get("prompt_tokens", 0)
            comp_t = usage.get("completion_tokens", 0)
            total_tokens += prompt_t + comp_t
            console.print(f"[dim]  {t('agent.tokens', prompt=prompt_t, completion=comp_t, total=total_tokens)}[/]")

        if not tool_calls:
            if full_content:
                messages.append({"role": "assistant", "content": full_content})
                save_message(session_id, {"role": "assistant", "content": full_content})
            return total_tokens

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_content or None, "tool_calls": tool_calls}
        messages.append(assistant_msg)
        save_message(session_id, assistant_msg)

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args_str = tc["function"]["arguments"]
            tc_id = tc["id"]

            try:
                fn_args = json.loads(fn_args_str) if fn_args_str else {}
            except json.JSONDecodeError:
                fn_args = {}
                console.print(f"[yellow]{t('agent.warning_malformed_args')}[/]")

            _show_tool_call(fn_name, fn_args)

            if not auto_confirm:
                if not _confirm():
                    result = t("agent.tool_declined")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                    save_message(session_id, {"role": "tool", "tool_call_id": tc_id, "content": result})
                    console.print(f"[dim]{t('agent.skipped')}[/]")
                    continue

            # Execute with retry on failure
            result = None
            for retry in range(2):
                with console.status(f"[bold cyan]{t('agent.running', name=fn_name)}", spinner="dots"):
                    result = execute_tool(fn_name, fn_args, client=client)
                if not result.startswith("Error:") or retry == 1:
                    break
                console.print(f"[yellow]  {t('agent.tool_failed_retry')}[/]")

            _show_result(fn_name, result)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            save_message(session_id, {"role": "tool", "tool_call_id": tc_id, "content": result})

    console.print(f"\n[yellow]{t('agent.reached_max', n=max_iterations)}[/]")
    return total_tokens


# ── Context compression ────────────────────────────────────────────


def _compact(client: AgnesClient, messages: list[dict[str, Any]], system: str) -> None:
    if len(messages) <= 3:
        console.print(f"[dim]{t('compact.nothing')}[/]")
        return

    before = len(messages)

    recent = messages[-6:]
    old = messages[1:-6]

    if not old:
        console.print(f"[dim]{t('compact.too_short')}[/]")
        return

    summary_messages = [
        {"role": "system", "content": COMPACT_PROMPT},
        {"role": "user", "content": json.dumps(old, ensure_ascii=False, default=str)[:6000]},
    ]

    with console.status(f"[bold cyan]{t('compact.compressing')}", spinner="dots"):
        try:
            resp = client.chat(summary_messages, stream=False, max_tokens=1024)
            summary = resp["choices"][0]["message"]["content"]
        except Exception as e:
            console.print(f"[red]{t('compact.failed', error=e)}[/]")
            return

    messages.clear()
    messages.append({"role": "system", "content": system})
    messages.append({"role": "system", "content": f"[Conversation Summary]\n{summary}"})
    messages.extend(recent)

    console.print(f"[green]{t('compact.done', before=before, after=len(messages))}[/]")


# ── Image / Video slash commands ────────────────────────────────────


def _handle_image(client: AgnesClient, prompt: str) -> str:
    with console.status(f"[bold cyan]{t('image.generating')}", spinner="dots"):
        result = client.image_generate(prompt)

    data = result.get("data", [])
    if not data:
        console.print(f"[red]{t('image.no_result')}[/]")
        return "Error: No image returned from API"

    url = data[0].get("url", "")
    if url:
        console.print(f"[green]{t('image.done')}[/] {url}")
        webbrowser.open(url)
        return f"Image generated: {url}"
    else:
        console.print(f"[red]{t('image.no_url')}[/]")
        return "Error: No image URL in response"


def _handle_video(client: AgnesClient, prompt: str) -> str:
    with console.status(f"[bold cyan]{t('video.creating')}", spinner="dots"):
        result = client.video_create(prompt)

    task_id = result.get("task_id") or result.get("id", "")
    if not task_id:
        console.print(f"[red]{t('video.no_task_id')}[/]")
        return "Error: No task ID returned"

    console.print(t("video.waiting", id=task_id))

    elapsed = 0
    max_wait = 300
    while elapsed < max_wait:
        time.sleep(5)
        elapsed += 5
        status = client.video_status(task_id)
        st = status.get("status", "")
        progress = status.get("progress", 0)

        if st == "completed":
            url = status.get("video_url") or status.get("remixed_from_video_id", "")
            console.print(f"\n[green]{t('video.ready')}[/] {url}")
            if url:
                webbrowser.open(url)
            return f"Video generated ({elapsed}s): {url}"
        if st == "failed":
            console.print(f"\n[red]{t('video.failed', error=status.get('error', 'unknown'))}[/]")
            return f"Error: Video generation failed - {status.get('error', 'unknown')}"

        bar = "=" * (progress // 5) + " " * (20 - progress // 5)
        console.print(f"  [{bar}] {progress}% - {st} ({elapsed}s)")

    console.print(f"\n[red]{t('video.timed_out', s=max_wait, id=task_id)}[/]")
    return f"Error: Video generation timed out after {max_wait}s (task: {task_id})"


# ── Helpers ──────────────────────────────────────────────────────────


def _show_status(state: dict, messages: list, session_id: str) -> None:
    """Quick status overview."""
    th = f"[green]{t('config.on')}[/]" if state["thinking"] else f"[dim]{t('config.off')}[/]"
    au = f"[green]{t('config.on')}[/]" if state["auto_confirm"] else f"[dim]{t('config.off')}[/]"
    msg_count = len(messages)
    tokens = state.get("total_tokens", 0)
    console.print(
        f"  {t('status.session')}: [cyan]{session_id}[/]  "
        f"{t('status.messages')}: {msg_count}  "
        f"{t('status.tokens')}: {tokens}  "
        f"{t('status.thinking')}: {th}  "
        f"{t('status.auto')}: {au}  "
        f"{t('status.model')}: [dim]{state.get('model', '?')}[/]"
    )


def _show_sessions() -> None:
    sessions = list_sessions(limit=15)
    if not sessions:
        console.print(f"[dim]{t('session.no_saved')}[/]")
        return

    table = Table(title=t("session.title"), show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column(t("session.msgs"), justify="right")
    table.add_column("Preview")
    table.add_column("Time")

    for i, s in enumerate(sessions, 1):
        ts = datetime.fromtimestamp(s["modified"]).strftime("%m-%d %H:%M")
        table.add_row(str(i), s["id"], str(s["messages"]), s["preview"][:60], ts)

    console.print(table)


def _show_config(state: dict, cfg: dict) -> None:
    table = Table(title=t("config.title"), show_lines=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    for k, v in cfg.items():
        if k == "api_key":
            v = v[:8] + "..." if len(v) > 8 else v
        table.add_row(k, str(v))

    table.add_row("-" * 10, "-" * 10)
    table.add_row("thinking", str(state["thinking"]))
    table.add_row("auto_confirm", str(state["auto_confirm"]))
    table.add_row("max_iterations", str(state["max_iterations"]))
    table.add_row("session_id", state["session_id"])

    console.print(table)


def _show_tool_call(name: str, args: dict[str, Any]) -> None:
    display = {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in args.items()}
    console.print(f"\n[bold yellow]>> {name}[/] [dim]{json.dumps(display, ensure_ascii=False)}[/]")


def _show_result(name: str, result: str) -> None:
    lines = result.splitlines()
    max_lines = 40
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n... ({len(lines) - max_lines} more lines)"
    if len(preview) > 3000:
        preview = preview[:3000] + "\n... (truncated)"
    console.print(Panel(preview, title=f"[dim]{name} result[/]", border_style="dim", expand=False))


def _confirm() -> bool:
    return prompt_confirm(t("confirm.run"))


def _save_history(messages: list[dict[str, Any]], path: str) -> None:
    lines = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        if role == "user":
            lines.append(f"**You:** {m['content']}\n")
        elif role == "assistant":
            content = m.get("content") or ""
            if content:
                lines.append(f"**Agent:** {content}\n")
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                lines.append(f"  -> `{fn.get('name')}({fn.get('arguments', '')})`\n")
        elif role == "tool":
            lines.append(f"  >> {m.get('content', '')[:200]}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    console.print(f"[dim]{t('history.saved', path=path)}[/]")
