"""Session persistence - save, resume, list conversations."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path.home() / ".agnescli" / "sessions"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.jsonl"


def save_message(session_id: str, message: dict[str, Any]) -> None:
    _ensure_dir()
    path = session_path(session_id)
    with open(path, "a", encoding="utf-8") as f:
        entry = {"ts": time.time(), "message": message}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_session(session_id: str) -> list[dict[str, Any]]:
    path = session_path(session_id)
    if not path.exists():
        return []
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entry = json.loads(line)
                messages.append(entry["message"])
            except (json.JSONDecodeError, KeyError):
                continue
    return messages


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    _ensure_dir()
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        sid = f.stem
        messages = load_session(sid)
        # Get first user message as preview
        preview = ""
        for m in messages:
            if m.get("role") == "user":
                preview = m.get("content", "")[:80]
                break
        sessions.append(
            {
                "id": sid,
                "modified": f.stat().st_mtime,
                "messages": len(messages),
                "preview": preview,
            }
        )
        if len(sessions) >= limit:
            break
    return sessions


def get_last_session_id() -> str | None:
    sessions = list_sessions(limit=1)
    return sessions[0]["id"] if sessions else None
