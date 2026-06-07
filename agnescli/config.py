"""API key resolution with persistent config."""

import json
import os
import sys
from pathlib import Path

API_KEY_ENV = "AGNES_API_KEY"
CONFIG_DIR = Path.home() / ".agnescli"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def save_api_key(key: str) -> None:
    cfg = _load_config()
    cfg["api_key"] = key
    _save_config(cfg)


def resolve_api_key(explicit: str | None = None) -> str:
    # 1. Explicit flag
    if explicit:
        return explicit
    # 2. Environment variable
    env_key = os.environ.get(API_KEY_ENV, "")
    if env_key:
        return env_key
    # 3. Saved config
    cfg = _load_config()
    saved = cfg.get("api_key", "")
    if saved:
        return saved
    # 4. Not found
    print(
        f"[!] No API key found.\n    Run: agnescli setup\n    Or set: {API_KEY_ENV}=your_key",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULTS = {
    "model": "agnes-2.0-flash",
    "thinking": False,
    "auto_confirm": False,
    "max_iterations": 50,
    "max_tokens": 4096,
    "temperature": 0.7,
}


def get_config() -> dict:
    """Load full config with defaults applied."""
    cfg = dict(DEFAULTS)
    cfg.update(_load_config())
    return cfg


def save_config_value(key: str, value: object) -> None:
    """Save a single config value."""
    cfg = _load_config()
    cfg[key] = value
    _save_config(cfg)
