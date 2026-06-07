"""Internationalization support for Agnescli."""

from __future__ import annotations

import json
import locale
from pathlib import Path

_LOCALES_DIR = Path(__file__).parent / "locales"
_translations: dict[str, str] = {}
_current_lang: str = "en"
_available_langs: list[str] = []


def _detect_lang() -> str:
    """Auto-detect language from system locale."""
    try:
        sys_lang = locale.getdefaultlocale()[0] or ""
    except Exception:
        sys_lang = ""

    if sys_lang.startswith("zh"):
        return "zh"
    return "en"


def _load_lang(lang: str) -> dict[str, str]:
    """Load a language file and return flat key-value dict."""
    path = _LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init(lang: str | None = None) -> None:
    """Initialize i18n. Call once at startup."""
    global _translations, _current_lang, _available_langs

    # Discover available languages
    if _LOCALES_DIR.exists():
        _available_langs = [p.stem for p in _LOCALES_DIR.glob("*.json")]
    else:
        _available_langs = ["en"]

    # Determine language
    if lang and lang in _available_langs:
        _current_lang = lang
    else:
        _current_lang = _detect_lang()

    _translations = _load_lang(_current_lang)


def set_lang(lang: str) -> bool:
    """Switch language. Returns True if successful."""
    global _translations, _current_lang
    if lang not in _available_langs:
        return False
    _translations = _load_lang(lang)
    _current_lang = lang
    return True


def get_lang() -> str:
    return _current_lang


def available_langs() -> list[str]:
    return list(_available_langs)


def lang_display_name(lang: str) -> str:
    """Get display name for a language from its own translation file."""
    translations = _load_lang(lang)
    return translations.get("lang_name", lang)


def t(key: str, **kwargs: object) -> str:
    """Get translated string by key. Supports {placeholder} substitution."""
    template = _translations.get(key)
    if template is None:
        # Fallback to English
        en = _load_lang("en")
        template = en.get(key, key)
    if kwargs:
        return template.format(**kwargs)
    return template
