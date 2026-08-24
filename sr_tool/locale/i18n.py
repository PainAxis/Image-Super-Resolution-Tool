"""Internationalization (i18n) support.

Loads translations from JSON files and provides a simple t(key, **kwargs)
interface. Notifies registered callbacks when the language changes so GUI
widgets can refresh their text.
"""

import json
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

_LOCALE_DIR = Path(__file__).resolve().parent.parent / "locale"

# Thread-safe translation cache
_LANGS: dict[str, dict[str, str]] = {}
_LANGS_LOCK = threading.Lock()

# Current language code
_current_lang = "en"

# Listeners to call when language changes
_listeners: list[Callable[[], None]] = []


def _load(lang: str) -> dict[str, str]:
    path = _LOCALE_DIR / f"{lang}.json"
    if not path.exists():
        raise FileNotFoundError(f"Language file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_languages() -> dict[str, str]:
    """Return {code: display_name} for all available languages."""
    result: dict[str, str] = {}
    for f_path in sorted(_LOCALE_DIR.glob("*.json")):
        try:
            data = json.loads(f_path.read_text(encoding="utf-8"))
            result[f_path.stem] = data.get("lang_name", f_path.stem)
        except (json.JSONDecodeError, OSError) as exc:
            # Corrupt or inaccessible file — skip it
            traceback.print_exc()
            print(f"[i18n] Skipping {f_path.name}: {exc}")
    return result


def _get_or_load(lang: str) -> dict[str, str]:
    """Thread-safe lazy-load of a language translation table."""
    # Fast path: read under lock
    with _LANGS_LOCK:
        cached = _LANGS.get(lang)
        if cached is not None:
            return cached

    # Slow path: load from disk, then install under lock
    try:
        table = _load(lang)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        traceback.print_exc()
        print(f"[i18n] Failed to load language '{lang}': {exc}")
        raise

    with _LANGS_LOCK:
        # Double-check in case another thread loaded it
        if lang not in _LANGS:
            _LANGS[lang] = table
        return _LANGS[lang]


def set_language(lang: str) -> None:
    """Switch the active language and notify listeners.

    Translations are loaded lazily on first access via t() — no eager
    disk I/O for languages that are never used.
    """
    global _current_lang
    try:
        _get_or_load(lang)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    _current_lang = lang

    for listener in _listeners:
        try:
            listener()
        except Exception:  # noqa: BLE001 - one listener must not break the others
            traceback.print_exc()


def current_language() -> str:
    return _current_lang


def t(key: str, **kwargs: object) -> str:
    """Return the translation for *key* in the current language.

    Format parameters can be passed as keyword arguments. Thread-safe.
    """
    lang = _current_lang
    table = _get_or_load(lang)

    text = table.get(key)
    if text is None:
        # Fall back to English
        en_table = _get_or_load("en")
        text = en_table.get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def on_language_change(callback: Callable[[], None]) -> None:
    """Register a callback to be called when set_language() is invoked."""
    _listeners.append(callback)


# ---------------------------------------------------------------------------
# Pre-load English so t() always has a fallback
# ---------------------------------------------------------------------------
_LANGS["en"] = _load("en")


def _detect_system_lang() -> str:
    """Detect system language; returns a language code or 'en'."""
    import locale as _locale

    try:
        sys_lang = _locale.getlocale()[0]
        if sys_lang:
            sys_lang = sys_lang.lower()
            for code in available_languages():
                if sys_lang.startswith(code):
                    return code
    except (ValueError, TypeError, _locale.Error):
        pass
    return "en"


# ---------------------------------------------------------------------------
# Set initial language based on system locale
# ---------------------------------------------------------------------------
try:
    detected = _detect_system_lang()
    if detected != "en":
        set_language(detected)
except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
    print(f"[i18n] Could not set detected language: {exc}")
