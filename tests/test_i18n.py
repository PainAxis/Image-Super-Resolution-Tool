"""Translation catalog completeness tests."""

import json
import locale
from pathlib import Path

import pytest

from sr_tool.locale import i18n


def test_every_catalog_has_the_english_keys() -> None:
    locale_dir = Path(__file__).parents[1] / "sr_tool" / "locale"
    catalogs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in locale_dir.glob("*.json")
    }
    expected = set(catalogs["en"])
    for language, catalog in catalogs.items():
        assert set(catalog) == expected, language


def test_language_switch_fallback_formatting_and_listener() -> None:
    called: list[str] = []
    i18n.on_language_change(lambda: called.append(i18n.current_language()))
    try:
        i18n.set_language("zh")
        assert i18n.current_language() == "zh"
        assert "已加载" in i18n.t("status_loaded", name="x", w=1, h=2)
        assert i18n.t("missing-key") == "missing-key"
        assert (
            i18n.t("status_loaded", wrong="value") == i18n._LANGS["zh"]["status_loaded"]
        )
        assert called[-1] == "zh"
    finally:
        i18n.set_language("en")


def test_unknown_language_does_not_change_selection() -> None:
    i18n.set_language("en")
    i18n.set_language("not-a-language")
    assert i18n.current_language() == "en"


def test_available_languages_skips_a_broken_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "ok.json").write_text('{"lang_name":"OK"}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(i18n, "_LOCALE_DIR", tmp_path)
    assert i18n.available_languages() == {"ok": "OK"}


def test_system_language_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locale, "getlocale", lambda: ("zh_CN", "UTF-8"))
    assert i18n._detect_system_lang() == "zh"
    monkeypatch.setattr(locale, "getlocale", lambda: (None, None))
    assert i18n._detect_system_lang() == "en"
