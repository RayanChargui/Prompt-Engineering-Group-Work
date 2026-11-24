"""Validate custom palette safety and compile the shared stylesheet in each theme."""
import json
from pathlib import Path

import pytest
from textual.app import App
from textual.css.stylesheet import Stylesheet

from medtech.themes import apply_theme, load_themes, theme_names


def test_custom_theme_inherits_complete_palette_and_is_discoverable(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path))
    (tmp_path / "themes.json").write_text(json.dumps({
        "ocean": {"primary": "#00aabb", "variables": {"medtech-user": "#193344"}},
        "light": {"accent": "#006688"},
    }), encoding="utf-8")
    themes = load_themes()
    assert themes["ocean"].primary == "#00aabb"
    assert themes["ocean"].background == themes["neutral"].background
    assert themes["ocean"].variables["medtech-user"] == "#193344"
    assert themes["ocean"].variables["medtech-border"] == themes["neutral"].variables["medtech-border"]
    assert themes["light"].dark is False
    assert themes["light"].accent == "#006688"
    assert theme_names() == ["neutral", "colorful", "light", "ocean"]
    app = App()
    assert apply_theme(app, "ocean") == "ocean"
    assert app.current_theme.primary == "#00aabb"


@pytest.mark.parametrize("values", [
    {"primary": "definitely-not-a-color"},
    {"primary": "#334455", "dark": "false"},
    {"primary": "#334455", "variables": {"medtech-border": "invalid!"}},
    {"primary": "#334455", "variables": ["not", "an", "object"]},
    {"primary": "#334455", "variables": {"bad variable": "#ffffff"}},
    {"primary": "#334455", "text_alpha": float("nan")},
    {"primary": "#334455", "unrecognized_property": "#ffffff"},
])
def test_invalid_user_palette_cannot_partially_override_known_good_theme(tmp_path, values):
    path = tmp_path / "themes.json"
    path.write_text(json.dumps({"neutral": values, "broken": values}), encoding="utf-8")
    themes = load_themes(path)
    assert themes["neutral"].primary == "#86a9c4"
    assert "broken" not in themes


@pytest.mark.parametrize("text", ["{incomplete", "[]", "null", '"palette"', "x" * 256_001])
def test_corrupt_or_excessive_user_config_preserves_fallback(tmp_path, text, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path))
    (tmp_path / "themes.json").write_text(text, encoding="utf-8")
    app = App()
    assert apply_theme(app, "missing-custom-theme") == "neutral"
    assert app.theme == "neutral"


@pytest.mark.parametrize("name", ["neutral", "colorful", "light"])
def test_entire_shared_stylesheet_compiles_for_each_palette(name, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path))
    app = App()
    apply_theme(app, name)
    stylesheet = Stylesheet(variables=app.get_css_variables())
    css_path = Path(__file__).resolve().parents[1] / "medtech" / "styles.tcss"
    stylesheet.add_source(css_path.read_text(encoding="utf-8"))
    stylesheet.parse()
