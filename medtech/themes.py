"""Small JSON-defined Textual palettes, with optional user-local extensions."""
import json
import os
import re
from dataclasses import fields
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual.color import Color, ColorParseError
from textual.theme import Theme

DEFAULT_THEME = "neutral"
_BUILTIN_NAMES = (DEFAULT_THEME, "colorful", "light")
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
_VARIABLE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_COLORS = {"primary", "secondary", "warning", "error", "success", "accent", "foreground",
           "background", "surface", "panel", "boost"}
_FALLBACK = {
    "primary": "#86a9c4", "background": "#14161b", "surface": "#1c1f26", "panel": "#22262e",
    "foreground": "#e7e9ee", "dark": True,
    "variables": {"medtech-border": "#343a46", "medtech-muted": "#a4aaba", "medtech-hover": "#292e38",
                  "medtech-selected": "#303b49", "medtech-user": "#252e3a",
                  "medtech-assistant": "#1c1f26", "medtech-code": "#15181e"},
}


def _read(text: str) -> Dict[str, dict]:
    """Reject non-object configuration; cap input before parsing user files."""
    if len(text) > 256_000:
        return {}
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _palette(name: str, values: Any, base: dict) -> Optional[Theme]:
    """Validate a complete palette before exposing it to Textual's CSS compiler."""
    if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(values, dict):
        return None
    allowed = {field.name for field in fields(Theme)} - {"name"}
    if set(values) - allowed:
        return None
    merged = {**base, **values}
    variables = values.get("variables", {})
    if not isinstance(variables, dict):
        return None
    merged["variables"] = {**base.get("variables", {}), **variables}
    try:
        for key in _COLORS:
            color = merged.get(key)
            if color is not None:
                if not isinstance(color, str):
                    return None
                Color.parse(color)
        if type(merged.get("dark", True)) is not bool:
            return None
        for key in ("luminosity_spread", "text_alpha"):
            value = merged.get(key, 0.15 if key == "luminosity_spread" else 0.95)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return None
        for key, value in merged["variables"].items():
            if not isinstance(key, str) or not _VARIABLE.fullmatch(key) or not isinstance(value, str):
                return None
            Color.parse(value)
        return Theme(name=name, **merged)
    except (ValueError, TypeError, ColorParseError):
        return None


def load_themes(user_path: Optional[Path] = None) -> Dict[str, Theme]:
    """Load built-ins and valid user overrides; bad files keep known-good defaults.

    The optional MEDTECH_HOME/themes.json file is a mapping from names to Theme
    properties. New palettes inherit neutral, so a single accent change is enough.
    Existing built-in palettes inherit their own values when partially overridden.
    Only color-valued CSS variables are supported by this small format.
    """
    try:
        bundled = _read(resources.files("medtech").joinpath("themes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bundled = {}
    source = {DEFAULT_THEME: _FALLBACK, **bundled}
    themes = {}
    definitions = {}
    for name, values in source.items():
        theme = _palette(name, values, _FALLBACK)
        if theme is not None:
            themes[name] = theme
            definitions[name] = {**_FALLBACK, **values,
                                 "variables": {**_FALLBACK["variables"], **values.get("variables", {})}}
    if DEFAULT_THEME not in themes:
        themes[DEFAULT_THEME] = Theme(name=DEFAULT_THEME, **_FALLBACK)
        definitions[DEFAULT_THEME] = _FALLBACK
    if user_path is None:
        directory = Path(os.environ.get("MEDTECH_HOME", str(Path.home() / ".medtech"))).expanduser()
        user_path = directory / "themes.json"
    try:
        # Avoid loading an unbounded custom file into memory.
        with Path(user_path).open(encoding="utf-8") as stream:
            custom = _read(stream.read(256_001))
    except (OSError, UnicodeError):
        custom = {}
    for name, values in custom.items():
        theme = _palette(name, values, definitions.get(name, definitions[DEFAULT_THEME]))
        if theme is not None:
            themes[name] = theme
    return themes


def theme_names() -> List[str]:
    """Return built-ins first, followed by alphabetically ordered custom palettes."""
    names = load_themes()
    return [name for name in _BUILTIN_NAMES if name in names] + sorted(set(names) - set(_BUILTIN_NAMES))


def apply_theme(app: Any, name: str) -> str:
    """Register JSON palettes and apply the requested one, falling back to neutral."""
    themes = load_themes()
    for theme in themes.values():
        app.register_theme(theme)
    selected = name if isinstance(name, str) and name in themes else DEFAULT_THEME
    app.theme = selected
    return selected
