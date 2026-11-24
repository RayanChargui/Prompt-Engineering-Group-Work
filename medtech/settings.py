"""Small, local JSON preferences with explicit environment overrides."""
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional



@dataclass
class PermissionsConfig:
    """Choose local path scope and confirmation of catastrophic operations."""
    mode: str = "full"
    root: Optional[str] = None
    confirm_dangerous: bool = True


@dataclass
class UIConfig:
    """Persist presentation choices independently of chat transcripts."""
    details: bool = False
    theme: str = "neutral"
    ascii: bool = False


@dataclass
class SandboxConfig:
    """Additional explicit host-to-container bind mounts; empty by default."""
    extra_mounts: list = field(default_factory=list)


@dataclass
class Settings:
    """The user-editable sections of MEDTECH_HOME/config.json."""
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


def config_path() -> Path:
    """Return the preferences path, separate from the SQLite database."""
    return Path(os.environ.get("MEDTECH_HOME", Path.home() / ".medtech")).expanduser() / "config.json"


def _boolean(value, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.casefold() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.casefold() in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read preferences {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Preferences {path} must contain a JSON object")
    return data


def load_settings() -> Settings:
    """Apply defaults, JSON preferences, then named environment overrides."""
    data = _read(config_path())
    for section in ("permissions", "ui", "sandbox"):
        if not isinstance(data.get(section, {}), dict):
            raise ValueError(f"config.{section} must be an object")
    permissions, ui, sandbox = (data.get(name, {}) for name in ("permissions", "ui", "sandbox"))
    mode = os.environ.get("MEDTECH_PERMISSIONS_MODE", permissions.get("mode", "full"))
    root = os.environ.get("MEDTECH_WORKSPACE_ROOT", permissions.get("root"))
    if mode not in {"full", "workspace"}:
        raise ValueError("permissions.mode must be full or workspace")
    if root is not None and (not isinstance(root, str) or not root.strip()):
        raise ValueError("permissions.root must be a nonempty path or null")
    confirm = _boolean(os.environ.get("MEDTECH_CONFIRM_DANGEROUS", permissions.get("confirm_dangerous", True)),
                       "permissions.confirm_dangerous")
    details = _boolean(os.environ.get("MEDTECH_DETAILS", ui.get("details", False)), "ui.details")
    ascii_only = _boolean(os.environ.get("MEDTECH_ASCII", ui.get("ascii", False)), "ui.ascii")
    theme = os.environ.get("MEDTECH_THEME", ui.get("theme", "neutral"))
    if not isinstance(theme, str) or not theme.strip():
        raise ValueError("ui.theme must be a nonempty theme name")
    try:
        mounts = json.loads(os.environ["MEDTECH_SANDBOX_MOUNTS"]) if "MEDTECH_SANDBOX_MOUNTS" in os.environ else sandbox.get("extra_mounts", [])
    except ValueError as exc:
        raise ValueError("MEDTECH_SANDBOX_MOUNTS must be a JSON array of mount objects") from exc
    if not isinstance(mounts, list):
        raise ValueError("sandbox.extra_mounts must be an array")
    checked_mounts = []
    for mount in mounts:
        if (not isinstance(mount, dict) or set(mount) - {"source", "target", "read_only"}
                or not all(isinstance(mount.get(key), str) and mount[key].strip() for key in ("source", "target"))):
            raise ValueError("Each sandbox.extra_mounts entry needs source and target paths")
        checked_mounts.append({"source": mount["source"], "target": mount["target"],
                               "read_only": _boolean(mount.get("read_only", False), "mount.read_only")})
    return Settings(PermissionsConfig(mode, root, confirm), UIConfig(details, theme.strip(), ascii_only),
                    SandboxConfig(checked_mounts))


def save_settings(settings: Settings) -> None:
    """Atomically save known sections, preserving unrelated user preferences."""
    path = config_path()
    data = _read(path)
    for section, values in asdict(settings).items():
        previous = data.get(section, {})
        data[section] = {**previous, **values} if isinstance(previous, dict) else values
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
