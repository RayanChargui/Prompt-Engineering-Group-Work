"""Keep saved application preferences isolated from developer/user state."""
import pytest


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    """Each test gets its own profile and can override it explicitly as needed."""
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "profile"))
    for name in ("MEDTECH_PERMISSIONS_MODE", "MEDTECH_WORKSPACE_ROOT", "MEDTECH_CONFIRM_DANGEROUS",
                 "MEDTECH_DETAILS", "MEDTECH_THEME", "MEDTECH_ASCII", "MEDTECH_SANDBOX_MOUNTS"):
        monkeypatch.delenv(name, raising=False)
