"""Ensure the user-facing acceptance runner fails clearly and conservatively."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from medtech.backends import LocalBackend

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_local.py"


@pytest.fixture(autouse=True)
def isolated_preferences(tmp_path, monkeypatch):
    """Keep persisted preferences isolated from the developer's actual user data."""
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "preferences"))


def test_worker_dependency_failure_is_brief(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--worker", "resume", "--root", str(tmp_path)],
                            capture_output=True, text=True,
                            env={**os.environ, "MEDTECH_HOME": str(tmp_path / "data")})
    assert result.returncode == 1
    assert "Prerequisite failed" in result.stderr
    assert "Traceback" not in result.stderr


def test_worker_unreachable_ollama_failure_is_brief(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--worker", "remember", "--root", str(tmp_path)],
                            capture_output=True, text=True, timeout=20,
                            env={**os.environ, "MEDTECH_HOME": str(tmp_path / "data"),
                                 "OLLAMA_URL": "http://127.0.0.1:1"})
    assert result.returncode == 1
    assert "Cannot reach Ollama" in result.stderr
    assert "Traceback" not in result.stderr


def test_creation_artifact_without_accepted_final_is_not_a_pass(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("verify_local", SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    class StoppedAgent:
        def __init__(self, cfg, backend, store, sid, ui, **kwargs):
            self.backend, self.ui = backend, ui

        def turn(self, text):
            result = self.backend.run("write_file", {"path": "test_dir/hello.txt", "content": "hi"})
            assert result["success"]
            self.ui.warn("Stopped: retry limit reached")

    monkeypatch.setattr(verifier, "Agent", StoppedAgent)
    with pytest.raises(RuntimeError, match="accepted final completion"):
        verifier.real_creation(LocalBackend(str(tmp_path / "workspace")), tmp_path / "check.db")


def test_headless_ui_check_does_not_contact_ollama(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("verify_local_ui", SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    calls = []

    def unexpected_model_call(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Headless startup must not contact Ollama")

    import medtech.agent
    monkeypatch.setattr(medtech.agent.urllib.request, "urlopen", unexpected_model_call)
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "data"))
    verifier.ui_check(tmp_path)
    assert not calls


def load_verifier(name="verify_followup"):
    """Load the local verification script without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def test_full_access_check_writes_outside_project_and_cleans_home(tmp_path, monkeypatch):
    verifier = load_verifier()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(verifier.Path, "home", classmethod(lambda cls: fake_home))
    verifier.full_access_check(project)
    assert list(fake_home.iterdir()) == []
    assert (project / "full-access-cwd").is_dir()


def test_concise_check_uses_fake_model_and_preserves_full_history(tmp_path, monkeypatch):
    verifier = load_verifier("verify_concise")
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "data"))
    verifier.concise_check(tmp_path)
    assert (tmp_path / "concise-cwd/result.txt").read_text() == "hi"


def test_legacy_failure_check_remains_explicitly_restricted(tmp_path):
    verifier = load_verifier("verify_safety")
    verifier.safety_checks(tmp_path)
    assert (tmp_path / "corrupted.db").read_bytes() == b"this is not a SQLite database"
