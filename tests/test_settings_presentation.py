"""Preference persistence and UI-only text reduction without changing facts."""
import json
from pathlib import Path

import pytest

from medtech.agent import Config
from medtech.presentation import brief_error, concise_text, tool_outcome
from medtech.settings import config_path, load_settings, save_settings


def test_defaults_do_not_create_profile_until_saved():
    path = config_path()
    assert not path.exists()
    cfg = Config()
    assert cfg.permissions.mode == "full" and cfg.permissions.confirm_dangerous
    assert not cfg.ui.details and not cfg.ui.ascii and cfg.ui.theme == "neutral"
    assert cfg.sandbox.extra_mounts == []
    assert not path.parent.exists()
    cfg.ui.details = True
    cfg.save_preferences()
    assert Config().ui.details is True


def test_settings_save_preserves_unrelated_user_keys():
    path = config_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"custom": {"keep": 1}, "ui": {"custom_ui_key": "retain", "theme": "light"}}))
    cfg = Config()
    cfg.permissions.mode = "workspace"
    cfg.permissions.root = "~/Projects"
    cfg.ui.details = True
    cfg.save_preferences()
    stored = json.loads(path.read_text())
    assert stored["custom"] == {"keep": 1}
    assert stored["ui"]["custom_ui_key"] == "retain"
    restored = Config()
    assert restored.permissions.mode == "workspace"
    assert restored.permissions.root == "~/Projects"
    assert restored.ui.details and restored.ui.theme == "light"


def test_environment_overrides_json_without_changing_file(monkeypatch):
    path = config_path()
    path.parent.mkdir(parents=True)
    original = '{"permissions":{"mode":"workspace"},"ui":{"details":false}}'
    path.write_text(original)
    monkeypatch.setenv("MEDTECH_PERMISSIONS_MODE", "full")
    monkeypatch.setenv("MEDTECH_CONFIRM_DANGEROUS", "off")
    monkeypatch.setenv("MEDTECH_DETAILS", "true")
    monkeypatch.setenv("MEDTECH_ASCII", "yes")
    monkeypatch.setenv("MEDTECH_THEME", "colorful")
    monkeypatch.setenv("MEDTECH_SANDBOX_MOUNTS", '[{"source":"/tmp/data","target":"/data","read_only":true}]')
    settings = load_settings()
    assert settings.permissions.mode == "full" and not settings.permissions.confirm_dangerous
    assert settings.ui.details and settings.ui.ascii and settings.ui.theme == "colorful"
    assert settings.sandbox.extra_mounts[0]["read_only"] is True
    assert path.read_text() == original


@pytest.mark.parametrize("config,reason", [
    ('[]', 'JSON object'), ('{"permissions":{"mode":"invalid"}}', 'permissions.mode'),
    ('{"permissions":{"confirm_dangerous":12}}', 'confirm_dangerous'),
    ('{"ui":{"details":"invalid"}}', 'ui.details'),
    ('{"sandbox":{"extra_mounts":["/tmp"]}}', 'entry needs'),
])
def test_invalid_preferences_fail_clearly_without_overwriting(config, reason):
    path = config_path()
    path.parent.mkdir(parents=True)
    path.write_text(config)
    with pytest.raises(ValueError, match=reason):
        Config()
    assert path.read_text() == config


def test_atomic_save_failure_preserves_original_config(monkeypatch):
    from medtech import settings
    cfg = Config()
    cfg.save_preferences()
    path = config_path()
    original = path.read_bytes()
    cfg.ui.details = True
    def denied(*args):
        raise PermissionError("read-only filesystem")
    monkeypatch.setattr(settings.os, "replace", denied)
    with pytest.raises(PermissionError, match="read-only filesystem"):
        cfg.save_preferences()
    assert path.read_bytes() == original
    assert not list(path.parent.glob('.config-*.json'))


def test_filler_reduction_keeps_result_facts_and_literal_code():
    source = "Sure! Let me check that for you.\nCreated /tmp/result.txt.\nPython was created in 1991.\nLet me know if you need anything else.\n```text\nLet me keep this literal sentence.\n```"
    reduced = concise_text(source)
    assert reduced == "Created /tmp/result.txt.\nPython was created in 1991.\n```text\nLet me keep this literal sentence.\n```"
    assert "Sure!" in source
    assert concise_text("<think>plan and internal work</think>Result: 42.") == "Result: 42."


def test_tool_outcome_uses_verified_path_and_real_failure_cause():
    result = {"success": True, "exit_code": 0, "cwd": "/start", "verified": [{"path": "/elsewhere/hello.txt", "exists": True, "kind": "file"}]}
    assert tool_outcome("write_file", {"path": "hello.txt"}, result) == "✓ Created /elsewhere/hello.txt"
    assert tool_outcome("write_file", {"path": "hello.txt"}, result, ascii=True) == "[OK] Created /elsewhere/hello.txt"
    trace = "Traceback (most recent call last):\n  File 'PRIVATE_LOG.py'\nPermissionError: [Errno 13] Permission denied: '/etc/hosts'"
    failed = {"success": False, "exit_code": 13, "stdout": "", "stderr": trace, "cwd": "/start", "verified": []}
    line = tool_outcome("write_file", {"path": "/etc/hosts"}, failed)
    assert "Permission denied" in line and "/etc/hosts" in line and "Ctrl+D" in line
    assert "Traceback" not in line and "PRIVATE_LOG" not in line
    assert failed["stderr"] == trace
    assert brief_error(trace).startswith("Permission denied")


def test_requested_tool_output_is_not_filtered_as_model_filler():
    output = "Let me know if the application printed this exact line.\n42"
    result = {"success": True, "exit_code": 0, "stdout": output, "stderr": "", "cwd": "/tmp"}
    assert tool_outcome("read_file", {"path": "output.txt"}, result) == output


def test_missing_filesystem_verification_never_displays_a_success_claim():
    result = {"success": True, "exit_code": 0, "cwd": "/tmp", "verified": []}
    line = tool_outcome("mkdir", {"path": "unverified"}, result)
    assert line.startswith("✗") and "could not be verified" in line
    assert "Created" not in line
