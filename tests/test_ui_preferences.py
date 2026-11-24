"""Concise presentation, persistent preferences, and local permission UX."""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from textual.containers import Vertical
from textual.widgets import Collapsible, Select, Static, Switch

from medtech.agent import Config
from medtech.backends import LocalBackend
from medtech.store import Store
from medtech.tui import MedTechApp


class ScriptedModel:
    """Keep extraction independent from deterministic scripted agent turns."""
    def __init__(self, *responses):
        self.responses = iter(responses)

    def __call__(self, messages):
        if messages[0]["content"].startswith("Extract only"):
            return '{"facts":[]}'
        return json.dumps(next(self.responses))


def reply(say="", tool=None, **args):
    return {"say": say, "action": None if tool is None else {"tool": tool, "args": args}}


def text(app):
    values = []
    for widget in app.query(Static):
        value = widget.renderable
        values.append(getattr(value, "plain", str(value)))
    return "\n".join(values)


async def wait_for(pilot, predicate):
    deadline = time.monotonic() + 6
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for preference/presentation state")
        await pilot.pause(0.03)


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "user-data"))
    store = Store(tmp_path / "chats.db")
    backend = LocalBackend(str(tmp_path / "workspace"))
    yield Config(), store, backend
    store.close()


def creation_model(backend):
    return ScriptedModel(
        reply("Let me think carefully about creating the requested folder.", "mkdir", path="test_dir"),
        reply("I will now proceed to write the file for you.", "write_file", path="test_dir/hello.txt", content="hi"),
        reply(f"Created folder {backend.root}/test_dir and file {backend.root}/test_dir/hello.txt."),
    )


@pytest.mark.asyncio
async def test_default_success_is_concise_without_losing_raw_audit(state):
    cfg, store, backend = state
    assert cfg.ui.details is False
    app = MedTechApp(cfg, backend, store, llm=creation_model(backend))
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Create folder test_dir with hello.txt containing hi")
        await wait_for(pilot, lambda: not app.busy)
        assert (Path(backend.root) / "test_dir/hello.txt").read_text() == "hi"
        visible = text(app)
        assert "hello.txt" in visible
        assert "Let me think" not in visible
        assert "proceed to write" not in visible
        assert '"exit_code"' not in visible
        assert '"verified"' not in visible
        assert not app.query(Collapsible)
        assert not app.screen.query("#tool-allow")
        messages = store.messages(app.sid)
        assert sum(row["role"] == "tool" for row in messages) == 2
        assert any("Let me think" in row["content"] for row in messages)
        assert all(json.loads(row["output"])["success"] for row in store.tasks(app.sid))
        await pilot.press("ctrl+d")
        assert "Let me think" in text(app)
        assert '"exit_code"' in text(app)
        await pilot.press("ctrl+d")
        assert "Let me think" not in text(app)


@pytest.mark.asyncio
async def test_failure_is_short_and_details_toggle_restores_complete_diagnostics(state, monkeypatch):
    cfg, store, backend = state
    stderr = ("Traceback (most recent call last):\n  File '/local/RAW_DIAGNOSTIC_SENTINEL.py', line 12, in run\n"
              "PermissionError: [Errno 13] Permission denied: /protected/notes.txt")
    result = {"success": False, "exit_code": 13, "stdout": "", "stderr": stderr,
              "cwd": backend.root, "duration": 0.25, "verified": []}
    monkeypatch.setattr(backend, "run", lambda tool, args: result)
    model = ScriptedModel(reply("Let me write your file.", "write_file", path="notes.txt", content="hi"),
                          reply("Could not write notes.txt: permission denied."))
    app = MedTechApp(cfg, backend, store, llm=model)
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Write notes.txt containing hi")
        await wait_for(pilot, lambda: not app.busy)
        visible = text(app)
        assert "permission denied" in visible.lower()
        assert "Ctrl+D" in visible
        assert "Traceback" not in visible
        assert "RAW_DIAGNOSTIC_SENTINEL" not in visible
        assert not app.query(Collapsible)
        assert json.loads(store.tasks(app.sid)[0]["output"])["stderr"] == stderr
        await pilot.press("ctrl+d")
        assert cfg.ui.details is True
        assert app.query(Collapsible)
        assert "RAW_DIAGNOSTIC_SENTINEL" in text(app)
        assert '"exit_code": 13' in text(app)
        assert Config().ui.details is True
        await app.send_message("/details off")
        assert cfg.ui.details is False
        assert not app.query(Collapsible)
        assert Config().ui.details is False
        await app.send_message("/details on")
        assert '"exit_code": 13' in text(app)


@pytest.mark.asyncio
async def test_details_retroactively_exposes_guard_retries_and_tool_arguments(state):
    cfg, store, backend = state
    model = ScriptedModel(reply("I created folder phantom."),
                          reply("Let me create the actual folder.", "mkdir", path="real_dir"),
                          reply(f"Created folder {backend.root}/real_dir."))
    app = MedTechApp(cfg, backend, store, llm=model)
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Create folder real_dir")
        await wait_for(pilot, lambda: not app.busy)
        assert "phantom" not in text(app)
        assert "Rejected unsupported" not in text(app)
        assert '"verified"' not in text(app)
        await app.send_message("/details on")
        detailed = text(app)
        assert "phantom" in detailed
        assert "Rejected unsupported execution claim" in detailed
        assert '"path": "real_dir"' in detailed
        assert '"verified"' in detailed
        assert not (Path(backend.root) / "phantom").exists()
        assert (Path(backend.root) / "real_dir").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [60, 80, 120, 200])
async def test_themes_ascii_and_responsive_layout_persist(state, width):
    cfg, store, backend = state
    app = MedTechApp(cfg, backend, store, llm=ScriptedModel(reply("Hello.")))
    async with app.run_test(size=(width, 36)) as pilot:
        for theme in ("colorful", "light", "neutral"):
            await app.send_message("/theme " + theme)
            await pilot.pause()
            assert cfg.ui.theme == theme
            assert Config().ui.theme == theme
            assert app.query_one("#message-input").region.width > max(15, width // 4)
        await pilot.press("ctrl+t")
        await pilot.pause()
        panel = app.screen.query_one(Vertical)
        assert 0 <= panel.region.x and panel.region.right <= width
        app.screen.query_one("#theme-select", Select).value = "light"
        app.screen.query_one("#ascii-switch", Switch).value = True
        await pilot.click("#theme-save")
        assert cfg.ui.theme == "light"
        assert cfg.ui.ascii is True
        loaded = Config()
        assert loaded.ui.theme == "light" and loaded.ui.ascii is True
        assert "◈" not in str(app.query_one("#brand", Static).renderable)


@pytest.mark.asyncio
async def test_permissions_context_and_workspace_toggle_are_visible_and_persisted(state):
    cfg, store, backend = state
    app = MedTechApp(cfg, backend, store, llm=ScriptedModel(reply("Hello.")))
    async with app.run_test(size=(120, 38)) as pilot:
        assert cfg.permissions.mode == "full"
        assert "full" in text(app).lower()
        assert backend.root in text(app)
        await app.send_message("/permissions workspace")
        assert cfg.permissions.mode == "workspace"
        assert backend.permissions_mode == "workspace"
        assert "workspace" in str(app.query_one("#sidebar-footer", Static).renderable).lower()
        assert Config().permissions.mode == "workspace"
        await app.send_message("/permissions full")
        assert cfg.permissions.mode == "full"
        assert backend.permissions_mode == "full"
        assert Config().permissions.mode == "full"


@pytest.mark.asyncio
async def test_dangerous_operation_requires_confirmation_and_decline_never_executes(state, monkeypatch):
    cfg, store, backend = state
    calls = []
    monkeypatch.setattr(backend, "run", lambda *args: calls.append(args))
    model = ScriptedModel(reply("", "delete", path="/", recursive=True))
    app = MedTechApp(cfg, backend, store, llm=model)
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Delete the entire root filesystem")
        await wait_for(pilot, lambda: bool(app.screen.query("#tool-allow")))
        await pilot.pause()
        assert calls == []
        await pilot.click("#cancel")
        await wait_for(pilot, lambda: not app.busy)
        assert calls == []
        assert store.tasks(app.sid)[0]["status"] == "denied"
        assert "skipped" in text(app).lower()


@pytest.mark.asyncio
async def test_danger_confirmation_can_be_skipped_without_real_destructive_execution(state, monkeypatch):
    cfg, store, backend = state
    cfg.permissions.confirm_dangerous = False
    backend.confirm_dangerous = False  # Match the CLI-created backend for this setting.
    calls = []

    def simulate_refusal(tool, args):
        calls.append((tool, args))
        return {"success": False, "exit_code": 1, "stdout": "", "stderr": "Simulated backend refused; no operation executed.",
                "cwd": backend.root, "duration": 0.0, "verified": []}

    monkeypatch.setattr(backend, "run", simulate_refusal)
    model = ScriptedModel(reply("", "delete", path="/", recursive=True),
                          reply("The simulated backend refused; no operation succeeded."))
    app = MedTechApp(cfg, backend, store, llm=model)
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Delete the entire root filesystem")
        await wait_for(pilot, lambda: not app.busy)
        assert not app.screen.query("#tool-allow")
        assert calls == [("delete", {"path": "/", "recursive": True})]
        assert store.tasks(app.sid)[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_export_requested_concise_and_details_screenshots(state, tmp_path, monkeypatch):
    """Generate portable terminal SVG evidence using only synthetic local fixtures."""
    cfg, store, backend = state
    target = Path(os.environ.get("MEDTECH_DOC_SCREENSHOTS", tmp_path / "screenshots"))
    target.mkdir(parents=True, exist_ok=True)

    def capture(app, name):
        svg = app.export_screenshot()
        assert "<svg" in svg
        (target / (name + ".svg")).write_text(svg, encoding="utf-8")

    app = MedTechApp(cfg, backend, store, llm=creation_model(backend))
    async with app.run_test(size=(120, 38)) as pilot:
        await pilot.pause()
        capture(app, "empty")
        await app.send_message("Create test_dir with hello.txt containing hi")
        await wait_for(pilot, lambda: not app.busy)
        capture(app, "success-concise")

    def failed_write(tool, args):
        return {"success": False, "exit_code": 13, "stdout": "", "stderr": "PermissionError: [Errno 13] Permission denied: /protected/notes.txt",
                "cwd": backend.root, "duration": 0.03, "verified": []}

    monkeypatch.setattr(backend, "run", failed_write)
    model = ScriptedModel(reply("", "write_file", path="notes.txt", content="hi"),
                          reply("Could not write notes.txt: permission denied."))
    app = MedTechApp(cfg, backend, store, llm=model)
    async with app.run_test(size=(120, 38)) as pilot:
        await app.send_message("Write notes.txt containing hi")
        await wait_for(pilot, lambda: not app.busy)
        capture(app, "failure-concise")
        await app.send_message("/details on")
        await pilot.pause()
        for section in app.query(Collapsible):
            section.collapsed = False
        await pilot.pause()
        capture(app, "failure-details")

    now = datetime.now()
    for title, days in (("Plan the terminal interface", 0), ("Fix local tool execution", 1),
                        ("Explore Python workspace tools", 4), ("Older project notes", 25)):
        sid = store.new_session("local", backend.root)
        store.add_message(sid, "user", title)
        store.add_message(sid, "assistant", json.dumps(reply("Saved project notes.")))
        store._w("UPDATE sessions SET updated_at=? WHERE id=?", (now - timedelta(days=days)).timestamp(), sid)
    cfg.ui.details = False
    app = MedTechApp(cfg, backend, store, llm=ScriptedModel(reply("Hello.")))
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        capture(app, "sidebar-grouped")
    assert {path.name for path in target.glob("*.svg")} >= {
        "empty.svg", "success-concise.svg", "failure-concise.svg", "failure-details.svg", "sidebar-grouped.svg",
    }
