"""Command routing and restart behavior without a model server."""
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from medtech import cli
from medtech.backends import LocalBackend
from medtech.store import Store


@pytest.fixture
def cli_state(tmp_path, monkeypatch):
    store = Store(tmp_path / "chat.db")
    backend = LocalBackend(str(tmp_path / "files"))
    sid = store.new_session("local", backend.root)
    agent = SimpleNamespace(sid=sid, backend=backend, auto=False)
    agent.resume = lambda new_sid: setattr(agent, "sid", new_sid)
    output = io.StringIO()
    monkeypatch.setattr(cli, "c", Console(file=output, width=120, color_system=None))
    yield store, agent, output
    store.close()


def test_chat_commands(cli_state):
    store, agent, output = cli_state
    first = agent.sid
    store.add_message(first, "user", "First conversation")
    cli.command(f'/rename {first} "Project plans"', agent, store)
    assert store.session(first)["title"] == "Project plans"
    cli.command("/history", agent, store)
    assert "Project plans" in output.getvalue()
    cli.command("/chats", agent, store)
    cli.command("/new", agent, store)
    assert agent.sid != first
    second = agent.sid
    cli.command(f"/resume {first}", agent, store)
    assert agent.sid == first
    assert agent.backend.root == store.session(first)["workspace"]
    cli.command(f"/history {first}", agent, store)
    assert "First conversation" in output.getvalue()
    cli.command(f"/delete {second}", agent, store)
    assert store.session(second) is None
    cli.command(f"/delete {first}", agent, store)
    assert store.session(agent.sid) is not None


def test_memory_commands(cli_state):
    store, agent, output = cli_state
    cli.command('/remember "my name is Rayan"', agent, store)
    assert store.memories()
    cli.command("/memory", agent, store)
    assert "Rayan" in output.getvalue()
    fact_id = store.memories()[0]["id"]
    cli.command(f"/forget {fact_id}", agent, store)
    assert not store.memories()
    cli.command("/remember I prefer Python", agent, store)
    cli.command("/forget I prefer Python", agent, store)
    assert not store.memories()
    cli.command("/remember I use Linux", agent, store)
    cli.command("/memory clear", agent, store)
    assert not store.memories()


def test_invalid_commands_do_not_lose_session(cli_state):
    store, agent, output = cli_state
    sid = agent.sid
    for line in ("/resume missing", "/resume 9999", "/rename 9999 title", "/history nope", '/remember "'):
        cli.command(line, agent, store)
    assert agent.sid == sid
    assert store.session(sid) is not None
    assert "no session 9999" in output.getvalue()


def test_resume_flags():
    parser = cli.build_parser()
    assert parser.parse_args(["--resume"]).resume == -1
    assert parser.parse_args(["--resume", "12"]).resume == 12
    assert parser.parse_args(["--continue"]).continue_latest
    assert parser.parse_args(["--debug"]).debug


def test_cli_restart_resumes_original_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "state"))
    store = Store()
    root = str(tmp_path / "original")
    sid = store.new_session("local", root)
    store.add_message(sid, "user", "My original conversation")
    store.add_message(sid, "assistant", json.dumps({"say": "Saved", "action": None}))
    store.close()
    # A separate Python process launches and restores the saved session.
    result = subprocess.run(
        [sys.executable, "-m", "medtech", "--resume", str(sid), "--workspace", str(tmp_path / "wrong")],
        input="exit\n", capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert root in result.stdout
    assert not (tmp_path / "wrong").exists()
    reopened = Store()
    assert len(reopened.messages(sid)) == 2
    reopened.close()


def test_corrupt_database_has_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path))
    (tmp_path / "medtech.db").write_bytes(b"not a SQLite database")
    output = io.StringIO()
    monkeypatch.setattr(cli, "c", Console(file=output, color_system=None))
    assert cli.main(["sessions"]) == 1
    assert "Error:" in output.getvalue()
    assert "Traceback" not in output.getvalue()


def test_unreachable_ollama_keeps_user_input_without_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "state"))
    # Bind an ephemeral local port without listening, guaranteeing connection refusal.
    import socket
    with socket.socket() as blocked:
        blocked.bind(("127.0.0.1", 0))
        monkeypatch.setenv("OLLAMA_URL", f"http://127.0.0.1:{blocked.getsockname()[1]}")
        result = subprocess.run([sys.executable, "-m", "medtech", "--mode", "local",
                                 "--workspace", str(tmp_path / "files")], input="Keep this message\nexit\n",
                                capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    assert "Cannot reach Ollama" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
    store = Store()
    assert store.messages(store.latest_session()["id"])[0]["content"] == "Keep this message"
    store.close()


def test_interactive_launch_uses_sidebar_ui(tmp_path, monkeypatch):
    """A real-terminal launch selects the TUI and forwards saved session/options."""
    from medtech import tui
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path / "data"))
    store = Store()
    workspace = str(tmp_path / "workspace")
    sid = store.new_session("local", workspace)
    store.add_message(sid, "user", "Saved chat")
    store.close()
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    launched = {}

    class FakeApp:
        def __init__(self, cfg, backend, store, **kwargs):
            launched.update(cfg=cfg, backend=backend, **kwargs)
        def run(self):
            launched["ran"] = True

    monkeypatch.setattr(tui, "MedTechApp", FakeApp)
    assert cli.main(["--continue", "--num-ctx", "8192", "--debug"]) is None
    assert launched["ran"] and launched["sid"] == sid
    assert launched["backend"].root == workspace
    assert launched["cfg"].num_ctx == 8192
    assert launched["debug"] is True


def test_explicit_plain_and_debug_command(cli_state):
    store, agent, output = cli_state
    agent.debug = False
    cli.command("/debug on", agent, store)
    assert agent.debug is True
    cli.command("/debug off", agent, store)
    assert agent.debug is False
    assert "debug.log" in output.getvalue()
    assert cli.build_parser().parse_args(["--plain", "--num-ctx", "8192"]).plain


def test_invalid_context_option_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDTECH_HOME", str(tmp_path))
    output = io.StringIO()
    monkeypatch.setattr(cli, "c", Console(file=output, color_system=None))
    assert cli.main(["--num-ctx", "99999", "--mode", "local"]) == 1
    assert "32768" in output.getvalue()
