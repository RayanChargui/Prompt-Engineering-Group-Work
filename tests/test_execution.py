"""Execution evidence, transport diagnostics, jail rules and optional Docker integration."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from medtech import ops
from medtech.backends import BOOT, IMAGE, BackendError, LocalBackend, SandboxBackend


def test_local_create_verify_and_cwd(tmp_path):
    backend = LocalBackend(str(tmp_path))
    folder = backend.run("mkdir", {"path": "test_dir"})
    assert folder["success"] and folder["exit_code"] == 0
    assert folder["verified"][0]["path"] == str(tmp_path / "test_dir")
    assert backend.run("cd", {"path": "test_dir"})["cwd"] == str(tmp_path / "test_dir")
    result = backend.run("write_file", {"path": "hello.txt", "content": "hi"})
    assert result["success"] and result["stderr"] == "" and result["duration"] >= 0
    assert (tmp_path / "test_dir/hello.txt").read_text() == "hi"
    assert result["verified"][0]["kind"] == "file"
    assert result["verified"][0]["path"] in result["stdout"]
    assert backend.where() == str(tmp_path / "test_dir")
    assert backend.run("pwd", {})["stdout"] == backend.where()
    assert backend.run("write_file", {"path": "hello.txt", "content": "!", "mode": "append"})["success"]
    assert (tmp_path / "test_dir/hello.txt").read_text() == "hi!"


def test_copy_move_verification(tmp_path):
    backend = LocalBackend(str(tmp_path))
    backend.run("write_file", {"path": "a", "content": "hi"})
    copied = backend.run("copy", {"src": "a", "dst": "b"})
    assert copied["success"] and copied["verified"][0]["path"] == str(tmp_path / "b")
    moved = backend.run("move", {"src": "b", "dst": "c"})
    assert moved["success"] and moved["verified"][0]["path"] == str(tmp_path / "c")
    assert {"path": str(tmp_path / "b"), "kind": "file", "exists": False} in moved["verified"]
    assert not (tmp_path / "b").exists()
    assert (tmp_path / "c").read_text() == "hi"


def test_local_failures_not_success(tmp_path, monkeypatch):
    backend = LocalBackend(str(tmp_path))
    for path in ("../../etc/passwd", "/etc/passwd", ".env"):
        result = backend.run("write_file", {"path": path, "content": "no"})
        assert not result["success"] and result["exit_code"] != 0
        assert "DENIED" in result["stderr"] and not result["verified"]
    (tmp_path / "link").symlink_to("/etc", target_is_directory=True)
    assert not backend.run("ls", {"path": "link"})["success"]
    backend.run("write_file", {"path": "existing", "content": "first"})
    assert not backend.run("write_file", {"path": "existing", "content": "second"})["success"]
    assert (tmp_path / "existing").read_text() == "first"
    original = ops.open if hasattr(ops, "open") else open

    def denied(path, *args, **kwargs):
        if str(path).endswith("readonly"):
            raise PermissionError("read-only filesystem")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ops, "open", denied, raising=False)
    result = backend.run("write_file", {"path": "readonly", "content": "no"})
    assert not result["success"] and "read-only filesystem" in result["stderr"]


def test_system_process_keeps_nonzero_exit_and_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 12, "out", "failure"))
    result = LocalBackend(str(tmp_path)).run("sysinfo", {"kind": "network"})
    assert result["success"] is False and result["exit_code"] == 12
    assert result["stdout"] == "out" and result["stderr"] == "failure"


def test_write_readback_failure_is_not_verified(tmp_path, monkeypatch):
    original = open

    class BadRead:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def seek(self, *args): pass
        def read(self): return b"wrong"

    def wrong(path, mode="r", *args, **kwargs):
        return BadRead() if mode == "rb" else original(path, mode, *args, **kwargs)

    monkeypatch.setattr(ops, "open", wrong, raising=False)
    result = LocalBackend(str(tmp_path)).run("write_file", {"path": "hello", "content": "hi"})
    assert not result["success"] and "verification failed" in result["stderr"]
    assert not result["verified"]


def test_docker_boot_protocol_without_docker(tmp_path):
    request = {"root": str(tmp_path), "cwd": str(tmp_path), "tool": "mkdir", "args": {"path": "actual"}}
    result = subprocess.run([sys.executable, "-c", BOOT],
                            input=json.dumps(request) + "\n" + Path(ops.__file__).read_text(),
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["success"]
    assert (tmp_path / "actual").is_dir()


def test_docker_nonzero_status_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 126, "partial", "permission denied"))
    result = SandboxBackend(str(tmp_path)).run("mkdir", {"path": "x"})
    assert not result["success"] and result["exit_code"] == 126
    assert result["stdout"] == "partial" and result["stderr"] == "permission denied"


def test_docker_invalid_payload_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, '{"ok":true}', "warn"))
    result = SandboxBackend(str(tmp_path)).run("mkdir", {"path": "x"})
    assert not result["success"] and "Invalid sandbox response" in result["stderr"]
    assert "warn" in result["stderr"]


def inspect_info(backend):
    return {"Config": {"WorkingDir": "/workspace", "Image": IMAGE, "User": backend.user},
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
                           "SecurityOpt": ["no-new-privileges"], "Memory": 512 * 1024 * 1024,
                           "NanoCpus": 1_000_000_000, "PidsLimit": 128},
            "Mounts": [{"Type": "bind", "Source": backend.host_root, "Destination": "/workspace", "RW": True}],
            "State": {"Running": True}}


def test_docker_commands_workdir_uid_mount_and_readiness(tmp_path, monkeypatch):
    backend = SandboxBackend(str(tmp_path))
    commands = []

    def docker(cmd, **kwargs):
        commands.append((cmd, kwargs))
        if cmd[1] == "inspect":
            return subprocess.CompletedProcess(cmd, 1, "", "No such container")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, "container-id", "")
        request, source = kwargs["input"].split("\n", 1)
        request = json.loads(request)
        assert source == Path(ops.__file__).read_text()
        assert cmd[cmd.index("-w") + 1] == request["cwd"]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"success": True, "exit_code": 0,
                                           "stdout": "/workspace", "stderr": "", "cwd": "/workspace",
                                           "duration": 0, "verified": []}), "")

    monkeypatch.setattr(subprocess, "run", docker)
    backend.ensure()
    run = commands[1][0]
    assert run[run.index("-v") + 1] == f"{tmp_path}:/workspace"
    assert run[run.index("--user") + 1] == backend.user
    assert run[run.index("-w") + 1] == "/workspace"
    assert "--rm" not in run
    assert backend.box in commands[-1][0]


def test_docker_reuses_only_matching_persistent_container(tmp_path, monkeypatch):
    backend = SandboxBackend(str(tmp_path))
    info = inspect_info(backend)
    calls = []

    def docker(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps([info]), "")

    monkeypatch.setattr(backend, "_docker", docker)
    monkeypatch.setattr(backend, "run", lambda *args: {"success": True})
    backend.ensure()
    assert calls == [("inspect", backend.box)]
    info["Mounts"][0]["Source"] = "/wrong"
    with pytest.raises(BackendError, match="incompatible workspace"):
        backend.ensure()
    assert all(call[0] == "inspect" for call in calls)


def docker_available():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.docker
@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_real_docker_creation_persistence_and_host_mapping(tmp_path):
    backend = SandboxBackend(str(tmp_path), name="medtech-test-" + uuid.uuid4().hex[:12])
    try:
        backend.ensure()
        assert backend.run("mkdir", {"path": "test_dir"})["success"]
        assert backend.run("cd", {"path": "test_dir"})["success"]
        result = backend.run("write_file", {"path": "hello.txt", "content": "hi"})
        assert result["success"] and result["verified"][0]["path"] == "/workspace/test_dir/hello.txt"
        assert backend.run("read_file", {"path": "hello.txt"})["stdout"] == "hi"
        assert (tmp_path / "test_dir/hello.txt").read_text() == "hi"
        stopped = backend._docker("stop", backend.box)
        assert stopped.returncode == 0
        fresh = SandboxBackend(str(tmp_path), name=backend.box)
        fresh.ensure()
        assert fresh.run("read_file", {"path": "test_dir/hello.txt"})["stdout"] == "hi"
    finally:
        backend._docker("rm", "-f", backend.box)


@pytest.mark.docker
@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_real_docker_agent_scripted_llm_acceptance(tmp_path):
    """Exercise the actual agent loop, Docker transport, disk audit, and host mount."""
    from medtech.agent import Agent, Config
    from medtech.store import Store

    class UI:
        def __init__(self): self.answers = []
        def say(self, text): self.answers.append(text)
        def warn(self, text): self.answers.append(text)
        def action(self, *args): pass
        def confirm(self, *args): return True
        def result(self, *args): pass

    def response(say="", tool=None, **args):
        return json.dumps({"say": say, "action": {"tool": tool, "args": args} if tool else None})

    steps = iter([response(tool="mkdir", path="test_dir"),
                  response(tool="write_file", path="test_dir/hello.txt", content="hi"),
                  response("Created directory /workspace/test_dir and file /workspace/test_dir/hello.txt.")])
    contexts = []

    def llm(messages):
        if messages[0]["content"].startswith("Extract only"):
            return '{"facts":[]}'
        contexts.append(messages)
        return next(steps)

    backend = SandboxBackend(str(tmp_path / "host-files"), name="medtech-agent-test-" + uuid.uuid4().hex[:12])
    store = Store(str(tmp_path / "agent.db"))
    try:
        backend.ensure()
        sid = store.new_session("sandbox", backend.host_root)
        ui = UI()
        agent = Agent(Config(), backend, store, sid, ui, auto=True, llm=llm)
        agent.turn("Create folder test_dir with hello.txt containing hi")
        assert (Path(backend.host_root) / "test_dir/hello.txt").read_text() == "hi"
        readback = backend.run("read_file", {"path": "/workspace/test_dir/hello.txt"})
        assert readback["success"] and readback["stdout"] == "hi"
        tasks = store.tasks(sid)
        assert [task["status"] for task in tasks] == ["done", "done"]
        results = [json.loads(task["output"]) for task in tasks]
        assert [result["verified"][0]["path"] for result in results] == [
            "/workspace/test_dir", "/workspace/test_dir/hello.txt"]
        assert len(contexts) == 3
        assert any('"success": true' in message["content"] for message in contexts[-1])
        assert ui.answers == ["Created directory /workspace/test_dir and file /workspace/test_dir/hello.txt."]
    finally:
        store.close()
        backend._docker("rm", "-f", backend.box)


def test_delete_records_verified_absence(tmp_path):
    backend = LocalBackend(str(tmp_path))
    backend.run("mkdir", {"path": "test_dir"})
    backend.run("write_file", {"path": "test_dir/hello.txt", "content": "hi"})
    result = backend.run("delete", {"path": "test_dir", "recursive": True})
    assert result["success"]
    assert result["verified"] == [{"path": str(tmp_path / "test_dir"), "kind": "directory", "exists": False}]
    assert not (tmp_path / "test_dir").exists()
