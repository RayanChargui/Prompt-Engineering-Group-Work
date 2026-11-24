"""Host permissions and explicit sandbox roots; destructive proposals are mocked."""
import subprocess
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from medtech import ops, policy
from medtech.backends import LocalBackend, SandboxBackend


def test_local_default_full_accepts_absolute_relative_and_home_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    starting = tmp_path / "starting"
    backend = LocalBackend(str(starting))
    assert backend.permissions_mode == "full"
    outside = tmp_path / "outside"
    outside.mkdir()
    assert backend.run("cd", {"path": str(outside)})["success"]
    relative = backend.run("write_file", {"path": "relative.txt", "content": "hi"})
    assert relative["success"] and relative["cwd"] == str(outside)
    assert relative["verified"][0]["path"] == str(outside / "relative.txt")
    assert backend.run("write_file", {"path": "~/hello.txt", "content": "home"})["success"]
    assert (home / "hello.txt").read_text() == "home"
    assert backend.run("read_file", {"path": str(outside / "relative.txt")})["stdout"] == "hi"
    assert backend.where() == str(outside)


def test_full_mode_does_not_filter_protected_names_or_symlinks(tmp_path):
    backend = LocalBackend(str(tmp_path / "starting"))
    target = tmp_path / "external"
    target.mkdir()
    (Path(backend.root) / "link").symlink_to(target, target_is_directory=True)
    assert backend.run("write_file", {"path": "link/.env", "content": "not a secret"})["success"]
    assert backend.run("write_file", {"path": str(target / ".ssh/id_rsa"), "content": "fixture"})["success"]
    assert backend.run("read_file", {"path": "link/.env"})["stdout"] == "not a secret"
    found = backend.run("search", {"path": str(target), "query": "id_rsa"})
    assert found["success"] and str(target / ".ssh/id_rsa") in found["stdout"]


def test_switch_to_workspace_restores_jail_and_safe_cwd(tmp_path):
    starting, outside = tmp_path / "starting", tmp_path / "outside"
    outside.mkdir()
    backend = LocalBackend(str(starting))
    assert backend.run("cd", {"path": str(outside)})["success"]
    backend.set_permissions("workspace")
    assert backend.cwd == str(starting)
    for path in (str(outside / "new.txt"), "../escape", "~/.config", ".env"):
        result = backend.run("write_file", {"path": path, "content": "no"})
        assert not result["success"] and "DENIED" in result["stderr"]
    backend.set_permissions("full")
    assert backend.run("write_file", {"path": str(outside / "new.txt"), "content": "yes"})["success"]
    second = tmp_path / "second"
    backend.set_permissions("workspace", root=str(second))
    assert backend.root == backend.host_root == backend.cwd == str(second)
    with pytest.raises(ValueError, match="mode"):
        backend.set_permissions("anything")


def test_full_mode_reports_actual_operating_system_errors(tmp_path, monkeypatch):
    backend = LocalBackend(str(tmp_path))
    denied = PermissionError("OS denied this fixture")
    monkeypatch.setattr(ops, "open", Mock(side_effect=denied), raising=False)
    result = backend.run("write_file", {"path": str(tmp_path / "file"), "content": "hi"})
    assert not result["success"] and result["exit_code"] != 0
    assert "PermissionError" in result["stderr"] and "OS denied this fixture" in result["stderr"]
    assert "DENIED:" not in result["stderr"]


@pytest.mark.parametrize("tool,args", [
    ("read_file", {"path": "/anywhere"}), ("mkdir", {"path": "/anywhere"}),
    ("write_file", {"path": "/anywhere", "content": "hi", "mode": "overwrite"}),
    ("delete", {"path": "/tmp/ordinary-fixture", "recursive": True}),
])
def test_full_local_ordinary_tools_need_no_confirmation(tool, args):
    decision = policy.check(tool, args, local=True, permissions_mode="full", auto=False, tainted=True,
                            cwd="/tmp", root="/tmp", home="/home/test-user")
    assert decision.level == "auto"


@pytest.mark.parametrize("path,cwd,home", [
    ("/", "/tmp", "/home/test-user"),
    ("~", "/tmp", "/home/test-user"),
    ("~/.", "/tmp", "/home/test-user"),
    ("/home/test-user", "/tmp", "/home/test-user"),
    ("/home", "/tmp", "/home/test-user"),
    ("..", "/home/test-user/project", "/home/test-user"),
    ("C:\\", "C:\\work", "C:\\Users\\test-user"),
    ("C:\\Users\\test-user", "C:\\work", "C:\\Users\\test-user"),
    ("~", "C:\\work", "C:\\Users\\test-user"),
    ("\\\\server\\share\\", "\\\\server\\share\\work", "C:\\Users\\test-user"),
])
def test_catastrophic_recursive_delete_classification_only(path, cwd, home):
    # Never execute these paths: only classify and mock the destructive callback.
    args = {"path": path, "recursive": True}
    destructive = Mock()
    assert policy.is_catastrophic("delete", args, cwd=cwd, root=cwd, home=home)
    decision = policy.check("delete", args, local=True, permissions_mode="full", cwd=cwd, root=cwd,
                            home=home, confirm_dangerous=True)
    if decision.level == "auto":
        destructive()
    assert decision.level == "confirm"
    destructive.assert_not_called()
    assert policy.check("delete", args, local=True, permissions_mode="full", cwd=cwd, home=home,
                        confirm_dangerous=False).level == "auto"


def test_nonrecursive_and_ordinary_delete_are_not_catastrophic():
    assert not policy.is_catastrophic("delete", {"path": "/", "recursive": False})
    assert not policy.is_catastrophic("delete", {"path": "/tmp/fixture", "recursive": True}, home="/home/test-user")
    assert not policy.is_catastrophic("write_file", {"path": "/", "recursive": True})
    assert not policy.is_catastrophic("delete", {"path": "C:\\work\\fixture", "recursive": True},
                                      cwd="C:\\work", home="C:\\Users\\test-user")


@pytest.mark.parametrize("tool,args", [("shell", {"cmd": "rm -rf /"}), ("rm", {"path": "/"}),
                                       ("mkfs", {}), ("dd", {"of": "/dev/example"})])
def test_command_like_unsupported_calls_remain_blocked(tool, args):
    assert policy.check(tool, args, local=True, permissions_mode="full", confirm_dangerous=False).level == "block"


def test_local_scope_does_not_add_prompts_and_sandbox_keeps_confirmation_rules():
    assert policy.check("delete", {"path": "x"}, local=True, permissions_mode="workspace").level == "auto"
    assert policy.check("delete", {"path": "x"}, local=False, permissions_mode="workspace", auto=True).level == "confirm"
    assert policy.check("write_file", {"path": "x", "content": "hi"}, auto=True, tainted=True).level == "confirm"


def test_explicit_extra_roots_allow_only_configured_container_paths(tmp_path):
    main, mapped, readonly = (tmp_path / name for name in ("main", "mapped", "readonly"))
    for path in (main, mapped, readonly): path.mkdir()
    (readonly / "reference.txt").write_text("read only")
    request = {"root": str(main), "cwd": str(main), "permissions_mode": "workspace",
               "allowed_roots": [str(main), str(mapped), str(readonly)], "read_only_roots": [str(readonly)]}
    def run(tool, **args):
        return ops.run(dict(request, tool=tool, args=args))
    assert run("write_file", path=str(mapped / "hello.txt"), content="hi")["success"]
    assert run("read_file", path=str(readonly / "reference.txt"))["stdout"] == "read only"
    result = run("write_file", path=str(readonly / "new.txt"), content="no")
    assert not result["success"] and "read-only" in result["stderr"]
    assert not run("read_file", path="/etc/passwd")["success"]
    (mapped / "escape").symlink_to(tmp_path, target_is_directory=True)
    assert not run("mkdir", path=str(mapped / "escape/unmapped"))["success"]


def test_sandbox_mount_configuration_changes_identity_and_is_validated(tmp_path):
    main, data, reference = (tmp_path / name for name in ("main", "data", "reference"))
    for path in (main, data, reference): path.mkdir()
    mounts = [{"source": str(data), "target": "/data"},
              {"source": str(reference), "target": "/reference", "read_only": True}]
    plain = SandboxBackend(str(main))
    configured = SandboxBackend(str(main), extra_mounts=mounts)
    assert configured.permissions_mode == "workspace"
    assert configured.allowed_roots == ["/workspace", "/data", "/reference"]
    assert configured.read_only_roots == ["/reference"]
    assert plain.box != configured.box
    assert configured.box == SandboxBackend(str(main), extra_mounts=list(reversed(mounts))).box
    changed = [dict(mounts[0]), dict(mounts[1], read_only=False)]
    assert configured.box != SandboxBackend(str(main), extra_mounts=changed).box
    for target in ("/", "/workspace", "/workspace/nested", "relative", "/data/../other", "/tmp/reference"):
        with pytest.raises(ValueError):
            SandboxBackend(str(main), extra_mounts=[{"source": str(data), "target": target}])
    with pytest.raises(ValueError, match="overlap"):
        SandboxBackend(str(main), extra_mounts=[{"source": str(data), "target": "/data"},
                                                {"source": str(reference), "target": "/data/nested"}])
    with pytest.raises(ValueError, match="existing directory"):
        SandboxBackend(str(main), extra_mounts=[{"source": str(tmp_path / "absent"), "target": "/data"}])


def test_docker_extra_mount_commands_and_existing_mount_checks(tmp_path, monkeypatch):
    source = tmp_path / "reference"
    source.mkdir()
    backend = SandboxBackend(str(tmp_path / "main"), extra_mounts=[
        {"source": str(source), "target": "/reference", "read_only": True}])
    commands = []
    def docker(*args):
        commands.append(args)
        return subprocess.CompletedProcess(args, 1 if args[0] == "inspect" else 0, "", "missing")
    monkeypatch.setattr(backend, "_docker", docker)
    monkeypatch.setattr(backend, "run", lambda *args: {"success": True})
    backend.ensure()
    assert f"{source}:/reference:ro" in commands[1]
    info = {"Config": {"WorkingDir": "/workspace", "Image": "python:3.12-alpine", "User": backend.user},
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
                           "SecurityOpt": ["no-new-privileges"], "Memory": 512 * 1024 * 1024,
                           "NanoCpus": 1_000_000_000, "PidsLimit": 128},
            "Mounts": [{"Type": "bind", "Source": backend.host_root, "Destination": "/workspace", "RW": True},
                       {"Type": "bind", "Source": str(source), "Destination": "/reference", "RW": False}],
            "State": {"Running": True}}
    assert backend._matches(info)
    info["Mounts"][1]["RW"] = True
    assert not backend._matches(info)


def docker_available():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.docker
@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_real_sandbox_explicit_mounts_and_read_only_enforcement(tmp_path):
    data, reference = tmp_path / "data", tmp_path / "reference"
    data.mkdir()
    reference.mkdir()
    (reference / "readme.txt").write_text("reference")
    backend = SandboxBackend(str(tmp_path / "main"), name="medtech-mount-test-" + uuid.uuid4().hex[:12], extra_mounts=[
        {"source": str(data), "target": "/data"},
        {"source": str(reference), "target": "/reference", "read_only": True}])
    try:
        backend.ensure()
        assert backend.run("write_file", {"path": "/data/hello.txt", "content": "hi"})["success"]
        assert (data / "hello.txt").read_text() == "hi"
        assert backend.run("read_file", {"path": "/reference/readme.txt"})["stdout"] == "reference"
        denied = backend.run("write_file", {"path": "/reference/readme.txt", "content": "no", "mode": "overwrite"})
        assert not denied["success"] and "read-only" in denied["stderr"]
        assert not backend.run("read_file", {"path": "/etc/passwd"})["success"]
        assert backend.run("cd", {"path": "/data"})["success"]
        assert backend.run("read_file", {"path": "hello.txt"})["stdout"] == "hi"
        fresh = SandboxBackend(str(tmp_path / "main"), name=backend.box, extra_mounts=backend.extra_mounts)
        fresh.ensure()
        assert fresh.run("read_file", {"path": "/data/hello.txt"})["stdout"] == "hi"
    finally:
        backend._docker("rm", "-f", backend.box)


@pytest.mark.parametrize("tool,arguments", [
    ("read_file", {"path": "denied"}), ("ls", {"path": "denied"}), ("cd", {"path": "denied"}),
    ("move", {"src": "denied", "dst": "destination"}),
    ("copy", {"src": "denied", "dst": "destination"}),
    ("delete", {"path": "denied", "recursive": True}),
    ("write_file", {"path": "denied", "content": "hi"}),
    ("search", {"path": "denied", "query": "hello"}),
])
def test_stat_permission_errors_keep_actual_cause(tmp_path, monkeypatch, tool, arguments):
    backend = LocalBackend(str(tmp_path))
    forbidden = str(tmp_path / "denied")
    original_stat, original_lstat = ops.os.stat, ops.os.lstat
    def stat_path(path, *args, **kwargs):
        if str(path) == forbidden:
            raise PermissionError("fixture parent traversal permission denied")
        return original_stat(path, *args, **kwargs)
    def lstat_path(path, *args, **kwargs):
        if str(path) == forbidden:
            raise PermissionError("fixture parent traversal permission denied")
        return original_lstat(path, *args, **kwargs)
    monkeypatch.setattr(ops.os, "stat", stat_path)
    monkeypatch.setattr(ops.os, "lstat", lstat_path)
    result = backend.run(tool, arguments)
    assert not result["success"] and result["exit_code"] != 0
    assert "PermissionError" in result["stderr"]
    assert "fixture parent traversal permission denied" in result["stderr"]
    assert "not a file" not in result["stderr"] and "not found" not in result["stderr"]


def test_search_scandir_permission_error_is_not_no_matches(tmp_path, monkeypatch):
    backend = LocalBackend(str(tmp_path))
    original = ops.os.scandir
    def scandir(path):
        if str(path) == str(tmp_path):
            raise PermissionError("fixture cannot scan requested base")
        return original(path)
    monkeypatch.setattr(ops.os, "scandir", scandir)
    result = backend.run("search", {"path": str(tmp_path), "query": "anything"})
    assert not result["success"]
    assert "PermissionError" in result["stderr"] and "cannot scan requested base" in result["stderr"]
    assert result["stdout"] != "no matches"


def test_raw_device_write_classification_uses_mocked_device_only(tmp_path, monkeypatch):
    import stat
    from types import SimpleNamespace
    mocked_device = str(tmp_path / "mock-device")
    original = policy.os.stat
    def stat_path(path, *args, **kwargs):
        if str(path) == mocked_device:
            return SimpleNamespace(st_mode=stat.S_IFBLK | 0o600)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(policy.os, "stat", stat_path)
    destructive = Mock()
    for mode in ("overwrite", "append"):
        args = {"path": mocked_device, "mode": mode, "content": "never actually written"}
        assert policy.is_catastrophic("write_file", args, cwd=str(tmp_path))
        decision = policy.check("write_file", args, local=True, permissions_mode="full", cwd=str(tmp_path))
        assert decision.level == "confirm"
        if decision.level == "auto": destructive()
    destructive.assert_not_called()
    assert not policy.is_catastrophic("write_file", {"path": mocked_device, "mode": "create"})
    assert policy.is_catastrophic("write_file", {"path": r"\\.\PhysicalDrive0", "mode": "overwrite"},
                                  cwd=r"C:\work", home=r"C:\Users\test-user")


def test_root_or_home_relocation_needs_confirmation_without_executing():
    destructive = Mock()
    for source in ("/", "/home/test-user", "/home", "~"):
        args = {"src": source, "dst": "/tmp/never-used-destination"}
        decision = policy.check("move", args, local=True, permissions_mode="full", cwd="/tmp",
                                home="/home/test-user")
        assert decision.level == "confirm"
        if decision.level == "auto": destructive()
    destructive.assert_not_called()
    assert not policy.is_catastrophic("move", {"src": "/tmp/ordinary", "dst": "/tmp/other"},
                                      cwd="/tmp", home="/home/test-user")


def test_restricted_tree_copy_never_follows_outside_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_text("outside source must not become a regular copied file")
    backend = LocalBackend(str(tmp_path / "workspace"), permissions_mode="workspace")
    tree = Path(backend.root) / "tree"
    tree.mkdir()
    (tree / "external-link").symlink_to(secret)
    result = backend.run("copy", {"src": "tree", "dst": "copied"})
    assert result["success"]
    copied = Path(backend.root) / "copied/external-link"
    assert copied.is_symlink()  # copytree must not materialize the external contents.
    denied = backend.run("read_file", {"path": "copied/external-link"})
    assert not denied["success"] and "outside" in denied["stderr"]
