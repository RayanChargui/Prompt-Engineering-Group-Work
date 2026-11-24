"""Local operations and a persistent, bind-mounted Docker workspace."""
import hashlib
import json
import os
import posixpath
import subprocess
import time
from pathlib import Path

from . import ops

BOX = "medtech-sandbox"
IMAGE = "python:3.12-alpine"
SHARED = os.path.expanduser("~/medtech-files")
BOOT = ("import sys,json;p,s=sys.stdin.read().split('\\n',1);g={'__name__':'medtech_ops'};"
        "exec(compile(s,'ops','exec'),g);print(json.dumps(g['run'](json.loads(p))))")


class BackendError(Exception):
    """An execution environment could not be prepared."""


def failure(cwd: str, error: str, code: int = 1, stdout: str = "", duration: float = 0) -> dict:
    """Build the same result envelope for transport and operation failures."""
    return {"success": False, "exit_code": code, "stdout": stdout, "stderr": error,
            "cwd": cwd, "duration": duration, "verified": []}


class _Base:
    name, root = "", ""

    def __init__(self):
        self.cwd = self.root

    def where(self) -> str:
        """Return the actual absolute path in the active execution environment."""
        return self.cwd

    def run(self, tool: str, args: dict) -> dict:
        """Execute a typed operation and retain its working directory."""
        res = self._exec({"root": self.root, "cwd": self.cwd, "tool": tool, "args": args,
                          "permissions_mode": self.permissions_mode,
                          "allowed_roots": self.allowed_roots, "read_only_roots": self.read_only_roots})
        if res["success"]:
            self.cwd = res["cwd"]
        return res


class LocalBackend(_Base):
    name = "local"

    def __init__(self, workspace: str, permissions_mode: str = "full", confirm_dangerous: bool = True):
        self.confirm_dangerous = confirm_dangerous
        self.set_permissions(permissions_mode, root=workspace)
        super().__init__()

    def set_permissions(self, mode: str, root: str = None) -> None:
        """Switch between native host access and an explicitly restricted workspace."""
        if mode not in {"full", "workspace"}:
            raise ValueError("permissions mode must be full or workspace")
        previous_root = getattr(self, "root", None)
        if root is not None:
            self.root = os.path.realpath(os.path.expanduser(root))
            self.host_root = self.root
            os.makedirs(self.root, exist_ok=True)
        self.permissions_mode = mode
        self.allowed_roots, self.read_only_roots = [self.root], []
        if root is not None and previous_root != self.root:
            self.cwd = self.root
        elif mode == "workspace" and not ops._inside(getattr(self, "cwd", self.root), self.root):
            self.cwd = self.root

    def _exec(self, req: dict) -> dict:
        return ops.run(req)


class SandboxBackend(_Base):
    """One hardened container per host workspace, bind-mounted at /workspace.

    A caller may supply a unique name for isolated tests. Existing containers are
    reused only when their mount, user, image and isolation settings match.
    """
    name, root = "sandbox", "/workspace"

    def __init__(self, workspace: str = None, name: str = None, extra_mounts: list = None):
        self.host_root = os.path.realpath(os.path.expanduser(workspace or SHARED))
        self.extra_mounts = self._validate_mounts(extra_mounts if extra_mounts is not None else [])
        identity = self.host_root + (json.dumps(self.extra_mounts, sort_keys=True) if self.extra_mounts else "")
        self.box = name or BOX + "-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        self.permissions_mode, self.confirm_dangerous = "workspace", True
        self.allowed_roots = [self.root] + [mount["target"] for mount in self.extra_mounts]
        self.read_only_roots = [mount["target"] for mount in self.extra_mounts if mount["read_only"]]
        self.user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        super().__init__()
        self.src = Path(ops.__file__).read_text(encoding="utf-8")

    @staticmethod
    def _validate_mounts(mounts: list) -> list:
        """Accept only explicit existing host directories with distinct container roots."""
        if not isinstance(mounts, list):
            raise ValueError("sandbox extra_mounts must be a list")
        normalized, targets = [], ["/workspace"]
        for mount in mounts:
            if (not isinstance(mount, dict) or set(mount) - {"source", "target", "read_only"}
                    or not isinstance(mount.get("source"), str) or not mount["source"].strip()
                    or not isinstance(mount.get("target"), str)
                    or type(mount.get("read_only", False)) is not bool):
                raise ValueError("each sandbox mount requires source, target, and optional read_only boolean")
            source = os.path.realpath(os.path.expanduser(mount["source"]))
            target = mount["target"]
            if (not target.startswith("/") or target.startswith("//") or target == "/" or "\x00" in target
                    or target != posixpath.normpath(target) or ":" in target or "\\" in target
                    or target == "/tmp" or target.startswith("/tmp/")):
                raise ValueError("mount target must be an absolute normalized container directory outside /tmp")
            if not os.path.isdir(source):
                raise ValueError(f"sandbox mount source is not an existing directory: {source}")
            if any(target == old or target.startswith(old + "/") or old.startswith(target + "/") for old in targets):
                raise ValueError(f"sandbox mount target overlaps another mount: {target}")
            targets.append(target)
            normalized.append({"source": source, "target": target, "read_only": mount.get("read_only", False)})
        return sorted(normalized, key=lambda mount: mount["target"])

    def _docker(self, *args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=300)
        except FileNotFoundError as e:
            raise BackendError("Docker is not installed or not in PATH.") from e
        except subprocess.TimeoutExpired as e:
            raise BackendError("Docker timed out; check the daemon and image availability.") from e

    def _matches(self, info: dict) -> bool:
        """Do not silently reuse a stale container pointing at another directory."""
        config, host = info.get("Config", {}), info.get("HostConfig", {})
        mounts = {m.get("Destination"): (m.get("Source"), m.get("RW"))
                  for m in info.get("Mounts", []) if m.get("Type") == "bind"}
        expected = {self.root: (self.host_root, True)}
        expected.update({mount["target"]: (mount["source"], not mount["read_only"]) for mount in self.extra_mounts})
        foreign = [m for m in info.get("Mounts", []) if m.get("Type") not in {"bind", "tmpfs"}]
        return (mounts == expected and not foreign
                and config.get("WorkingDir") == self.root and config.get("Image") == IMAGE
                and config.get("User") == self.user and host.get("NetworkMode") == "none"
                and host.get("ReadonlyRootfs") is True and "ALL" in host.get("CapDrop", [])
                and "no-new-privileges" in host.get("SecurityOpt", [])
                and host.get("Memory") == 512 * 1024 * 1024
                and host.get("NanoCpus") == 1_000_000_000 and host.get("PidsLimit") == 128)

    def ensure(self) -> None:
        """Start or validate the persistent sandbox without deleting user containers."""
        os.makedirs(self.host_root, exist_ok=True)
        inspected = self._docker("inspect", self.box)
        if inspected.returncode == 0:
            try:
                info = json.loads(inspected.stdout)[0]
                matches = self._matches(info)
            except (ValueError, IndexError, TypeError, AttributeError) as e:
                raise BackendError("Docker returned invalid container inspection data.") from e
            if not matches:
                raise BackendError(f"Container {self.box} has incompatible workspace or security settings. "
                                   "Inspect it and choose a different container name; it was not changed.")
            if not info.get("State", {}).get("Running"):
                result = self._docker("start", self.box)
                if result.returncode:
                    raise BackendError(f"Could not start sandbox: {result.stderr.strip()}")
        else:
            # A missing container is normal. Daemon/auth/image failures remain visible
            # in docker run's status and diagnostics, rather than being swallowed.
            arguments = ["run", "-d", "--name", self.box, "--network", "none", "--cap-drop", "ALL",
                         "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
                         "--memory", "512m", "--cpus", "1", "--pids-limit", "128", "--user", self.user,
                         "-v", f"{self.host_root}:/workspace", "-w", "/workspace"]
            for mount in self.extra_mounts:
                mode = "ro" if mount["read_only"] else "rw"
                arguments += ["-v", f"{mount['source']}:{mount['target']}:{mode}"]
            result = self._docker(*arguments, IMAGE, "sleep", "infinity")
            if result.returncode:
                raise BackendError(f"Could not create sandbox: {result.stderr.strip()}")
        ready = self.run("pwd", {})
        if not ready["success"]:
            raise BackendError(f"Sandbox readiness failed: {ready['stderr']}")

    def reset(self) -> None:
        """Remove only this backend's container; bind-mounted files are retained."""
        result = self._docker("rm", "-f", self.box)
        if result.returncode:
            raise BackendError(f"Could not remove sandbox: {result.stderr.strip()}")

    def _exec(self, req: dict) -> dict:
        started = time.monotonic()
        try:
            result = subprocess.run(["docker", "exec", "-i", "-w", req["cwd"], self.box, "python", "-c", BOOT],
                                    input=json.dumps(req) + "\n" + self.src,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                return failure(req["cwd"], result.stderr or "Docker exec failed", result.returncode,
                               result.stdout, time.monotonic() - started)
            try:
                res = json.loads(result.stdout)
                valid = (isinstance(res, dict) and type(res.get("success")) is bool
                         and type(res.get("exit_code")) is int
                         and res["success"] == (res["exit_code"] == 0)
                         and all(isinstance(res.get(k), str) for k in ("stdout", "stderr", "cwd"))
                         and isinstance(res.get("duration"), (int, float))
                         and isinstance(res.get("verified"), list))
                if not valid:
                    raise ValueError("invalid result schema")
            except (ValueError, TypeError) as e:
                return failure(req["cwd"], f"Invalid sandbox response: {e}; {result.stderr}",
                               stdout=result.stdout, duration=time.monotonic() - started)
            if result.stderr:
                res["stderr"] += result.stderr
            res["duration"] = time.monotonic() - started
            return res
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return failure(req["cwd"], f"Sandbox call failed: {e}; {stderr}", stdout=stdout,
                           duration=time.monotonic() - started)
