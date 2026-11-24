"""Local operations and a persistent, bind-mounted Docker workspace."""
import hashlib
import json
import os
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
        res = self._exec({"root": self.root, "cwd": self.cwd, "tool": tool, "args": args})
        if res["success"]:
            self.cwd = res["cwd"]
        return res


class LocalBackend(_Base):
    name = "local"

    def __init__(self, workspace: str):
        self.root = os.path.realpath(os.path.expanduser(workspace))
        self.host_root = self.root
        os.makedirs(self.root, exist_ok=True)
        super().__init__()

    def _exec(self, req: dict) -> dict:
        return ops.run(req)


class SandboxBackend(_Base):
    """One hardened container per host workspace, bind-mounted at /workspace.

    A caller may supply a unique name for isolated tests. Existing containers are
    reused only when their mount, user, image and isolation settings match.
    """
    name, root = "sandbox", "/workspace"

    def __init__(self, workspace: str = None, name: str = None):
        self.host_root = os.path.realpath(os.path.expanduser(workspace or SHARED))
        self.box = name or BOX + "-" + hashlib.sha256(self.host_root.encode()).hexdigest()[:12]
        self.user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        super().__init__()
        self.src = Path(ops.__file__).read_text(encoding="utf-8")

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
        mounts = [m for m in info.get("Mounts", []) if m.get("Destination") == self.root]
        return (len(mounts) == 1 and mounts[0].get("Type") == "bind"
                and mounts[0].get("Source") == self.host_root and mounts[0].get("RW") is True
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
            result = self._docker("run", "-d", "--name", self.box, "--network", "none", "--cap-drop", "ALL",
                                  "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
                                  "--memory", "512m", "--cpus", "1", "--pids-limit", "128", "--user", self.user,
                                  "-v", f"{self.host_root}:/workspace", "-w", "/workspace",
                                  IMAGE, "sleep", "infinity")
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
