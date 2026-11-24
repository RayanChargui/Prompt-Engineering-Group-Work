"""Two execution backends with identical behaviour: local (your PC) and the MedTech sandbox (Docker)."""
import json
import os
import subprocess
from pathlib import Path

from . import ops

BOX = "medtech-sandbox"
IMAGE = "python:3.12-alpine"
SHARED = os.path.expanduser("~/medtech-files")  # visible from Windows, mounted as /workspace
# Reads "<json>\n<ops.py source>" from stdin, runs ops.run(json) and prints the JSON result.
BOOT = ("import sys,json;p,s=sys.stdin.read().split('\\n',1);g={'__name__':'medtech_ops'};"
        "exec(compile(s,'ops','exec'),g);print(json.dumps(g['run'](json.loads(p))))")


class BackendError(Exception):
    pass


class _Base:
    name, root = "", ""

    def __init__(self):
        self.cwd = self.root

    def where(self):
        r = os.path.relpath(self.cwd, self.root).replace(os.sep, "/")
        return "workspace" if r == "." else f"workspace/{r}"

    def run(self, tool, args):
        res = self._exec({"root": self.root, "cwd": self.cwd, "tool": tool, "args": args})
        self.cwd = res.get("cwd", self.cwd)
        return res["ok"], res["output"]


class LocalBackend(_Base):
    name = "local"

    def __init__(self, workspace):
        self.root = os.path.realpath(os.path.expanduser(workspace))
        os.makedirs(self.root, exist_ok=True)
        super().__init__()

    def _exec(self, req):
        return ops.run(req)


class SandboxBackend(_Base):
    """Hardened container: no network, no capabilities, read-only rootfs, resource limits,
    only the shared folder /workspace is writable."""
    name, root = "sandbox", "/workspace"

    def __init__(self):
        super().__init__()
        self.src = Path(ops.__file__).read_text(encoding="utf-8")

    def ensure(self):
        os.makedirs(SHARED, exist_ok=True)
        def sh(*a):
            try:
                return subprocess.run(a, capture_output=True, text=True, timeout=300)
            except FileNotFoundError:
                raise BackendError("Docker is not installed or not in PATH.")
            except subprocess.TimeoutExpired:
                raise BackendError("Docker timed out (is it running / pulling the image?).")
        if sh("docker", "start", BOX).returncode == 0:
            return
        r = sh("docker", "run", "-d", "--name", BOX, "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
               "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
               "-v", f"{SHARED}:/workspace", "-w", "/workspace", IMAGE, "sleep", "infinity")
        if r.returncode:
            raise BackendError(f"Could not start the sandbox: {r.stderr.strip()}")

    def reset(self):
        subprocess.run(["docker", "rm", "-f", BOX], capture_output=True)

    def _exec(self, req):
        try:
            r = subprocess.run(["docker", "exec", "-i", BOX, "python", "-c", BOOT],
                               input=json.dumps(req) + "\n" + self.src,
                               capture_output=True, text=True, timeout=60)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError) as e:
            return {"ok": False, "output": f"ERROR: sandbox call failed ({type(e).__name__})",
                    "cwd": req["cwd"]}
