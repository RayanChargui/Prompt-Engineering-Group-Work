"""MedTech operations: host filesystem or explicitly mapped workspace tools. Stdlib only.

Runs in-process (local mode) or is piped into the Docker sandbox (sandbox mode),
Both execution modes share the same operations and structured verification results.
"""
import datetime
import fnmatch
import getpass
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import time

MAX_READ, MAX_WRITE, MAX_LIST, MAX_HITS = 20_000, 100_000, 200, 50
BLOCKED_FILES = {".env", "id_rsa", "id_ed25519", ".netrc", ".pgpass"}
BLOCKED_DIRS = {".ssh", ".aws", ".gnupg"}


class Denied(Exception):
    pass


class Ctx:
    def __init__(self, root, cwd, permissions_mode="workspace", allowed_roots=None, read_only_roots=None):
        self.root, self.cwd = root, cwd
        self.permissions_mode = permissions_mode
        self.allowed_roots = allowed_roots or [root]
        self.read_only_roots = read_only_roots or []
        self.verified = []
        self.stderr = ""
        self.exit_code = 0


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def resolve(c: Ctx, path: str, write: bool = False) -> str:
    """Resolve native host paths in full mode, or enforce explicit allowed roots."""
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise Denied("invalid path")
    path = path.strip()
    if c.permissions_mode == "full":
        path = os.path.expanduser(path)
    elif path.startswith("~"):
        raise Denied("'~' is not allowed; use paths inside the workspace")
    p = path if os.path.isabs(path) else os.path.join(c.cwd, path)
    p = os.path.realpath(p)
    if c.permissions_mode == "full":
        return p
    roots = [root for root in c.allowed_roots if _inside(p, root)]
    if not roots:
        raise Denied(f"outside the workspace and configured mounts: {path}")
    if write and any(_inside(p, root) for root in c.read_only_roots):
        raise Denied(f"read-only mounted path: {p}")
    parts = {x.lower() for x in os.path.relpath(p, max(roots, key=len)).replace("\\", "/").split("/")}
    if parts & BLOCKED_DIRS or os.path.basename(p).lower() in BLOCKED_FILES:
        raise Denied(f"protected file or folder: {path}")
    return p


def _pwd(c, a):
    return c.cwd, None


def _cd(c, a):
    p = resolve(c, a["path"])
    if not stat.S_ISDIR(os.stat(p).st_mode):
        raise Denied("not a directory")
    return f"now in {p}", p


def _ls(c, a):
    p = resolve(c, a.get("path", "."))
    if not stat.S_ISDIR(os.stat(p).st_mode):
        raise Denied("not a directory")
    rows = []
    for n in sorted(os.listdir(p))[:MAX_LIST]:
        f = os.path.join(p, n)
        entry = os.stat(f)
        rows.append(f"{n}/" if stat.S_ISDIR(entry.st_mode) else f"{n}  ({entry.st_size} B)")
    return "\n".join(rows) or "(empty)", None


def _read(c, a):
    p = resolve(c, a["path"])
    if not stat.S_ISREG(os.stat(p).st_mode):
        raise Denied("not a file")
    with open(p, "rb") as f:
        data = f.read(MAX_READ + 1)
    if b"\x00" in data:
        raise Denied("binary file")
    text = data[:MAX_READ].decode("utf-8", errors="replace")
    return text + ("\n...[truncated]" if len(data) > MAX_READ else ""), None


def _write(c, a):
    p = resolve(c, a["path"], write=True)
    mode, content = a.get("mode", "create"), a["content"]
    if len(content) > MAX_WRITE:
        raise Denied("content too large")
    existing = optional_stat(p)
    if existing is not None and stat.S_ISDIR(existing.st_mode):
        raise Denied("path is a directory")
    if mode == "create" and existing is not None:
        raise Denied("file exists (use mode 'overwrite' or 'append')")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if mode not in {"create", "overwrite", "append"}:
        raise Denied("invalid write mode")
    before = existing.st_size if mode == "append" and existing is not None else 0
    with open(p, {"create": "x", "append": "a", "overwrite": "w"}[mode], encoding="utf-8", newline="") as f:
        f.write(content)
    data = content.encode("utf-8")
    with open(p, "rb") as f:
        f.seek(before)
        if f.read() != data:
            raise OSError("write read-back verification failed")
    verify(c, p)
    return f"verified write of {len(content)} chars to {p} ({mode})", None


def _mkdir(c, a):
    p = resolve(c, a["path"], write=True)
    os.makedirs(p, exist_ok=True)
    verify(c, p)
    return f"verified directory {p}", None


def _dest(c, src, dst_arg):
    dst = resolve(c, dst_arg, write=True)
    destination = optional_stat(dst)
    if destination is not None and stat.S_ISDIR(destination.st_mode):
        dst = resolve(c, os.path.join(dst, os.path.basename(src)), write=True)
        destination = optional_stat(dst)
    if destination is not None:
        raise Denied(f"destination exists: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst


def _move(c, a):
    src = resolve(c, a["src"], write=True)
    if c.permissions_mode != "full" and src in c.allowed_roots:
        raise Denied("cannot move the workspace root")
    source = os.stat(src)
    kind = "directory" if stat.S_ISDIR(source.st_mode) else "file"
    dst = _dest(c, src, a["dst"])
    shutil.move(src, dst)
    if optional_stat(src, follow_symlinks=False) is not None:
        raise OSError("move verification failed: source still exists")
    verify(c, dst)
    c.verified.append({"path": src, "kind": kind, "exists": False})
    return f"verified move {src} -> {dst}", None


def _copy(c, a):
    src = resolve(c, a["src"])
    source = os.stat(src)
    dst = _dest(c, src, a["dst"])
    if stat.S_ISDIR(source.st_mode):
        # Restricted copies preserve links without following them outside allowed
        # roots. Any subsequent tool access still goes through resolve().
        shutil.copytree(src, dst, symlinks=c.permissions_mode != "full")
    else:
        shutil.copy2(src, dst)
    verify(c, dst)
    return f"verified copy {src} -> {dst}", None


def _delete(c, a):
    p = resolve(c, a["path"], write=True)
    if c.permissions_mode != "full" and p in c.allowed_roots:
        raise Denied("cannot delete the workspace root")
    existing = os.lstat(p)
    kind = "directory" if stat.S_ISDIR(existing.st_mode) else "file"
    if stat.S_ISDIR(existing.st_mode):
        (shutil.rmtree if a.get("recursive") else os.rmdir)(p)
    else:
        os.remove(p)
    if optional_stat(p, follow_symlinks=False) is not None:
        raise OSError("delete verification failed: path still exists")
    c.verified.append({"path": p, "kind": kind, "exists": False})
    return f"deleted {p}", None


def _search(c, a):
    q, kind = a["query"].lower(), a.get("kind", "name")
    base, hits = resolve(c, a.get("path", ".")), []
    if not stat.S_ISDIR(os.stat(base).st_mode):
        raise Denied("not a directory")
    def walk_error(error):
        raise error
    for d, dirs, files in os.walk(base, onerror=walk_error):
        if c.permissions_mode != "full":
            dirs[:] = [x for x in dirs if x not in BLOCKED_DIRS]
        for f in files:
            fp = os.path.join(d, f)
            if c.permissions_mode != "full" and (f in BLOCKED_FILES or os.path.islink(fp)):
                continue
            if kind == "name":
                if q in f.lower() or fnmatch.fnmatch(f.lower(), q):
                    hits.append(fp)
            else:
                try:
                    if os.path.getsize(fp) > 1_000_000:
                        continue
                    with open(fp, "r", encoding="utf-8") as fh:
                        for n, line in enumerate(fh, 1):
                            if q in line.lower():
                                hits.append(f"{fp}:{n}: {line.strip()[:120]}")
                                break
                except UnicodeDecodeError:
                    continue
            if len(hits) >= MAX_HITS:
                return "\n".join(hits) + "\n...[more results]", None
    return "\n".join(hits) or "no matches", None


def _sysinfo(c, a):
    k, sysname = a.get("kind", "os"), platform.system()
    if k == "network":  # fixed commands only, never user-supplied arguments
        cmds = {"Windows": [["ipconfig"]], "Darwin": [["ifconfig"]]}.get(
            sysname, [["ip", "addr"], ["ifconfig"]])
        for cmd in cmds:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                c.exit_code, c.stderr = r.returncode, r.stderr
                return r.stdout, None
            except FileNotFoundError:
                continue
        raise FileNotFoundError("no network tool found")
    if k == "os":
        return f"{sysname} {platform.release()} ({platform.machine()}), Python {platform.python_version()}", None
    if k == "whoami":
        return getpass.getuser(), None
    if k == "disk":
        u = shutil.disk_usage(c.cwd if c.permissions_mode == "full" else c.root)
        return f"total {u.total >> 30} GiB, free {u.free >> 30} GiB", None
    if k == "date":
        return datetime.datetime.now().isoformat(timespec="seconds"), None
    raise Denied("unknown kind")


TOOLS = {"pwd": _pwd, "cd": _cd, "ls": _ls, "read_file": _read, "write_file": _write,
         "mkdir": _mkdir, "move": _move, "copy": _copy, "delete": _delete,
         "search": _search, "sysinfo": _sysinfo}


def optional_stat(path: str, follow_symlinks: bool = True):
    """Treat only missing paths as absent; access and traversal failures propagate."""
    try:
        return os.stat(path) if follow_symlinks else os.lstat(path)
    except FileNotFoundError:
        return None


def verify(c: Ctx, path: str) -> None:
    """Record a stat/read-back result in the environment that performed the write."""
    st = os.stat(path)
    record = {"path": path, "kind": "directory" if stat.S_ISDIR(st.st_mode) else "file",
              "exists": True, "size": st.st_size}
    if stat.S_ISREG(st.st_mode) and st.st_size <= MAX_WRITE:
        with open(path, "rb") as f:
            record["sha256"] = hashlib.sha256(f.read()).hexdigest()
    c.verified.append(record)


def run(req: dict) -> dict:
    """Execute one jailed operation and preserve success, diagnostics, and evidence."""
    started = time.monotonic()
    root = os.path.realpath(req["root"])
    cwd = os.path.realpath(req.get("cwd") or root)
    mode = req.get("permissions_mode", "workspace")
    allowed_roots = [os.path.realpath(path) for path in req.get("allowed_roots", [root])]
    read_only_roots = [os.path.realpath(path) for path in req.get("read_only_roots", [])]
    if mode != "full" and not any(_inside(cwd, path) for path in allowed_roots):
        cwd = root
    c = Ctx(root, cwd, mode, allowed_roots, read_only_roots)
    out, err, code = "", "", 0
    try:
        out, new = TOOLS[req["tool"]](c, req.get("args") or {})
        cwd = new or cwd
        code, err = c.exit_code, c.stderr
    except Denied as e:
        code, err = 1, f"DENIED: {e}"
    except Exception as e:  # report failures to both the UI and model
        code, err = 1, f"ERROR: {type(e).__name__}: {e}"
    return {"success": code == 0, "exit_code": code, "stdout": out, "stderr": err,
            "cwd": cwd, "duration": time.monotonic() - started,
            "verified": c.verified if code == 0 else []}
