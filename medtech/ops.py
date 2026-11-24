"""MedTech operations: jailed file + system tools. Stdlib only.

Runs in-process (local mode) or is piped into the Docker sandbox (sandbox mode),
so both modes share exactly the same code and the same jail rules.
"""
import datetime
import fnmatch
import getpass
import hashlib
import os
import platform
import shutil
import subprocess
import time

MAX_READ, MAX_WRITE, MAX_LIST, MAX_HITS = 20_000, 100_000, 200, 50
BLOCKED_FILES = {".env", "id_rsa", "id_ed25519", ".netrc", ".pgpass"}
BLOCKED_DIRS = {".ssh", ".aws", ".gnupg"}


class Denied(Exception):
    pass


class Ctx:
    def __init__(self, root, cwd):
        self.root, self.cwd = root, cwd
        self.verified = []
        self.stderr = ""
        self.exit_code = 0


def resolve(c, path):
    """Resolve a user/LLM-supplied path and make sure it stays inside the workspace."""
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise Denied("invalid path")
    path = path.strip()
    if path.startswith("~"):
        raise Denied("'~' is not allowed; use paths inside the workspace")
    p = path if os.path.isabs(path) else os.path.join(c.cwd, path)
    p = os.path.realpath(p)  # resolves symlinks and '..'
    try:
        inside = os.path.commonpath([c.root, p]) == c.root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise Denied(f"outside the workspace: {path}")
    parts = {x.lower() for x in rel(c, p).replace("\\", "/").split("/")}
    if parts & BLOCKED_DIRS or os.path.basename(p).lower() in BLOCKED_FILES:
        raise Denied(f"protected file or folder: {path}")
    return p


def rel(c, p):
    r = os.path.relpath(p, c.root)
    return "." if r == "." else r.replace(os.sep, "/")


def _pwd(c, a):
    return c.cwd, None


def _cd(c, a):
    p = resolve(c, a["path"])
    if not os.path.isdir(p):
        raise Denied("not a directory")
    return f"now in {p}", p


def _ls(c, a):
    p = resolve(c, a.get("path", "."))
    if not os.path.isdir(p):
        raise Denied("not a directory")
    rows = []
    for n in sorted(os.listdir(p))[:MAX_LIST]:
        f = os.path.join(p, n)
        rows.append(f"{n}/" if os.path.isdir(f) else f"{n}  ({os.path.getsize(f)} B)")
    return "\n".join(rows) or "(empty)", None


def _read(c, a):
    p = resolve(c, a["path"])
    if not os.path.isfile(p):
        raise Denied("not a file")
    with open(p, "rb") as f:
        data = f.read(MAX_READ + 1)
    if b"\x00" in data:
        raise Denied("binary file")
    text = data[:MAX_READ].decode("utf-8", errors="replace")
    return text + ("\n...[truncated]" if len(data) > MAX_READ else ""), None


def _write(c, a):
    p = resolve(c, a["path"])
    mode, content = a.get("mode", "create"), a["content"]
    if len(content) > MAX_WRITE:
        raise Denied("content too large")
    if os.path.isdir(p):
        raise Denied("path is a directory")
    if mode == "create" and os.path.exists(p):
        raise Denied("file exists (use mode 'overwrite' or 'append')")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if mode not in {"create", "overwrite", "append"}:
        raise Denied("invalid write mode")
    before = os.path.getsize(p) if mode == "append" and os.path.exists(p) else 0
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
    p = resolve(c, a["path"])
    os.makedirs(p, exist_ok=True)
    verify(c, p)
    return f"verified directory {p}", None


def _dest(c, src, dst_arg):
    dst = resolve(c, dst_arg)
    if os.path.isdir(dst):
        dst = resolve(c, os.path.join(dst, os.path.basename(src)))
    if os.path.exists(dst):
        raise Denied(f"destination exists: {rel(c, dst)}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst


def _move(c, a):
    src = resolve(c, a["src"])
    if src == c.root:
        raise Denied("cannot move the workspace root")
    if not os.path.exists(src):
        raise Denied("source not found")
    dst = _dest(c, src, a["dst"])
    shutil.move(src, dst)
    if os.path.lexists(src):
        raise OSError("move verification failed: source still exists")
    verify(c, dst)
    return f"verified move {src} -> {dst}", None


def _copy(c, a):
    src = resolve(c, a["src"])
    if not os.path.exists(src):
        raise Denied("source not found")
    dst = _dest(c, src, a["dst"])
    (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    verify(c, dst)
    return f"verified copy {src} -> {dst}", None


def _delete(c, a):
    p = resolve(c, a["path"])
    if p == c.root:
        raise Denied("cannot delete the workspace root")
    if os.path.isdir(p) and not os.path.islink(p):
        (shutil.rmtree if a.get("recursive") else os.rmdir)(p)
    elif os.path.exists(p) or os.path.islink(p):
        os.remove(p)
    else:
        raise Denied("not found")
    if os.path.lexists(p):
        raise OSError("delete verification failed: path still exists")
    return f"deleted {p}", None


def _search(c, a):
    q, kind = a["query"].lower(), a.get("kind", "name")
    base, hits = resolve(c, a.get("path", ".")), []
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in BLOCKED_DIRS]
        for f in files:
            fp = os.path.join(d, f)
            if f in BLOCKED_FILES or os.path.islink(fp):
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
                except (UnicodeDecodeError, OSError):
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
        u = shutil.disk_usage(c.root)
        return f"total {u.total >> 30} GiB, free {u.free >> 30} GiB", None
    if k == "date":
        return datetime.datetime.now().isoformat(timespec="seconds"), None
    raise Denied("unknown kind")


TOOLS = {"pwd": _pwd, "cd": _cd, "ls": _ls, "read_file": _read, "write_file": _write,
         "mkdir": _mkdir, "move": _move, "copy": _copy, "delete": _delete,
         "search": _search, "sysinfo": _sysinfo}


def verify(c: Ctx, path: str) -> None:
    """Record a stat/read-back result in the environment that performed the write."""
    st = os.stat(path)
    record = {"path": path, "kind": "directory" if os.path.isdir(path) else "file",
              "exists": True, "size": st.st_size}
    if os.path.isfile(path) and st.st_size <= MAX_WRITE:
        with open(path, "rb") as f:
            record["sha256"] = hashlib.sha256(f.read()).hexdigest()
    c.verified.append(record)


def run(req: dict) -> dict:
    """Execute one jailed operation and preserve success, diagnostics, and evidence."""
    started = time.monotonic()
    root = os.path.realpath(req["root"])
    cwd = os.path.realpath(req.get("cwd") or root)
    try:
        if os.path.commonpath([root, cwd]) != root:
            cwd = root
    except ValueError:
        cwd = root
    c = Ctx(root, cwd)
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
