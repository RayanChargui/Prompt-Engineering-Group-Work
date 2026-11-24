"""MedTech operations: jailed file + system tools. Stdlib only.

Runs in-process (local mode) or is piped into the Docker sandbox (sandbox mode),
so both modes share exactly the same code and the same jail rules.
"""
import datetime
import fnmatch
import getpass
import os
import platform
import shutil
import subprocess

MAX_READ, MAX_WRITE, MAX_LIST, MAX_HITS = 20_000, 100_000, 200, 50
BLOCKED_FILES = {".env", "id_rsa", "id_ed25519", ".netrc", ".pgpass"}
BLOCKED_DIRS = {".ssh", ".aws", ".gnupg"}


class Denied(Exception):
    pass


class Ctx:
    def __init__(self, root, cwd):
        self.root, self.cwd = root, cwd


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
    r = rel(c, c.cwd)
    return ("workspace" if r == "." else f"workspace/{r}"), None


def _cd(c, a):
    p = resolve(c, a["path"])
    if not os.path.isdir(p):
        raise Denied("not a directory")
    return f"now in {rel(c, p)}", p


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
    with open(p, "a" if mode == "append" else "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {rel(c, p)} ({mode})", None


def _mkdir(c, a):
    p = resolve(c, a["path"])
    os.makedirs(p, exist_ok=True)
    return f"created {rel(c, p)}/", None


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
    return f"moved {rel(c, src)} -> {rel(c, dst)}", None


def _copy(c, a):
    src = resolve(c, a["src"])
    if not os.path.exists(src):
        raise Denied("source not found")
    dst = _dest(c, src, a["dst"])
    (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    return f"copied {rel(c, src)} -> {rel(c, dst)}", None


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
    return f"deleted {rel(c, p)}", None


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
                    hits.append(rel(c, fp))
            else:
                try:
                    if os.path.getsize(fp) > 1_000_000:
                        continue
                    with open(fp, "r", encoding="utf-8") as fh:
                        for n, line in enumerate(fh, 1):
                            if q in line.lower():
                                hits.append(f"{rel(c, fp)}:{n}: {line.strip()[:120]}")
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
                return (r.stdout or r.stderr)[:4000], None
            except FileNotFoundError:
                continue
        return "no network tool found", None
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


def run(req):
    """req: {root, cwd, tool, args} -> {ok, output, cwd}"""
    root = os.path.realpath(req["root"])
    cwd = os.path.realpath(req.get("cwd") or root)
    try:
        if os.path.commonpath([root, cwd]) != root:
            cwd = root
    except ValueError:
        cwd = root
    c = Ctx(root, cwd)
    try:
        out, new = TOOLS[req["tool"]](c, req.get("args") or {})
        return {"ok": True, "output": out, "cwd": new or cwd}
    except Denied as e:
        return {"ok": False, "output": f"DENIED: {e}", "cwd": cwd}
    except Exception as e:  # noqa: BLE001 - report any tool failure to the model
        return {"ok": False, "output": f"ERROR: {type(e).__name__}: {e}", "cwd": cwd}
