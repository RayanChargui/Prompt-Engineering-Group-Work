"""Security layer. The LLM only *proposes* actions; this module decides what runs.

Defence in depth:
  1. Capability tools only (no free-form shell) with strict argument schemas.
  2. Local typed tools normally execute directly; sandbox mode retains approvals.
  3. Sandbox taint tracking requires write confirmation after file contents enter
     the model context; local mode confirms only clearly catastrophic operations.
  4. Untrusted text is wrapped, sanitised and scanned for injection phrases.
  5. Configured workspace roots are enforced in ops.py (not by the model).
"""
import re
import os
import ntpath
import stat
from dataclasses import dataclass

STR, BOOL = str, bool
SCHEMA = {
    "pwd": {}, "ls": {"path": (STR, False)}, "cd": {"path": (STR, True)},
    "read_file": {"path": (STR, True)},
    "search": {"query": (STR, True), "path": (STR, False), "kind": (STR, False)},
    "sysinfo": {"kind": (STR, False)},
    "write_file": {"path": (STR, True), "content": (STR, True), "mode": (STR, False)},
    "mkdir": {"path": (STR, True)},
    "move": {"src": (STR, True), "dst": (STR, True)},
    "copy": {"src": (STR, True), "dst": (STR, True)},
    "delete": {"path": (STR, True), "recursive": (BOOL, False)},
}
READ = {"pwd", "ls", "cd", "read_file", "search", "sysinfo"}
DANGER = {"delete"}
ENUMS = {("write_file", "mode"): {"create", "overwrite", "append"},
         ("search", "kind"): {"name", "content"},
         ("sysinfo", "kind"): {"os", "network", "whoami", "disk", "date"}}
TAINTING = {"read_file", "search"}  # outputs that may carry attacker-controlled text
MAX_STR = 100_000


@dataclass
class Decision:
    level: str  # auto | confirm | block
    risk: str   # read | write | danger
    why: str = ""


def is_catastrophic(tool: str, args: dict, cwd: str = None, root: str = None, home: str = None) -> bool:
    """Identify root/home removal or relocation and destructive raw-device writes.

    This classifies typed proposals only; no command string is parsed or executed.
    Windows drive/UNC anchors can also be classified safely by tests on Unix.
    """
    if not isinstance(args, dict) or tool not in {"delete", "move", "write_file"}:
        return False
    if tool == "delete" and args.get("recursive") is not True:
        return False
    if tool == "write_file" and args.get("mode", "create") not in {"overwrite", "append"}:
        return False
    path = args.get("src" if tool == "move" else "path")
    if not isinstance(path, str) or not path.strip():
        return False
    path = path.strip()
    current = cwd or root or os.getcwd()
    user_home = home or os.path.expanduser("~")
    windows = bool(ntpath.splitdrive(path)[0] or ntpath.splitdrive(current)[0])
    paths = ntpath if windows else os.path
    if path == "~" or path.startswith(("~/", "~\\")):
        path = paths.join(user_home, path[2:]) if len(path) > 1 else user_home
    else:
        path = os.path.expanduser(path)
    if not paths.isabs(path):
        path = paths.join(current, path)
    resolved = paths.normcase(paths.normpath(path)) if windows else os.path.realpath(path)
    native_home = paths.normcase(paths.normpath(user_home)) if windows else os.path.realpath(user_home)
    if tool == "write_file":
        drive_prefix = "\\\\.\\physicaldrive"
        lowered = resolved.lower()
        if lowered.startswith(drive_prefix) and lowered[len(drive_prefix):].isdigit():
            return True
        try:
            return stat.S_ISBLK(os.stat(resolved).st_mode)
        except OSError:
            # The actual operation still reports stat/open errors faithfully.
            return False
    if paths.dirname(resolved) == resolved:
        return True
    try:
        return paths.commonpath([resolved, native_home]) == resolved
    except ValueError:
        return False


def check(tool, args, auto=False, tainted=False, *, local=False, permissions_mode="workspace",
          root=None, cwd=None, confirm_dangerous=True, home=None):
    if tool not in SCHEMA:
        return Decision("block", "danger", f"unknown tool '{tool}'")
    if not isinstance(args, dict):
        return Decision("block", "danger", "arguments must be an object")
    spec = SCHEMA[tool]
    for k, v in args.items():
        if k not in spec:
            return Decision("block", "danger", f"unexpected argument '{k}'")
        if not isinstance(v, spec[k][0]):
            return Decision("block", "danger", f"argument '{k}' has the wrong type")
        if isinstance(v, str) and len(v) > MAX_STR:
            return Decision("block", "danger", f"argument '{k}' is too long")
        if (tool, k) in ENUMS and v not in ENUMS[(tool, k)]:
            return Decision("block", "danger", f"invalid value for '{k}'")
    missing = [k for k, (_, req) in spec.items() if req and k not in args]
    if missing:
        return Decision("block", "danger", f"missing argument(s): {', '.join(missing)}")

    if local:
        catastrophic = is_catastrophic(tool, args, cwd=cwd, root=root, home=home)
        if catastrophic and confirm_dangerous:
            return Decision("confirm", "danger", "catastrophic filesystem operation: root/home removal or relocation, or raw-device write")
        risk = "danger" if tool in DANGER or catastrophic else "read" if tool in READ else "write"
        return Decision("auto", risk)

    if tool in DANGER or (tool == "write_file" and args.get("mode") in ("overwrite", "append")):
        return Decision("confirm", "danger", "destructive or overwriting action")
    if tool in READ:
        return Decision("auto", "read")
    if tainted:
        return Decision("confirm", "write", "untrusted content is in context")
    return Decision("auto" if auto else "confirm", "write")


_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard .{0,40}(instructions|rules|system prompt)",
    r"(reveal|print|show|leak) .{0,30}(system prompt|api key|password|secret)",
    r"you are now\b", r"new instructions?:", r"</?(system|assistant)>", r"<\|im_(start|end)\|>",
    r"(delete|erase|wipe) (all|every|everything)", r"\brm\s+-rf\b",
)]


def injection_flags(text):
    return [p.pattern for p in _INJECTION if p.search(text)]


def wrap_untrusted(source, text):
    text = re.sub(r"</?\s*untrusted[^>]*>", "[tag removed]", text, flags=re.I)
    return f'<untrusted source="{source}">\n{text}\n</untrusted>'
