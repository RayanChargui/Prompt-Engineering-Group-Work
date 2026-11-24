"""Security layer. The LLM only *proposes* actions; this module decides what runs.

Defence in depth:
  1. Capability tools only (no free-form shell) with strict argument schemas.
  2. Risk levels: read -> auto, write -> confirm (or auto in sandbox), danger -> always confirm.
  3. Taint tracking: after untrusted text (documents, file contents) entered the
     model's context, even write actions need explicit confirmation.
  4. Untrusted text is wrapped, sanitised and scanned for injection phrases.
  5. The workspace jail itself is enforced in ops.py (not by the model).
"""
import re
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


def check(tool, args, auto=False, tainted=False):
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
