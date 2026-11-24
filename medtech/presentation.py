"""Conservative UI-only summaries; the original model and tool records stay intact."""
import os
import re
from typing import Optional

_FILLER = re.compile(r"^(?:let me\b|(?:i(?:'ll| will)|we(?:'ll| will))\s+(?:now\s+)?(?:check|look|create|make|start|proceed|use|run|first|take|do)\b|(?:here(?:'s| is) (?:a |the )?(?:summary|recap)|plan|steps)\s*:)", re.I)
_OFFER = re.compile(r"(?:\s*(?:please )?let me know if\b.*|\s*(?:feel free to|if you(?:'d| would) like,? i can)\b.*)$", re.I)


def concise_text(text: str) -> str:
    """Remove recognizable filler without rewriting facts, code, or saved history."""
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I)
    text = re.sub(r"<think>.*$", "", text, flags=re.S | re.I)
    lines, code = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            code = not code
            lines.append(line)
            continue
        if code:
            lines.append(line)
            continue
        line = re.sub(r"^\s*(?:(?:sure|certainly|absolutely|of course|okay|ok)[!,:.]\s*)+", "", line, flags=re.I)
        line = _OFFER.sub("", line)
        if _FILLER.match(line.strip()):
            # Preserve a concrete result after a planning sentence on the same line.
            sentences = re.split(r"(?<=[.!?])\s+", line)
            line = " ".join(sentence for sentence in sentences if not _FILLER.match(sentence.strip()))
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines).strip()


def brief_error(text: str, limit: int = 260) -> str:
    """Keep the actual last exception/cause, without a traceback or long log."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    cause = lines[-1] if str(text).lstrip().startswith("Traceback") else lines[0] if lines else "Operation failed"
    cause = re.sub(r"^(?:(?:ERROR|DENIED):\s*)+", "", cause, flags=re.I)
    cause = re.sub(r"^(?:[\w.]+(?:Error|Exception)):\s*", "", cause)
    cause = re.sub(r"^\[Errno \d+\]\s*", "", cause)
    return cause if len(cause) <= limit else cause[:limit - 1] + "…"


def _path(tool: str, args: dict, result: dict) -> Optional[str]:
    for item in result.get("verified", []):
        if item.get("path") and (tool == "delete" or item.get("exists")):
            return str(item["path"])
    proposed = args.get("path") or args.get("dst") or args.get("src")
    if not isinstance(proposed, str):
        return None
    if os.path.isabs(proposed):
        return os.path.normpath(proposed)
    if proposed.startswith("~"):
        return os.path.expanduser(proposed)
    return os.path.normpath(os.path.join(result.get("cwd", ""), proposed))


def tool_outcome(tool: str, args: dict, result: dict, ascii: bool = False) -> str:
    """Summarize verified execution data, never a model's unverified narration."""
    passed = bool(result.get("success")) and result.get("exit_code") == 0
    mark = ("[OK]" if passed else "[ERROR]") if ascii else ("✓" if passed else "✗")
    path = _path(tool, args, result)
    if not passed:
        action = {"mkdir": "create directory", "write_file": "write", "read_file": "read",
                  "delete": "delete", "copy": "copy to", "move": "move to", "cd": "change directory to"}.get(tool, tool.replace("_", " "))
        cause = brief_error(result.get("stderr") or result.get("stdout") or f"Process exited with code {result.get('exit_code', '?')}")
        return f"{mark} Couldn't {action}{' ' + path if path else ''}: {cause} (Ctrl+D for details)"
    if tool in {"write_file", "mkdir", "delete", "copy", "move"}:
        verb = {"mkdir": "Created directory", "delete": "Deleted", "copy": "Copied to", "move": "Moved to"}.get(tool)
        if tool == "write_file":
            verb = {"create": "Created", "append": "Appended to", "overwrite": "Updated"}.get(args.get("mode", "create"), "Wrote")
        # A failed verification must never become a positive filesystem assertion.
        if not result.get("verified"):
            missing = "[ERROR]" if ascii else "✗"
            return f"{missing} Filesystem outcome could not be verified (Ctrl+D for details)"
        return f"{mark} {verb} {path or result.get('cwd', '')}"
    if tool in {"pwd", "cd"}:
        return f"{mark} Current directory: {result.get('cwd', '')}"
    return str(result.get("stdout") or f"{mark} Completed {tool}")
