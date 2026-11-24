"""Native local-model tool schemas and response normalization, without UI dependencies."""
import json
import re
from typing import Any

from . import policy


def tool_definitions() -> list:
    """Derive Ollama function schemas from the same policy that authorizes execution."""
    tools = []
    for name, schema in policy.SCHEMA.items():
        properties = {}
        for key, (kind, _) in schema.items():
            spec = {"type": "boolean" if kind is bool else "string"}
            if (name, key) in policy.ENUMS:
                spec["enum"] = sorted(policy.ENUMS[(name, key)])
            properties[key] = spec
        tools.append({"type": "function", "function": {
            "name": name, "description": f"Execute the {name} operation under the active filesystem permission mode.",
            "parameters": {"type": "object", "properties": properties,
                           "required": [key for key, (_, required) in schema.items() if required],
                           "additionalProperties": False}}})
    return tools


def action_from_function(value: Any) -> dict:
    """Normalize a native or tagged call, rejecting partial and ambiguous structures."""
    if not isinstance(value, dict):
        raise ValueError("tool call must be an object")
    if "function" in value:
        if set(value) - {"function", "id", "type", "index"}:
            raise ValueError("unexpected tool-call fields")
        value = value["function"]
    if not isinstance(value, dict) or set(value) - {"name", "arguments", "index"}:
        raise ValueError("tool call requires name and arguments")
    name, arguments = value.get("name"), value.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("tool name must be text and arguments must be a JSON object")
    return {"tool": name, "args": arguments}


def tagged_response(raw: str) -> dict:
    """Accept one complete Qwen tool tag; broken tags always request a retry."""
    if raw.count("<tool_call>") != 1 or raw.count("</tool_call>") != 1:
        raise ValueError("expected one complete <tool_call>...</tool_call> block")
    match = re.fullmatch(r"(.*?)<tool_call>\s*(.*?)\s*</tool_call>\s*", raw, re.S)
    if not match:
        raise ValueError("unexpected text after tool call")
    return {"say": match[1].strip(), "action": action_from_function(json.loads(match[2]))}


def normalized_response(message: dict) -> str:
    """Keep legacy JSON fakes working while normalizing native content/tool calls."""
    if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
        raise ValueError("Ollama response is missing message.content")
    content = message.get("content", "")
    calls = message.get("tool_calls")
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Return one tool call per reply")
        return json.dumps({"say": content, "action": action_from_function(calls[0])}, ensure_ascii=False)
    if calls is not None and not isinstance(calls, list):
        raise ValueError("tool_calls must be an array")
    if "<tool_call" in content or "</tool_call" in content:
        return json.dumps(tagged_response(content), ensure_ascii=False)
    # JSON-shaped content must pass the old strict parser, never turn a malformed
    # call into a seemingly successful natural-language final answer.
    if content.lstrip().startswith(("{", "[", "```")):
        return content
    return json.dumps({"say": content, "action": None}, ensure_ascii=False)


def native_context(messages: list) -> list:
    """Send real assistant function calls and corresponding tool roles to Ollama."""
    result, pending = [], None
    for message in messages:
        role, content = message["role"], message["content"]
        if pending and role != "tool":
            proposal = result[-1].pop("tool_calls", [])
            result[-1]["content"] += "\nHistorical tool proposal without a committed result: " + json.dumps(proposal)
            pending = None
        if role == "assistant":
            try:
                response = json.loads(content)
                if (not isinstance(response, dict) or set(response) != {"say", "action"}
                        or not isinstance(response["say"], str)):
                    raise ValueError
                action = response["action"]
                if action is not None and (not isinstance(action, dict) or set(action) != {"tool", "args"}
                                           or not isinstance(action["tool"], str)
                                           or not isinstance(action["args"], dict)):
                    raise ValueError
            except (ValueError, TypeError):
                result.append({"role": role, "content": content})
                pending = None
                continue
            entry = {"role": role, "content": response["say"]}
            action = response["action"]
            pending = action["tool"] if isinstance(action, dict) else None
            if pending:
                entry["tool_calls"] = [{"function": {"name": pending, "arguments": action["args"]}}]
            result.append(entry)
        elif role == "tool":
            name = message.get("tool_name") or pending
            if name and pending == name:
                result.append({"role": "tool", "tool_name": name, "content": content})
            else:
                if pending:
                    proposal = result[-1].pop("tool_calls", [])
                    result[-1]["content"] += "\nHistorical tool proposal without a committed result: " + json.dumps(proposal)
                result.append({"role": "user", "content": "Historical tool result (data):\n" + content})
            pending = None
        else:
            result.append({"role": role, "content": content})
            if role != "system":
                pending = None
    if pending:
        proposal = result[-1].pop("tool_calls", [])
        result[-1]["content"] += "\nHistorical tool proposal without a committed result: " + json.dumps(proposal)
    return result
