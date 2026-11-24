"""Native Ollama requests, real incremental transport, and safe UI-independent drafts."""
import io
import json
import threading

import pytest

from medtech import agent, policy
from medtech.agent import Agent, Config, LLMError, ParseError, TurnCancelled, chat, parse
from medtech.backends import LocalBackend
from medtech.ollama import native_context, tool_definitions
from medtech.store import Store


def final(text):
    return {"message": {"content": text}}


def tool(name, arguments):
    return {"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}


def stream(*events):
    return io.BytesIO(b"".join(json.dumps(event).encode() + b"\n" for event in events))


class UI:
    """Record provisional and accepted output separately, as the UI adapter must."""
    def __init__(self): self.events = []
    def delta(self, text): self.events.append(("delta", text))
    def discard_draft(self): self.events.append(("discard",))
    def say(self, text): self.events.append(("say", text))
    def warn(self, text): self.events.append(("warn", text))
    def status(self, text): pass
    def saved(self): self.events.append(("saved",))
    def action(self, *args): pass
    def confirm(self, *args): return True
    def result(self, *args): pass


def make_agent(tmp_path, ui=None, **kwargs):
    store = Store(str(tmp_path / "chat.db"))
    backend = LocalBackend(str(tmp_path / "files"))
    sid = store.new_session("local", backend.root)
    return Agent(Config(), backend, store, sid, ui or UI(), auto=True,
                 memory_extractor=lambda text: None, **kwargs)


def test_config_default_environment_and_validation(monkeypatch):
    monkeypatch.delenv("MEDTECH_MODEL", raising=False)
    monkeypatch.delenv("MEDTECH_NUM_CTX", raising=False)
    assert Config().model == "qwen2.5:7b"
    assert Config().num_ctx == 16384
    monkeypatch.setenv("MEDTECH_NUM_CTX", "8192")
    assert Config().num_ctx == 8192
    assert Config(num_ctx=32768).num_ctx == 32768
    for value in (4096, 32769, "16384", True):
        with pytest.raises(ValueError, match="8192 to 32768"):
            Config(num_ctx=value)
    with pytest.raises(ValueError, match="num_predict"):
        Config(num_predict=0)


def test_native_tools_derive_all_policy_argument_types_and_enums():
    tools = {entry["function"]["name"]: entry["function"]["parameters"] for entry in tool_definitions()}
    assert tools.keys() == policy.SCHEMA.keys()
    for name, schema in policy.SCHEMA.items():
        assert tools[name]["additionalProperties"] is False
        assert set(tools[name]["required"]) == {key for key, (_, required) in schema.items() if required}
        for key, (kind, _) in schema.items():
            assert tools[name]["properties"][key]["type"] == ("boolean" if kind is bool else "string")
            if (name, key) in policy.ENUMS:
                assert set(tools[name]["properties"][key]["enum"]) == policy.ENUMS[(name, key)]


def test_native_request_uses_configured_model_context_and_tools(monkeypatch):
    requests = []
    monkeypatch.setattr(agent, "_post", lambda url, body, timeout: requests.append(body) or final("Hello Rayan."))
    response = chat(Config(num_ctx=8192, num_predict=700), [{"role": "user", "content": "hello"}])
    assert parse(response) == {"say": "Hello Rayan.", "action": None}
    assert requests[0]["model"] == "qwen2.5:7b"
    assert requests[0]["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 700}
    assert requests[0]["tools"] == tool_definitions()
    assert "format" not in requests[0]


def test_memory_request_is_json_without_tools_or_streaming(monkeypatch):
    requests = []
    monkeypatch.setattr(agent, "_post", lambda url, body, timeout: requests.append(body) or final('{"facts":[]}'))
    assert chat(Config(), [], memory=True, on_delta=lambda text: pytest.fail("memory leaked a draft")) == '{"facts":[]}'
    body = requests[0]
    assert body["format"] == "json" and body["stream"] is False and "tools" not in body
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] <= 400


def test_native_call_and_followup_use_real_tool_role(tmp_path, monkeypatch):
    ui = UI()
    # Deliberately omit delta here to exercise the legacy _post injection seam.
    ui.delta = None
    runner = make_agent(tmp_path, ui)
    replies = iter([tool("mkdir", '{"path":"actual"}'), final(f"Created folder {runner.backend.root}/actual")])
    requests = []
    monkeypatch.setattr(agent, "_post", lambda url, body, timeout: requests.append(body) or next(replies))
    runner.turn("Create folder actual")
    assert (tmp_path / "files/actual").is_dir()
    context = requests[1]["messages"]
    call = next(message for message in context if message.get("tool_calls"))
    assert call["tool_calls"][0]["function"] == {"name": "mkdir", "arguments": {"path": "actual"}}
    result = next(message for message in context if message["role"] == "tool")
    assert result["tool_name"] == "mkdir" and '"success": true' in result["content"]
    assert ui.events[-1] == ("say", f"Created folder {runner.backend.root}/actual")


@pytest.mark.parametrize("raw", [
    '<tool_call>{"name":"mkdir","arguments":{"path":"actual"}}</tool_call>',
    'I will use the tool. <tool_call>{"function":{"name":"mkdir","arguments":"{\\"path\\":\\"actual\\"}"}}</tool_call>',
])
def test_tagged_qwen_fallback(raw):
    assert parse(raw)["action"] == {"tool": "mkdir", "args": {"path": "actual"}}


@pytest.mark.parametrize("raw", [
    '<tool_call>{"name":"mkdir","arguments":{}}',
    '<tool_call>{"name":"mkdir","arguments":{}}</tool_call> trailing',
    '<tool_call>{"name":"mkdir","arguments":42}</tool_call>',
    '<tool_call>{"name":"mkdir","arguments":{}}</tool_call><tool_call>{}</tool_call>',
])
def test_partial_or_ambiguous_tagged_calls_are_rejected(raw):
    with pytest.raises(ParseError):
        parse(raw)


def test_multiple_native_calls_force_corrective_retry_not_silent_drop(tmp_path, monkeypatch):
    ui = UI()
    ui.delta = None
    runner = make_agent(tmp_path, ui)
    multiple = tool("mkdir", {"path": "first"})
    multiple["message"]["tool_calls"].append(tool("mkdir", {"path": "second"})["message"]["tool_calls"][0])
    replies = iter([multiple, final("Neither operation has been executed.")])
    requests = []
    monkeypatch.setattr(agent, "_post", lambda url, body, timeout: requests.append(body) or next(replies))
    runner.turn("Create two folders")
    assert not runner.store.tasks(runner.sid)
    assert any("Invalid response" in message["content"] for message in requests[1]["messages"])
    assert ui.events[-1] == ("say", "Neither operation has been executed.")


def test_real_stream_transport_emits_actual_incoming_deltas(monkeypatch):
    deltas, requests = [], []
    observed = threading.Event()

    def delta(text):
        deltas.append(text)
        observed.set()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield json.dumps({"message": {"content": "Hello "}, "done": False}).encode()
            assert observed.wait(1) and deltas == ["Hello "]  # Display precedes completion.
            observed.clear()
            yield json.dumps({"message": {"content": "Rayan."}, "done": False}).encode()
            assert observed.wait(1) and deltas == ["Hello ", "Rayan."]
            yield b'{"done":true,"message":{"content":""}}'

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(agent.urllib.request, "urlopen", urlopen)
    result = chat(Config(), [], on_delta=delta)
    assert parse(result) == {"say": "Hello Rayan.", "action": None}
    assert requests[0]["stream"] is True and "tools" in requests[0]


def test_streamed_hallucination_is_discarded_before_accepted_answer(tmp_path, monkeypatch):
    runner = make_agent(tmp_path)
    replies = iter([
        stream({"message": {"content": "I created folder imaginary"}, "done": True}),
        stream({"message": {"content": "I have not executed that request."}, "done": True}),
    ])
    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda *args, **kwargs: next(replies))
    runner.turn("Create folder imaginary")
    events = runner.ui.events
    assert ("delta", "I created folder imaginary") in events
    assert ("say", "I created folder imaginary") not in events
    assert events.index(("discard",)) > events.index(("delta", "I created folder imaginary"))
    assert events[-1] == ("say", "I have not executed that request.")
    assert not runner.store.tasks(runner.sid)


def test_incomplete_stream_is_clear_error_and_discards_draft(tmp_path, monkeypatch):
    runner = make_agent(tmp_path)
    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda *a, **kw: stream({"message": {"content": "Half an answer"}}))
    with pytest.raises(LLMError, match="ended before completion"):
        runner.turn("hello")
    assert runner.ui.events[-1] == ("discard",)
    assert not any(event[0] == "say" for event in runner.ui.events)


def test_cancellation_stops_stream_and_next_tool(tmp_path, monkeypatch):
    event = threading.Event()
    runner = make_agent(tmp_path, cancel_event=event)
    original_delta = runner.ui.delta
    def cancel_after_piece(text):
        original_delta(text)
        event.set()
    runner.ui.delta = cancel_after_piece
    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda *a, **kw: stream(
        {"message": {"content": "Starting"}}, {"done": True, "message": {"content": "done"}}))
    with pytest.raises(TurnCancelled):
        runner.turn("Create folder actual")
    assert not runner.store.tasks(runner.sid) and runner.ui.events[-1] == ("discard",)
    assert runner.store.messages(runner.sid)[0]["content"] == "Create folder actual"


def test_background_memory_callback_skips_blocking_extraction(tmp_path):
    runner = make_agent(tmp_path)
    scheduled = []
    runner.memory_extractor = scheduled.append
    runner.llm = lambda messages: json.dumps({"say": "Hello Rayan.", "action": None})
    runner.turn("My name is Rayan")
    assert scheduled == ["My name is Rayan"]
    assert not runner.store.memories()
    assert ("saved",) in runner.ui.events


def test_orphan_context_does_not_send_invalid_tool_role_chains():
    action = json.dumps({"say": "", "action": {"tool": "mkdir", "args": {"path": "x"}}})
    missing_result = native_context([{"role": "assistant", "content": action}, {"role": "user", "content": "cancelled"}])
    assert not any(message.get("tool_calls") for message in missing_result)
    orphan = native_context([{"role": "tool", "tool_name": "mkdir", "content": "historical result"}])
    assert orphan[0]["role"] == "user"
    interrupted = native_context([{"role": "assistant", "content": action}])
    assert "tool_calls" not in interrupted[0]


def test_debug_is_local_file_only(tmp_path, capsys):
    runner = make_agent(tmp_path, debug=True, debug_log=str(tmp_path / "debug.log"))
    replies = iter([json.dumps({"say": "", "action": {"tool": "mkdir", "args": {"path": "x"}}}),
                    json.dumps({"say": "Created folder x", "action": None})])
    runner.llm = lambda messages: next(replies)
    runner.turn("Create folder x")
    log = (tmp_path / "debug.log").read_text()
    assert "DEBUG tool" in log and "DEBUG result" in log and '"success": true' in log
    assert not capsys.readouterr().err


def test_cancellation_does_not_wait_for_blocked_network_read(monkeypatch):
    blocked, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            blocked.set()
            release.wait(2)
            yield b'{"done":true,"message":{"content":"late answer"}}'

    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    caught = []
    def request():
        try:
            chat(Config(), [], on_delta=lambda text: pytest.fail("cancelled draft leaked"), cancel_event=cancelled)
        except TurnCancelled:
            caught.append(True)

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert blocked.wait(1)
        cancelled.set()
        worker.join(1)
        assert not worker.is_alive() and caught == [True]
    finally:
        release.set()
        worker.join(2)


def test_memory_cancellation_does_not_wait_for_blocked_post(monkeypatch):
    blocked, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    def post(*args, **kwargs):
        blocked.set()
        release.wait(2)
        return final('{"facts":[]}')
    monkeypatch.setattr(agent, "_post_raw", post)
    caught = []
    def request():
        try:
            chat(Config(), [], memory=True, cancel_event=cancelled)
        except TurnCancelled:
            caught.append(True)
    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert blocked.wait(1)
        cancelled.set()
        worker.join(1)
        assert not worker.is_alive() and caught == [True]
    finally:
        release.set()
        worker.join(2)


def test_native_history_budget_reserves_function_schema_window(tmp_path):
    runner = make_agent(tmp_path)
    runner.cfg.num_ctx = 8192
    for index in range(20):
        runner._record("user", f"Entry {index}: " + "界" * 300)
        runner._record("assistant", json.dumps({"say": "Noted.", "action": None}))
    messages = runner._context()
    request_bytes = len(json.dumps({"messages": messages, "tools": tool_definitions()}, ensure_ascii=False).encode())
    assert request_bytes + runner.cfg.num_predict < runner.cfg.num_ctx
    assert "Entry 19" in str(messages)
    assert len(runner.store.messages(runner.sid)) == 40
