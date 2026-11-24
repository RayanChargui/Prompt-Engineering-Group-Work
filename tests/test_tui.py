"""Headless terminal interactions with real SQLite and a scripted local model."""
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from textual.widgets import Input, Static, TextArea

from medtech.agent import Config
from medtech.backends import LocalBackend
from medtech.store import Store
from medtech.tui import MedTechApp


class FakeLLM:
    """Use strict extraction and answer schemas without a running model service."""

    def __call__(self, messages):
        if messages[0]["content"].startswith("Extract only"):
            return '{"facts":[]}'
        return json.dumps({"say": "Acknowledged your message.", "action": None})


@pytest.fixture
def tui_state(tmp_path):
    store = Store(tmp_path / "chats.db")
    backend = LocalBackend(str(tmp_path / "files"))
    yield store, backend
    store.close()


async def eventually(pilot, predicate, timeout=6):
    """Wait for asynchronous events and database workers without blocking the UI."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for terminal state")
        await pilot.pause(0.03)


def visible_text(app):
    """Collect rendered text from terminal labels, messages, and status widgets."""
    rendered = []
    for widget in app.query(Static):
        value = widget.renderable
        rendered.append(getattr(value, "plain", str(value)))
    return "\n".join(rendered)


def add_chat(store, backend, title, when=None):
    sid = store.new_session("local", backend.root)
    store.add_message(sid, "user", title)
    store.add_message(sid, "assistant", json.dumps({"say": "Saved reply for " + title, "action": None}))
    if when is not None:
        store._w("UPDATE sessions SET updated_at=? WHERE id=?", when.timestamp(), sid)
    return sid


@pytest.mark.asyncio
async def test_sidebar_groups_dates_and_click_loads_full_history(tui_state):
    store, backend = tui_state
    now = datetime.now()
    ids = [add_chat(store, backend, title, when) for title, when in (
        ("Today's discussion", now),
        ("Yesterday's discussion", now - timedelta(days=1)),
        ("This week's discussion", now - timedelta(days=3)),
        ("Old discussion", now - timedelta(days=20)),
    )]
    # More than the model's recent-message window must still render in the UI.
    for index in range(18):
        store.add_message(ids[-1], "user", f"Archived message {index}")
    store._w("UPDATE sessions SET updated_at=? WHERE id=?", (now - timedelta(days=20)).timestamp(), ids[-1])
    app = MedTechApp(Config(), backend, store, llm=FakeLLM())
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        text = visible_text(app)
        for group in ("Today", "Yesterday", "Previous 7 days", "Older"):
            assert group in text
        await pilot.click(f"#chat-{ids[-1]}")
        await eventually(pilot, lambda: app.sid == ids[-1])
        text = visible_text(app)
        assert "Old discussion" in text
        assert "Archived message 0" in text
        assert "Archived message 17" in text
        assert "Saved reply for Old discussion" in text


@pytest.mark.asyncio
async def test_new_chat_first_message_updates_title_live(tui_state):
    store, backend = tui_state
    previous = add_chat(store, backend, "Previous chat")
    app = MedTechApp(Config(), backend, store, sid=previous, llm=FakeLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.click("#new-chat")
        assert app.sid != previous
        app.query_one("#message-input", TextArea).load_text("My first terminal conversation")
        await pilot.click("#send")
        sid = app.sid
        assert sid is not None
        await eventually(pilot, lambda: not app.busy and bool(store.messages(sid)))
        assert store.session(sid)["title"] == "My first terminal conversation"
        assert "My first terminal conversation" in visible_text(app)
        assert "Acknowledged your message." in visible_text(app)
        assert len(store.messages(previous)) == 2


@pytest.mark.asyncio
async def test_rename_and_delete_dialog_refresh_sidebar(tui_state):
    store, backend = tui_state
    sid = add_chat(store, backend, "Original title")
    app = MedTechApp(Config(), backend, store, sid=sid, llm=FakeLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.press("ctrl+o")
        await pilot.click("#chat-rename")
        app.screen.query_one("#rename-title", Input).value = "Renamed from dialog"
        await pilot.click("#rename-save")
        await eventually(pilot, lambda: store.session(sid)["title"] == "Renamed from dialog")
        assert "Renamed from dialog" in visible_text(app)
        await pilot.press("ctrl+o")
        await pilot.click("#chat-delete")
        await pilot.click("#delete-confirm")
        await eventually(pilot, lambda: store.session(sid) is None)
        assert app.sid != sid
        assert not app.query(f"#chat-{sid}")
        assert "Renamed from dialog" not in visible_text(app)


@pytest.mark.asyncio
async def test_sidebar_toggle_filter_and_narrow_resize(tui_state):
    store, backend = tui_state
    match = add_chat(store, backend, "Find this Python conversation")
    other = add_chat(store, backend, "Gardening notes")
    app = MedTechApp(Config(), backend, store, llm=FakeLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        assert app.query_one("#sidebar").display
        await pilot.press("ctrl+b")
        assert not app.query_one("#sidebar").display
        await pilot.press("ctrl+b")
        assert app.query_one("#sidebar").display
        await pilot.click("#chat-search")
        await pilot.press(*"Python")
        await eventually(pilot, lambda: not app.query(f"#chat-{other}"))
        assert app.query(f"#chat-{match}")
        app.query_one("#chat-search", Input).value = ""
        await pilot.pause()
        await pilot.resize_terminal(65, 30)
        await pilot.pause()
        assert not app.query_one("#sidebar").display
        assert app.query_one("#message-input").region.width > 35


@pytest.mark.asyncio
async def test_memory_modal_deletes_persisted_fact(tui_state):
    store, backend = tui_state
    store.remember("name", "Rayan")
    fact_id = store.memories()[0]["id"]
    app = MedTechApp(Config(), backend, store, llm=FakeLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.click("#memory")
        assert "Rayan" in visible_text(app)
        await pilot.click(f"#forget-{fact_id}")
        await eventually(pilot, lambda: not store.memories())
        assert "Rayan" not in visible_text(app)
        await pilot.click("#memory-close")
        assert app.query_one("#message-input")


@pytest.mark.asyncio
async def test_background_memory_does_not_block_reply_sidebar_or_persistence(tui_state):
    store, backend = tui_state
    entered, release = threading.Event(), threading.Event()

    class SlowLLM(FakeLLM):
        def __call__(self, messages):
            if messages[0]["content"].startswith("Extract only"):
                entered.set()
                release.wait(8)
            return super().__call__(messages)

    app = MedTechApp(Config(), backend, store, llm=SlowLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        try:
            await app.send_message("Persist before the model finishes")
            await eventually(pilot, entered.is_set)
            await eventually(pilot, lambda: "Acknowledged your message." in visible_text(app))
            assert not release.is_set()
            assert store.messages(app.sid)[0]["content"] == "Persist before the model finishes"
            await pilot.press("ctrl+b")
            assert not app.query_one("#sidebar").display
            await pilot.press("ctrl+b")
            assert app.query_one("#sidebar").display
        finally:
            release.set()
        await eventually(pilot, lambda: not app.busy)
        assert "Acknowledged your message." in visible_text(app)


@pytest.mark.asyncio
async def test_background_generation_keeps_terminal_responsive(tui_state):
    store, backend = tui_state
    entered, release = threading.Event(), threading.Event()

    class SlowGeneration(FakeLLM):
        def __call__(self, messages):
            if not messages[0]["content"].startswith("Extract only"):
                entered.set()
                release.wait(8)
            return super().__call__(messages)

    app = MedTechApp(Config(), backend, store, llm=SlowGeneration())
    async with app.run_test(size=(140, 42)) as pilot:
        try:
            await app.send_message("Persist during generation")
            await eventually(pilot, entered.is_set)
            assert app.busy
            assert store.messages(app.sid)[0]["content"] == "Persist during generation"
            await pilot.press("ctrl+b")
            assert not app.query_one("#sidebar").display
            await pilot.press("ctrl+b")
            assert app.query_one("#sidebar").display
        finally:
            release.set()
        await eventually(pilot, lambda: not app.busy)
        assert "Acknowledged your message." in visible_text(app)


@pytest.mark.asyncio
async def test_terminal_layout_exports_wide_and_narrow_screenshots(tui_state, tmp_path):
    store, backend = tui_state
    sid = add_chat(store, backend, "Plan a local Python project")
    add_chat(store, backend, "Workspace tools and files")
    store.remember("name", "Rayan")
    app = MedTechApp(Config(), backend, store, sid=sid, llm=FakeLLM())
    destination = Path(os.environ.get("MEDTECH_SCREENSHOT_DIR", tmp_path))
    destination.mkdir(parents=True, exist_ok=True)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        wide = app.export_screenshot()
        assert "<svg" in wide
        (destination / "ui-wide.svg").write_text(wide, encoding="utf-8")
        await pilot.resize_terminal(65, 30)
        await pilot.pause()
        narrow = app.export_screenshot()
        assert "<svg" in narrow
        (destination / "ui-narrow.svg").write_text(narrow, encoding="utf-8")
        assert not app.query_one("#sidebar").display


@pytest.mark.asyncio
async def test_keyboard_sidebar_selection_f2_and_delete_confirmation(tui_state):
    store, backend = tui_state
    older = add_chat(store, backend, "Earlier conversation")
    latest = add_chat(store, backend, "Latest conversation")
    app = MedTechApp(Config(), backend, store, sid=latest, llm=FakeLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        listing = app.query_one("#chat-list")
        listing.focus()
        await pilot.press("down", "enter")
        await eventually(pilot, lambda: app.sid == older)
        assert "Saved reply for Earlier conversation" in visible_text(app)
        app.query_one("#chat-list").focus()
        await pilot.press("f2")
        app.screen.query_one("#rename-title", Input).value = "Keyboard renamed"
        await pilot.press("enter")
        await eventually(pilot, lambda: store.session(older)["title"] == "Keyboard renamed")
        app.query_one("#chat-list").focus()
        await pilot.press("delete")
        await pilot.click("#cancel")
        assert store.session(older) is not None
        app.query_one("#chat-list").focus()
        await pilot.press("delete")
        await pilot.click("#delete-confirm")
        await eventually(pilot, lambda: store.session(older) is None)
        assert not app.query(f"#chat-{older}")
        assert store.session(latest) is not None


@pytest.mark.asyncio
async def test_local_tools_use_permission_dialogs_and_verified_real_files(tui_state):
    from textual.widgets import Collapsible

    store, backend = tui_state
    responses = iter([
        {"say": "", "action": {"tool": "mkdir", "args": {"path": "test_dir"}}},
        {"say": "", "action": {"tool": "write_file", "args": {"path": "test_dir/hello.txt", "content": "hi"}}},
        {"say": f"Created folder {backend.root}/test_dir and file {backend.root}/test_dir/hello.txt.", "action": None},
    ])

    class ToolLLM(FakeLLM):
        def __call__(self, messages):
            if messages[0]["content"].startswith("Extract only"):
                return super().__call__(messages)
            return json.dumps(next(responses))

    app = MedTechApp(Config(), backend, store, llm=ToolLLM(), auto=False)
    async with app.run_test(size=(140, 42)) as pilot:
        await app.send_message("Create folder `test_dir` with file `hello.txt` containing 'hi'")
        await eventually(pilot, lambda: bool(app.screen.query("#tool-allow")))
        await pilot.pause()
        assert not (Path(backend.root) / "test_dir").exists()
        await pilot.click("#tool-allow")
        await eventually(pilot, lambda: len(store.tasks(app.sid)) == 1 and bool(app.screen.query("#tool-allow")))
        await pilot.pause()
        assert (Path(backend.root) / "test_dir").is_dir()
        assert not (Path(backend.root) / "test_dir/hello.txt").exists()
        await pilot.click("#tool-allow")
        await eventually(pilot, lambda: not app.busy)
        assert (Path(backend.root) / "test_dir/hello.txt").read_text() == "hi"
        tasks = store.tasks(app.sid)
        assert [row["status"] for row in tasks] == ["done", "done"]
        for row in tasks:
            result = json.loads(row["output"])
            assert result["success"] and result["exit_code"] == 0
            assert result["verified"][0]["path"].startswith(backend.root)
            assert {"stdout", "stderr", "duration", "cwd"} <= result.keys()
        sections = list(app.query(Collapsible))
        assert len(sections) == 4
        assert all(section.collapsed for section in sections)
        result_sections = [section for section in sections if "Tool result" in section.title]
        assert len(result_sections) == 2
        result_title = result_sections[0].query_one("CollapsibleTitle")
        result_title.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click(result_title)
        assert not result_sections[0].collapsed
        assert '"exit_code": 0' in visible_text(app)
        assert '"verified"' in visible_text(app)
        assert str(Path(backend.root) / "test_dir/hello.txt") in visible_text(app)


@pytest.mark.asyncio
async def test_memory_dialog_fits_narrow_terminal(tui_state):
    from textual.containers import Vertical

    store, backend = tui_state
    store.remember("name", "Rayan")
    app = MedTechApp(Config(), backend, store, llm=FakeLLM())
    async with app.run_test(size=(50, 28)) as pilot:
        assert not app.query_one("#sidebar").display
        await app.slash_command("/memory")
        await pilot.pause()
        panel = app.screen.query_one(Vertical)
        assert 0 <= panel.region.x
        assert panel.region.right <= app.size.width
        assert panel.region.bottom <= app.size.height
        assert app.screen.query_one("#memory-close").region.right <= app.size.width
        await pilot.click("#memory-close")


@pytest.mark.asyncio
async def test_model_error_remains_visible_after_transcript_reload(tui_state):
    from medtech.agent import LLMError

    store, backend = tui_state

    class UnavailableLLM(FakeLLM):
        def __call__(self, messages):
            if messages[0]["content"].startswith("Extract only"):
                return super().__call__(messages)
            raise LLMError("Cannot reach Ollama at local service. Run ollama serve.")

    app = MedTechApp(Config(), backend, store, llm=UnavailableLLM())
    async with app.run_test(size=(140, 42)) as pilot:
        await app.send_message("Keep this message despite the service error")
        await eventually(pilot, lambda: not app.busy)
        assert "Cannot reach Ollama" in visible_text(app)
        assert store.messages(app.sid)[0]["content"] == "Keep this message despite the service error"
        assert not app.query_one("#send").disabled


@pytest.mark.asyncio
async def test_streamed_draft_is_provisional_and_unsupported_claim_is_discarded(tui_state, monkeypatch):
    import medtech.agent as agent_module
    import medtech.tui as tui_module

    store, backend = tui_state
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def streamed_chat(cfg, messages, *, memory=False, on_delta=None, cancel_event=None):
        nonlocal calls
        if memory:
            return '{"facts":[]}'
        calls += 1
        if calls == 1:
            assert on_delta is not None
            on_delta("I created ")
            on_delta("folder phantom.")
            entered.set()
            release.wait(8)
            return json.dumps({"say": "I created folder phantom.", "action": None})
        on_delta("I have not created any files.")
        return json.dumps({"say": "I have not created any files.", "action": None})

    monkeypatch.setattr(agent_module, "chat", streamed_chat)
    monkeypatch.setattr(tui_module, "chat", streamed_chat)
    app = MedTechApp(Config(), backend, store)
    async with app.run_test(size=(140, 42)) as pilot:
        try:
            await app.send_message("Create folder phantom")
            await eventually(pilot, entered.is_set)
            assert app.query("#draft")
            assert "Generating — unverified" in visible_text(app)
            assert "I created folder phantom." in visible_text(app)
            assert not (Path(backend.root) / "phantom").exists()
        finally:
            release.set()
        await eventually(pilot, lambda: not app.busy)
        assert calls == 2
        assert not app.query("#draft")
        assistant_text = "\n".join(getattr(widget.renderable, "plain", str(widget.renderable))
                                   for widget in app.query(".assistant Static"))
        assert "I have not created any files." in assistant_text
        assert "I created folder phantom." not in assistant_text
        assert any(json.loads(row["metadata"]).get("accepted") is False for row in store.messages(app.sid))


@pytest.mark.asyncio
async def test_live_ollama_deltas_are_provisional_until_final_validation(tui_state, monkeypatch):
    """Real NDJSON parsing reaches the mounted UI before the final event arrives."""
    import io
    from medtech import agent
    store, backend = tui_state
    release = threading.Event()

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield b'{"message":{"content":"Hello "},"done":false}\n'
            release.wait(5)
            yield b'{"message":{"content":"Rayan."},"done":true}\n'

    def local_response(request, **kwargs):
        payload = json.loads(request.data)
        assert payload["model"] == "qwen2.5:7b"
        assert payload["options"]["num_ctx"] == 16384
        if "tools" in payload:
            assert payload["stream"] is True
            return Stream()
        return io.BytesIO(json.dumps({"message": {"content": '{"facts":[]}'}}).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", local_response)
    app = MedTechApp(Config(), backend, store)
    async with app.run_test(size=(120, 36)) as pilot:
        try:
            await app.send_message("Say hello to Rayan")
            await eventually(pilot, lambda: bool(app.query("#draft")))
            await eventually(pilot, lambda: "Hello" in visible_text(app))
            assert app.busy
            assert "Generating — unverified" in visible_text(app)
            assert "Hello" in visible_text(app)
            assert not any(row["role"] == "assistant" for row in store.messages(app.sid))
        finally:
            release.set()
        await eventually(pilot, lambda: not app.busy)
        assert not app.query("#draft")
        assert "Hello Rayan." in visible_text(app)
        assert json.loads(store.messages(app.sid)[-1]["content"])["say"] == "Hello Rayan."


@pytest.mark.asyncio
async def test_background_fact_updates_keep_user_turn_order(tui_state):
    """A slow older extraction cannot overwrite a newer name from a later chat."""
    store, backend = tui_state
    first_started, release = threading.Event(), threading.Event()

    class NameLLM(FakeLLM):
        def __call__(self, messages):
            if messages[0]["content"].startswith("Extract only"):
                text = messages[-1]["content"]
                if "Rayan" in text:
                    first_started.set()
                    release.wait(5)
                name = "Rayan" if "Rayan" in text else "Ray"
                return json.dumps({"facts": [{"key": "name", "value": name}]})
            return json.dumps({"say": "Acknowledged.", "action": None})

    app = MedTechApp(Config(), backend, store, llm=NameLLM())
    async with app.run_test(size=(120, 36)) as pilot:
        try:
            await app.send_message("My name is Rayan")
            await eventually(pilot, lambda: first_started.is_set() and not app.busy)
            await pilot.press("ctrl+n")
            await app.send_message("My name is Ray")
            await eventually(pilot, lambda: not app.busy)
        finally:
            release.set()
        await eventually(pilot, lambda: app._memory_active == 0)
        assert [(fact["key"], fact["value"]) for fact in store.memories()] == [("name", "Ray")]


@pytest.mark.asyncio
async def test_history_debug_shortcuts_and_quit_keep_working(tui_state):
    """Secondary slash commands retain visible IDs and work without the model."""
    store, backend = tui_state
    sid = add_chat(store, backend, "Shortcut chat")
    app = MedTechApp(Config(), backend, store, sid=sid, llm=FakeLLM())
    async with app.run_test(size=(120, 36)) as pilot:
        await app.send_message("/history")
        assert f"{sid} · Shortcut chat" in visible_text(app)
        await app.send_message("/debug on")
        assert app.debug_enabled
        await app.send_message("/debug")
        assert "debug.log" in visible_text(app)
        await app.send_message("/debug off")
        assert not app.debug_enabled
        await app.send_message("exit")
        assert app._ui_closing
    assert len(store.messages(sid)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remember", "forget", "clear", "modal"])
async def test_explicit_memory_edits_override_pending_extraction(tui_state, operation):
    """A delayed model candidate must not undo a newer explicit memory decision."""
    store, backend = tui_state
    store.remember("name", "Rayan")
    fact_id = store.memories()[0]["id"]
    started, release = threading.Event(), threading.Event()

    class DelayedFacts(FakeLLM):
        def __call__(self, messages):
            if messages[0]["content"].startswith("Extract only"):
                started.set()
                release.wait(5)
                return '{"facts":[{"key":"name","value":"Rayan"}]}'
            return super().__call__(messages)

    app = MedTechApp(Config(), backend, store, llm=DelayedFacts())
    async with app.run_test(size=(120, 36)) as pilot:
        try:
            await app.send_message("My name is Rayan")
            await eventually(pilot, lambda: started.is_set() and not app.busy)
            if operation == "modal":
                await pilot.click("#memory")
                await pilot.pause()
                await pilot.click(f"#forget-{fact_id}")
            else:
                command = {"remember": "/remember name=Ray", "forget": "/forget name", "clear": "/memory clear"}[operation]
                await app.send_message(command)
        finally:
            release.set()
        await eventually(pilot, lambda: app._memory_active == 0)
        expected = [("name", "Ray")] if operation == "remember" else []
        assert [(fact["key"], fact["value"]) for fact in store.memories()] == expected
