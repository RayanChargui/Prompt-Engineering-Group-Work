"""Persistence, migration, and defensive user-memory extraction tests."""
import json
import sqlite3

import pytest

from medtech.memory import extract_memories, memory_prompt
from medtech.store import SCHEMA_VERSION, Store, StoreError


def test_restart_preserves_every_message_and_metadata(tmp_path):
    path = tmp_path / "history.db"
    store = Store(path)
    sid = store.new_session("local", str(tmp_path))
    store.add_message(sid, "user", "  First\nmessage  ")
    store.add_message(sid, "assistant", '{"action":{"tool":"mkdir"}}', {"kind": "tool_call"})
    store.add_message(sid, "tool", '{"success":true}', {"tool": "mkdir"})
    # A concurrent connection sees committed messages before the original closes.
    other = Store(path)
    assert len(other.messages(sid)) == 3
    other.close()
    store.close()
    reopened = Store(path)
    assert reopened.session(sid)["title"] == "First message"
    assert json.loads(reopened.messages(sid)[-1]["metadata"]) == {"tool": "mkdir"}
    assert reopened.session(sid)["updated_at"] >= reopened.session(sid)["created_at"]


def test_session_management_and_cascade(tmp_path):
    store = Store(":memory:")
    first = store.new_session("local", str(tmp_path))
    second = store.new_session("local", str(tmp_path))
    assert store.latest_session()["id"] == second
    assert store.rename_session(first, "Renamed")
    store.add_message(first, "user", "Do not overwrite my title")
    store.add_task(first, "mkdir", {"path": "x"}, "write", "done", {"success": True})
    assert store.sessions()[0]["id"] == first
    assert store.session(first)["title"] == "Renamed"
    store.save_summary(first, "Earlier conversation")
    assert store.session(first)["summary"] == "Earlier conversation"
    assert store.delete_session(first)
    assert store.messages(first) == store.tasks(first) == []
    assert not store.delete_session(first)
    with pytest.raises(ValueError):
        store.rename_session(second, " ")


def test_migrate_original_database_without_losing_conversations(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE sessions(id INTEGER PRIMARY KEY,mode TEXT NOT NULL,workspace TEXT,title TEXT,started_at REAL);
    CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id INTEGER,role TEXT,content TEXT,ts REAL);
    INSERT INTO sessions VALUES(1,'local','/tmp','Original',100);
    INSERT INTO messages VALUES(1,1,'user','hello',101);
    """)
    conn.close()
    store = Store(path)
    assert store.session(1)["created_at"] == 100
    assert store.session(1)["updated_at"] == 101
    assert store.messages(1)[0]["content"] == "hello"
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    store.close()
    assert len(Store(path).messages(1)) == 1


def test_corrupted_database_is_not_overwritten(tmp_path):
    path = tmp_path / "broken.db"
    content = b"this is not a sqlite database"
    path.write_bytes(content)
    with pytest.raises(StoreError, match="Cannot open database"):
        Store(path)
    assert path.read_bytes() == content


def test_future_schema_is_not_downgraded(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version=999")
    conn.close()
    with pytest.raises(StoreError, match="newer"):
        Store(path)


def test_memory_survives_restart_and_replaces_contradiction(tmp_path):
    path = tmp_path / "facts.db"
    store = Store(path)
    assert extract_memories(store, "my name is Rayan", lambda _: '{"facts":[{"key":"name","value":"Rayan"}]}')
    store.close()
    store = Store(path)
    store.new_session("local", str(tmp_path))
    assert "Rayan" in memory_prompt(store)
    assert extract_memories(store, "my name is Ray", lambda _: '{"facts":[{"key":"Name","value":"Ray"}]}')
    assert [(m["key"], m["value"]) for m in store.memories()] == [("name", "Ray")]
    store.remember("my name is Rayan")
    assert len(store.memories()) == 1
    assert store.memories()[0]["value"] == "Rayan"


@pytest.mark.parametrize("response", ["oops", "[]", '{"facts":{}}', '{"facts":[null]}',
                                      '{"facts":[{"key":"name","value":5}]}',
                                      '{"facts":[{"key":"name","value":"Ray"},{"key":"bad"}]}',
                                      '{"facts":[],"extra":true}'])
def test_malformed_extraction_has_no_partial_writes(response):
    store = Store(":memory:")
    store.remember("name", "Rayan")
    assert not extract_memories(store, "my name changed", lambda _: response)
    assert store.memories()[0]["value"] == "Rayan"


def test_extraction_failure_and_memory_commands():
    store = Store(":memory:")
    def unavailable(_):
        raise ConnectionError("offline")
    assert not extract_memories(store, "hello", unavailable)
    store.remember("I prefer Python")
    store.remember("I prefer Python")
    assert len(store.memories()) == 1
    assert store.forget_memory("I prefer Python") == 1
    store.remember("name=Rayan")
    assert store.forget_memory(str(store.memories()[0]["id"])) == 1
    store.remember("name=Rayan")
    assert store.forget_memory("name") == 1
    store.remember("name=Rayan")
    store.clear_memories()
    assert store.memories() == []
