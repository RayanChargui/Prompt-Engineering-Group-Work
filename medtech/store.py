"""Versioned, durable SQLite conversations, tool audit records, and user facts."""
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, mode TEXT NOT NULL, workspace TEXT, title TEXT, started_at REAL,
  created_at REAL, updated_at REAL, summary TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, ts REAL, metadata TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  ts REAL, tool TEXT, args TEXT, risk TEXT, status TEXT, output TEXT);
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, value TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
"""


class StoreError(Exception):
    """A database could not be opened, migrated, or written safely."""


def home() -> Path:
    """Return the configured data directory, creating it if necessary."""
    p = Path(os.environ.get("MEDTECH_HOME", Path.home() / ".medtech")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def memory_key(key: str) -> str:
    """Normalize stable property names so updates replace earlier values."""
    return re.sub(r"[^\w]+", "_", key.casefold()).strip("_")[:80]


class Store:
    """Commit every message and its session metadata as one transaction."""

    def __init__(self, path: Optional[str] = None):
        self.conn = None
        try:
            self.path = str(path or home() / "medtech.db")
            self.conn = sqlite3.connect(self.path, timeout=10)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA synchronous=FULL")
            self._migrate()
        except StoreError:
            if self.conn is not None:
                self.conn.close()
            raise
        except (sqlite3.Error, OSError) as e:
            if self.conn is not None:
                self.conn.close()
            raise StoreError(f"Cannot open database {path or 'medtech.db'}: {e}. "
                             "Check permissions or restore a backup; the database was not replaced.") from e

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StoreError(f"Database version {version} is newer than supported version {SCHEMA_VERSION}.")
        with self._transaction():
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    self.conn.execute(statement)
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(sessions)")}
            for name, declaration in (("created_at", "REAL"), ("updated_at", "REAL"),
                                      ("summary", "TEXT NOT NULL DEFAULT ''")):
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {declaration}")
            if "metadata" not in {r[1] for r in self.conn.execute("PRAGMA table_info(messages)")}:
                self.conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            self.conn.execute("UPDATE sessions SET created_at=COALESCE(created_at,started_at,?), "
                              "updated_at=COALESCE(updated_at,(SELECT MAX(ts) FROM messages "
                              "WHERE session_id=sessions.id),started_at,?)", (time.time(), time.time()))
            # Remove the obsolete document index during upgrades; conversation tables stay intact.
            self.conn.execute("DROP TABLE IF EXISTS chunks")
            self.conn.execute("DROP TABLE IF EXISTS docs")
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            with self.conn:
                yield
        except sqlite3.Error as e:
            raise StoreError(f"Database operation failed: {e}. Check permissions or restore a backup.") from e

    def _q(self, sql: str, *args: Any) -> list:
        try:
            return self.conn.execute(sql, args).fetchall()
        except sqlite3.Error as e:
            raise StoreError(f"Database read failed: {e}. Check the database or restore a backup.") from e

    def _w(self, sql: str, *args: Any) -> int:
        with self._transaction():
            cur = self.conn.execute(sql, args)
        return cur.lastrowid

    def close(self) -> None:
        """Release the database connection; writes are already committed."""
        self.conn.close()

    def new_session(self, mode: str, workspace: str) -> int:
        now = time.time()
        return self._w("INSERT INTO sessions(mode,workspace,started_at,created_at,updated_at) VALUES(?,?,?,?,?)",
                       mode, workspace, now, now, now)

    def session(self, sid: int) -> Optional[sqlite3.Row]:
        rows = self._q("SELECT * FROM sessions WHERE id=?", sid)
        return rows[0] if rows else None

    def latest_session(self) -> Optional[sqlite3.Row]:
        rows = self._q("SELECT * FROM sessions ORDER BY updated_at DESC,id DESC LIMIT 1")
        return rows[0] if rows else None

    def add_message(self, sid: int, role: str, content: str, metadata: Optional[dict] = None) -> int:
        now = time.time()
        with self._transaction():
            cur = self.conn.execute("INSERT INTO messages(session_id,role,content,ts,metadata) VALUES(?,?,?,?,?)",
                                    (sid, role, content, now, json.dumps(metadata or {}, ensure_ascii=False)))
            self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
            if role == "user":
                title = " ".join(content.split())[:60] or "New chat"
                self.conn.execute("UPDATE sessions SET title=? WHERE id=? AND title IS NULL", (title, sid))
        return cur.lastrowid

    def messages(self, sid: int) -> list:
        return self._q("SELECT * FROM messages WHERE session_id=? ORDER BY id", sid)

    def rename_session(self, sid: int, title: str) -> bool:
        title = title.strip()[:200]
        if not title:
            raise ValueError("Session title cannot be empty")
        with self._transaction():
            cur = self.conn.execute("UPDATE sessions SET title=?,updated_at=? WHERE id=?", (title, time.time(), sid))
        return bool(cur.rowcount)

    def delete_session(self, sid: int) -> bool:
        with self._transaction():
            cur = self.conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        return bool(cur.rowcount)

    def save_summary(self, sid: int, summary: str) -> None:
        self._w("UPDATE sessions SET summary=? WHERE id=?", summary, sid)

    def add_task(self, sid: int, tool: str, args: dict, risk: str, status: str, output: Any = "") -> int:
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return self._w("INSERT INTO tasks(session_id,ts,tool,args,risk,status,output) VALUES(?,?,?,?,?,?,?)",
                       sid, time.time(), tool, json.dumps(args, ensure_ascii=False), risk, status, output)

    def tasks(self, sid: Optional[int] = None, limit: int = 200) -> list:
        if sid is None:
            return self._q("SELECT t.*, s.mode FROM tasks t JOIN sessions s ON s.id=t.session_id "
                           "ORDER BY t.id DESC LIMIT ?", limit)
        return self._q("SELECT t.*, s.mode FROM tasks t JOIN sessions s ON s.id=t.session_id "
                       "WHERE t.session_id=? ORDER BY t.id", sid)

    def sessions(self, limit: int = 50) -> list:
        return self._q(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n_msgs, "
            "(SELECT COUNT(*) FROM tasks t WHERE t.session_id=s.id) AS n_tasks "
            "FROM sessions s ORDER BY s.updated_at DESC,s.id DESC LIMIT ?", limit)

    def memories(self) -> list:
        """List durable facts independently of any conversation."""
        return self._q("SELECT * FROM memories ORDER BY id")

    def remember_many(self, facts: list) -> None:
        """Atomically save a validated set of stable key/value properties."""
        now = time.time()
        with self._transaction():
            for fact in facts:
                self.conn.execute("INSERT INTO memories(key,value,created_at,updated_at) VALUES(?,?,?,?) "
                                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                                  (memory_key(fact["key"]), fact["value"], now, now))

    def remember(self, key: str, value: Optional[str] = None) -> None:
        """Save key/value or a manual fact; recognize common named properties."""
        if value is None:
            fact = key.strip()
            match = re.fullmatch(r"(?:my\s+)?(name|location|timezone|project|preference)\s+(?:is\s+|=\s*)(.+)",
                                 fact, re.I)
            if match:
                key, value = match.groups()
            elif "=" in fact:
                key, value = fact.split("=", 1)
            else:
                key, value = "fact_" + hashlib.sha256(fact.casefold().encode()).hexdigest()[:16], fact
        key, value = memory_key(key), value.strip()
        if not key or not value or len(value) > 500:
            raise ValueError("Memory needs a nonempty property and a value of at most 500 characters")
        self.remember_many([{"key": key, "value": value}])

    def forget_memory(self, id_or_text: str) -> int:
        """Remove a fact by id, exact property, or exact value (case insensitive)."""
        text = str(id_or_text).strip()
        with self._transaction():
            if text.isdigit():
                cur = self.conn.execute("DELETE FROM memories WHERE id=?", (int(text),))
            else:
                cur = self.conn.execute("DELETE FROM memories WHERE key=? OR value=? COLLATE NOCASE",
                                        (memory_key(text), text))
        return cur.rowcount

    def clear_memories(self) -> None:
        self._w("DELETE FROM memories")
