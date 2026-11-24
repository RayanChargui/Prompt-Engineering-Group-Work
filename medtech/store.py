"""Persistent storage (one SQLite file): sessions, chat messages, tasks, RAG index."""
import json
import os
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, mode TEXT NOT NULL, workspace TEXT, title TEXT, started_at REAL);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, ts REAL);
CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  ts REAL, tool TEXT, args TEXT, risk TEXT, status TEXT, output TEXT);
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT, sha TEXT, added_at REAL);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
  idx INTEGER, page INTEGER, text TEXT, emb BLOB);
"""


def home():
    p = Path(os.environ.get("MEDTECH_HOME", Path.home() / ".medtech"))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Store:
    def __init__(self, path=None):
        self.conn = sqlite3.connect(str(path or home() / "medtech.db"))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def _q(self, sql, *a):
        return self.conn.execute(sql, a).fetchall()

    def _w(self, sql, *a):
        cur = self.conn.execute(sql, a)
        self.conn.commit()
        return cur.lastrowid

    def new_session(self, mode, workspace):
        return self._w("INSERT INTO sessions(mode,workspace,started_at) VALUES(?,?,?)",
                       mode, workspace, time.time())

    def session(self, sid):
        r = self._q("SELECT * FROM sessions WHERE id=?", sid)
        return r[0] if r else None

    def add_message(self, sid, role, content):
        self._w("INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                sid, role, content, time.time())
        if role == "user":
            self._w("UPDATE sessions SET title=? WHERE id=? AND title IS NULL", content[:60], sid)

    def messages(self, sid):
        return self._q("SELECT * FROM messages WHERE session_id=? ORDER BY id", sid)

    def add_task(self, sid, tool, args, risk, status, output=""):
        return self._w("INSERT INTO tasks(session_id,ts,tool,args,risk,status,output) VALUES(?,?,?,?,?,?,?)",
                       sid, time.time(), tool, json.dumps(args, ensure_ascii=False)[:2000],
                       risk, status, output[:2000])

    def tasks(self, sid=None, limit=200):
        if sid is None:
            return self._q("SELECT t.*, s.mode FROM tasks t JOIN sessions s ON s.id=t.session_id "
                           "ORDER BY t.id DESC LIMIT ?", limit)
        return self._q("SELECT t.*, s.mode FROM tasks t JOIN sessions s ON s.id=t.session_id "
                       "WHERE t.session_id=? ORDER BY t.id", sid)

    def sessions(self, limit=50):
        return self._q(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n_msgs, "
            "(SELECT COUNT(*) FROM tasks t WHERE t.session_id=s.id) AS n_tasks "
            "FROM sessions s ORDER BY s.id DESC LIMIT ?", limit)
