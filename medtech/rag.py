"""Local RAG: ingest -> chunk -> embed (Ollama) -> SQLite -> cosine search."""
import hashlib
import heapq
import math
import os
import re
import time
from array import array
from dataclasses import dataclass
from operator import mul
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".rst", ".csv", ".json", ".log", ".py"}
SUPPORTED = TEXT_EXT | {".pdf"}
MIN_SCORE = float(os.environ.get("MEDTECH_MIN_SCORE", "0.45"))


@dataclass
class Hit:
    source: str
    page: object
    text: str
    score: float


def chunk(text, size=900, overlap=150):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        if j < len(text):
            cut = max(text.rfind("\n\n", i, j), text.rfind(". ", i, j))
            if cut > i + size // 2:
                j = cut + 1
        piece = text[i:j].strip()
        if piece:
            out.append(piece)
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


def load(path):
    """-> list of (page_number_or_None, text)"""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
        if not any(t.strip() for _, t in pages):
            raise ValueError("no extractable text (scanned PDF?)")
        return pages
    return [(None, Path(path).read_text(encoding="utf-8", errors="ignore"))]


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return array("f", [x / n for x in v])


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class RAG:
    def __init__(self, store, embed):
        """embed(texts: list[str], kind: 'document'|'query') -> list[list[float]]"""
        self.db, self.embed = store, embed

    def ingest(self, path):
        """Ingest a file or folder. Returns list of (name, status)."""
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"not found: {p}")
        files = [p] if p.is_file() else sorted(
            f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED)[:500]
        return [(f.name, self._ingest_file(f)) for f in files]

    def _ingest_file(self, f):
        if f.suffix.lower() not in SUPPORTED:
            return "skipped (unsupported type)"
        path, sha = str(f.resolve()), _sha(f)
        old = self.db._q("SELECT id, sha FROM docs WHERE path=?", path)
        if old and old[0]["sha"] == sha:
            return "already up to date"
        pieces = [(pg, c) for pg, t in load(f) for c in chunk(t)]
        if not pieces:
            return "skipped (empty)"
        vecs = []
        for i in range(0, len(pieces), 16):
            vecs += self.embed([c for _, c in pieces[i:i + 16]], "document")
        if old:
            self.db._w("DELETE FROM docs WHERE id=?", old[0]["id"])
        did = self.db._w("INSERT INTO docs(path,name,sha,added_at) VALUES(?,?,?,?)",
                         path, f.name, sha, time.time())
        self.db.conn.executemany(
            "INSERT INTO chunks(doc_id,idx,page,text,emb) VALUES(?,?,?,?,?)",
            [(did, n, pg, c, _norm(v).tobytes()) for n, ((pg, c), v) in enumerate(zip(pieces, vecs))])
        self.db.conn.commit()
        return f"{len(pieces)} chunks"

    def search(self, query, k=4, min_score=None):
        rows = self.db._q("SELECT d.name, c.page, c.text, c.emb FROM chunks c JOIN docs d ON d.id=c.doc_id")
        if not rows:
            return []
        q = _norm(self.embed([query], "query")[0])
        scored = []
        for r in rows:
            e = array("f")
            e.frombytes(r["emb"])
            scored.append((sum(map(mul, q, e)), r))
        floor = MIN_SCORE if min_score is None else min_score
        top = heapq.nlargest(k, scored, key=lambda x: x[0])
        return [Hit(r["name"], r["page"], r["text"], s) for s, r in top if s >= floor]

    def docs(self):
        return self.db._q("SELECT d.id, d.name, d.path, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) AS n "
                          "FROM docs d ORDER BY d.id")

    def forget(self, doc_id):
        self.db._w("DELETE FROM docs WHERE id=?", doc_id)
