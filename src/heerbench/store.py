"""Durable single-orchestrator journal and content-addressed artifacts.

No automatic recovery of ambiguous submissions. A run-directory lock prevents
two orchestrators from spending on the same planned request concurrently.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import os

from .util import atomic_text, canonical, read_jsonl, write_jsonl


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "journal.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, plan TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          request TEXT, result TEXT, error TEXT, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL,
          task_id TEXT, event TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches (
          id TEXT PRIMARY KEY, model_id TEXT NOT NULL, kind TEXT NOT NULL,
          state TEXT NOT NULL, handle TEXT, task_ids TEXT NOT NULL, updated TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    @contextmanager
    def lock(self):
        # OS lock is released on process exit. The journal still records uncertain attempts.
        import fcntl
        with (self.root / "run.lock").open("a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another orchestrator owns this run directory") from exc
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def put_blob(self, value: str) -> str:
        key = hashlib.sha256(value.encode()).hexdigest()
        path = self.root / "blobs" / key[:2] / (key + ".txt")
        if not path.exists():
            atomic_text(path, value)
        return key

    def get_blob(self, key: str) -> str:
        value = (self.root / "blobs" / key[:2] / (key + ".txt")).read_text()
        if hashlib.sha256(value.encode()).hexdigest() != key:
            raise ValueError("Artifact hash mismatch")
        return value

    def pack_request(self, request: dict) -> dict:
        value = dict(request)
        for field in ("system", "prefix", "task"):
            value[field + "_blob"] = self.put_blob(value.pop(field))
        return value

    def unpack_request(self, packed: dict) -> dict:
        value = dict(packed)
        for field in ("system", "prefix", "task"):
            value[field] = self.get_blob(value.pop(field + "_blob"))
        return value

    def event(self, task_id: str | None, event: str, detail: dict):
        self.db.execute("INSERT INTO events(time,task_id,event,detail) VALUES(?,?,?,?)",
                        (now(), task_id, event, canonical(detail)))
        self.db.commit()

    def add(self, row: dict, kind: str = "generation"):
        existing = self.get(row["request_id"])
        if existing:
            if existing["plan"] != row or existing["kind"] != kind:
                raise ValueError("Existing request ID has a different frozen plan")
            return
        self.db.execute("INSERT INTO tasks(id,kind,plan,updated) VALUES(?,?,?,?)",
                        (row["request_id"], kind, canonical(row), now()))
        self.db.commit()

    def _decode(self, row):
        if row is None:
            return None
        d = dict(row)
        for key in ("plan", "request", "result", "error"):
            d[key] = json.loads(d[key]) if d[key] is not None else None
        return d

    def get(self, task_id: str):
        return self._decode(self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def rows(self, kind: str | None = None):
        rows = self.db.execute("SELECT * FROM tasks ORDER BY rowid") if kind is None else self.db.execute(
            "SELECT * FROM tasks WHERE kind=? ORDER BY rowid", (kind,))
        return [self._decode(x) for x in rows]

    def update(self, task_id: str, state: str, *, request=None, result=None, error=None, attempt=False):
        old = self.get(task_id)
        if old is None:
            raise KeyError(task_id)
        self.db.execute("UPDATE tasks SET state=?,request=?,result=?,error=?,attempts=?,updated=? WHERE id=?",
                        (state, canonical(request) if request is not None else canonical(old["request"]) if old["request"] else None,
                         canonical(result) if result is not None else canonical(old["result"]) if old["result"] else None,
                         canonical(error) if error is not None else None,
                         old["attempts"] + int(attempt), now(), task_id))
        self.db.commit()
        self.event(task_id, state, {"attempt": old["attempts"] + int(attempt), "error": error})

    def batch(self, batch_id: str, model_id: str, kind: str, state: str, ids: list[str], handle=None):
        self.db.execute("INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?)",
                        (batch_id, model_id, kind, state, canonical(handle) if handle else None,
                         canonical(ids), now()))
        self.db.commit()

    def batches(self, kind: str):
        return [{**dict(r), "handle": json.loads(r["handle"]) if r["handle"] else None,
                 "task_ids": json.loads(r["task_ids"])} for r in self.db.execute(
                     "SELECT * FROM batches WHERE kind=?", (kind,))]

    def export(self):
        write_jsonl(self.root / "events.jsonl", [dict(r) for r in self.db.execute("SELECT * FROM events ORDER BY seq")])
        write_jsonl(self.root / "generation.jsonl", [{**r["plan"], "generation_status": r["state"],
                     "attempts": r["attempts"], "request": r["request"], "response": r["result"],
                     "error": r["error"]} for r in self.rows("generation")])
        write_jsonl(self.root / "judgements.jsonl", [{**r["plan"], "judge_status": r["state"],
                     "attempts": r["attempts"], "response": r["result"], "error": r["error"]}
                    for r in self.rows("judge")])
