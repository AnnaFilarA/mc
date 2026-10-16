"""Хранилище MVP (SPEC 3.1): SQLite `data/attrguard.db` + JSONL разметки `data/feedback/feedback.jsonl`. Владелец — поток C.

Таблицы: cards (демо-список и карточки продавца), extract_cache (идемпотентность F16),
feedback (решения менеджера и ответы продавца, F14), seller_reads (отметка «прочитан»).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path

from src.contracts import Card, ExtractResponse

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    product_id TEXT PRIMARY KEY,
    card_json  TEXT NOT NULL,
    owner      TEXT NOT NULL DEFAULT 'demo',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS extract_cache (
    product_id    TEXT NOT NULL,
    text_hash     TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (product_id, text_hash)
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    product_id TEXT NOT NULL,
    attribute  TEXT NOT NULL,
    action     TEXT NOT NULL,
    value      TEXT,
    actor      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_product ON feedback (product_id);
CREATE TABLE IF NOT EXISTS seller_reads (
    product_id TEXT PRIMARY KEY,
    read_at    TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: str | Path | None = None, feedback_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else ROOT / "data" / "attrguard.db"
        self.feedback_path = Path(feedback_path) if feedback_path else ROOT / "data" / "feedback" / "feedback.jsonl"
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # --- карточки -------------------------------------------------------
    def save_card(self, card: Card, owner: str = "demo", replace: bool = True) -> None:
        sql = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        with self._lock, closing(self._conn()) as c, c:
            c.execute(
                f"{sql} INTO cards (product_id, card_json, owner, created_at) VALUES (?, ?, ?, ?)",
                (card.product_id, card.model_dump_json(), owner, now_iso()),
            )

    def get_card(self, product_id: str) -> tuple[Card, str, str] | None:
        with closing(self._conn()) as c:
            r = c.execute("SELECT * FROM cards WHERE product_id = ?", (product_id,)).fetchone()
        return (Card.model_validate_json(r["card_json"]), r["owner"], r["created_at"]) if r else None

    def list_cards(self) -> list[tuple[Card, str, str]]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT * FROM cards ORDER BY created_at DESC, rowid DESC").fetchall()
        return [(Card.model_validate_json(r["card_json"]), r["owner"], r["created_at"]) for r in rows]

    # --- кэш извлечения (F16) -------------------------------------------
    def get_cached(self, product_id: str, text_hash: str) -> ExtractResponse | None:
        with closing(self._conn()) as c:
            r = c.execute(
                "SELECT response_json FROM extract_cache WHERE product_id = ? AND text_hash = ?",
                (product_id, text_hash),
            ).fetchone()
        return ExtractResponse.model_validate_json(r["response_json"]) if r else None

    def put_cached(self, product_id: str, text_hash: str, resp: ExtractResponse) -> None:
        with self._lock, closing(self._conn()) as c, c:
            c.execute(
                "INSERT OR REPLACE INTO extract_cache (product_id, text_hash, response_json, created_at) VALUES (?, ?, ?, ?)",
                (product_id, text_hash, resp.model_dump_json(), now_iso()),
            )

    # --- разметка (F14) -------------------------------------------------
    def add_feedback(self, product_id: str, attribute: str, action: str, value: str | None, actor: str,
                     extra: dict | None = None) -> dict:
        ts = now_iso()
        with self._lock:
            with closing(self._conn()) as c, c:
                cur = c.execute(
                    "INSERT INTO feedback (ts, product_id, attribute, action, value, actor) VALUES (?, ?, ?, ?, ?, ?)",
                    (ts, product_id, attribute, action, value, actor),
                )
                fid = cur.lastrowid
            row = {"id": fid, "ts": ts, "product_id": product_id, "attribute": attribute, "action": action,
                   "value": value, "actor": actor, **(extra or {})}
            with self.feedback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def feedback_for(self, product_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM feedback", ()
        if product_id is not None:
            sql, args = sql + " WHERE product_id = ?", (product_id,)
        with closing(self._conn()) as c:
            rows = c.execute(sql + " ORDER BY id", args).fetchall()
        return [dict(r) for r in rows]

    # --- «прочитан» -----------------------------------------------------
    def mark_read(self, product_id: str) -> None:
        with self._lock, closing(self._conn()) as c, c:
            c.execute("INSERT OR IGNORE INTO seller_reads (product_id, read_at) VALUES (?, ?)", (product_id, now_iso()))

    def reads(self) -> dict[str, str]:
        with closing(self._conn()) as c:
            return {r["product_id"]: r["read_at"] for r in c.execute("SELECT * FROM seller_reads")}
