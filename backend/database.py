"""
Long-term memory — SQLite-backed session & message storage.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            has_image   INTEGER DEFAULT 0,
            model_used  TEXT,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        """)


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(session_id: str, name: str) -> dict:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, name, now, now),
        )
    return {"id": session_id, "name": name, "created_at": now, "updated_at": now}


def get_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def update_session_name(session_id: str, name: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
            (name, datetime.utcnow().isoformat(), session_id),
        )


def touch_session(session_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )


def delete_session(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(
    session_id: str,
    role: str,
    content: str,
    has_image: bool = False,
    model_used: str | None = None,
):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, has_image, model_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, int(has_image), model_used, now),
        )
    touch_session(session_id)


def get_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]
