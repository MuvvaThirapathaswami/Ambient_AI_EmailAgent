"""
memory.py - Persistent memory layer using SQLite.

Stores and retrieves user preferences and feedback so the agent
can learn and adapt across separate email sessions.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


DB_PATH = Path("agent_memory.db")


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure tables exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            key       TEXT    NOT NULL UNIQUE,
            value     TEXT    NOT NULL,
            updated_at TEXT   NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interaction_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id     TEXT,
            triage       TEXT,
            draft        TEXT,
            human_action TEXT,
            human_edit   TEXT,
            created_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ──────────────────────────────────────────────
# Preference helpers
# ──────────────────────────────────────────────

def save_preference(key: str, value: str) -> None:
    """Upsert a single user preference."""
    conn = _get_connection()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO preferences (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """, (key, value, now))
    conn.commit()
    conn.close()


def load_all_preferences() -> str:
    """Return all stored preferences as a formatted string for the LLM prompt."""
    conn = _get_connection()
    rows = conn.execute("SELECT key, value FROM preferences").fetchall()
    conn.close()

    if not rows:
        return "No stored preferences yet."

    lines = ["User Preferences & Learned Behaviours:"]
    for row in rows:
        lines.append(f"  - {row['key']}: {row['value']}")
    return "\n".join(lines)


def extract_and_save_preferences_from_edit(original_draft: str, human_edit: str) -> None:
    """
    Lightweight heuristic: when the human edits the draft, save a
    preference that captures the correction so the agent avoids the
    same mistake next time.

    In production you'd call an LLM here; for now we store the raw
    correction with a timestamp-based key.
    """
    key = f"correction_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    value = f"Human changed '{original_draft[:80]}...' → '{human_edit[:80]}...'"
    save_preference(key, value)


# ──────────────────────────────────────────────
# Interaction log
# ──────────────────────────────────────────────

def log_interaction(
    email_id: str,
    triage: str,
    draft: str | None,
    human_action: str | None,
    human_edit: str | None,
) -> None:
    """Persist a record of what happened for a given email."""
    conn = _get_connection()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO interaction_log
            (email_id, triage, draft, human_action, human_edit, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (email_id, triage, draft, human_action, human_edit, now))
    conn.commit()
    conn.close()


def get_interaction_history(limit: int = 20) -> list[dict]:
    """Retrieve the N most recent interaction records."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT * FROM interaction_log ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
