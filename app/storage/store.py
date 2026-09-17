"""SQLite-backed persistent state store for audit-critical data.

Ensures amendment history, telemetry, and audit trail survive restarts.
Uses WAL mode for concurrent read/write safety.
"""
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class StateStore:
    """SQLite-backed persistent JSON key-value store.

    Tables are namespaced by 'kind' (amendment, telemetry, audit, runtime, etc.)
    Each kind stores JSON blobs keyed by string ID.

    Thread-safe via a lock; suitable for the single-process FastAPI server.
    """

    def __init__(self, path: str = "data/evolving.db"):
        self.path = path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS kv (
                        kind TEXT NOT NULL,
                        key TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (kind, key)
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def save(self, kind: str, key: str, payload: Any):
        """Persist a JSON-serializable blob under (kind, key)."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                    (kind, key, json.dumps(payload)),
                )
                conn.commit()
            finally:
                conn.close()

    def load(self, kind: str, key: str) -> Optional[Dict]:
        """Load a single blob by (kind, key)."""
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT payload FROM kv WHERE kind = ? AND key = ?",
                    (kind, key),
                ).fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()

    def load_all(self, kind: str) -> Dict[str, Dict]:
        """Load all blobs for a given kind. Returns {key: payload_dict}."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT key, payload FROM kv WHERE kind = ?",
                    (kind,),
                ).fetchall()
                return {k: json.loads(p) for k, p in rows}
            finally:
                conn.close()

    def delete(self, kind: str, key: str):
        """Delete a single blob."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "DELETE FROM kv WHERE kind = ? AND key = ?",
                    (kind, key),
                )
                conn.commit()
            finally:
                conn.close()

    def count(self, kind: str) -> int:
        """Count entries of a given kind."""
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM kv WHERE kind = ?",
                    (kind,),
                ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
