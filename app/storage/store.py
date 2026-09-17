"""SQLite-backed persistent state store for audit-critical data.

Ensures amendment history, telemetry, and audit trail survive restarts.
Uses WAL mode for concurrent read/write safety.

Phase 6 hardening:
- Schema versioning: the store records a schema version and refuses to
  boot against a newer (or unreadable) schema rather than mis-reading it.
- Explicit errors: corrupt rows surface as ``StoreIntegrityError`` instead
  of silently producing ``None`` or crashing later far from the cause.
- Atomic batches: ``save_many`` writes multiple rows in one transaction.
- Fail startup on corruption: ``load_all``/``load`` raise on malformed rows
  instead of skipping them.
"""
import json
import os
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1


class StateStoreError(Exception):
    """Base class for StateStore failures."""


class SchemaVersionError(StateStoreError):
    """The on-disk schema is incompatible with this build."""


class StoreIntegrityError(StateStoreError):
    """A stored row is corrupt (unparseable JSON, wrong shape, etc.)."""


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
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                else:
                    try:
                        stored = int(row[0])
                    except (TypeError, ValueError):
                        raise SchemaVersionError(
                            f"Unreadable schema version {row[0]!r} in {self.path}"
                        )
                    if stored > SCHEMA_VERSION:
                        raise SchemaVersionError(
                            f"Database schema {stored} is newer than supported "
                            f"{SCHEMA_VERSION} at {self.path}; refusing to boot"
                        )
                conn.commit()
            finally:
                conn.close()

    def _parse_payload(self, kind: str, key: str, raw: str) -> Dict:
        try:
            value = json.loads(raw)
        except (ValueError, TypeError) as e:
            raise StoreIntegrityError(
                f"Corrupt row in '{kind}' key '{key}': invalid JSON ({e})"
            ) from e
        if not isinstance(value, dict):
            raise StoreIntegrityError(
                f"Corrupt row in '{kind}' key '{key}': expected a JSON object, got {type(value).__name__}"
            )
        return value

    def save(self, kind: str, key: str, payload: Any):
        """Persist a JSON-serializable blob under (kind, key)."""
        try:
            encoded = json.dumps(payload)
        except (TypeError, ValueError) as e:
            raise StateStoreError(
                f"Payload for '{kind}' key '{key}' is not JSON-serializable"
            ) from e
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                    (kind, key, encoded),
                )
                conn.commit()
            finally:
                conn.close()

    def save_many(self, rows: Sequence[Tuple[str, str, Any]]):
        """Persist multiple (kind, key, payload) rows in a single transaction.

        Either every row is written or none is: callers doing multi-key state
        updates (e.g. runtime + current_version) must use this to stay atomic.
        """
        if not rows:
            return
        encoded_rows = []
        for kind, key, payload in rows:
            try:
                encoded = json.dumps(payload)
            except (TypeError, ValueError) as e:
                raise StateStoreError(
                    f"Payload for '{kind}' key '{key}' is not JSON-serializable"
                ) from e
            encoded_rows.append((kind, key, encoded))
        with self._lock:
            conn = self._conn()
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                    encoded_rows,
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
                return self._parse_payload(kind, key, row[0]) if row else None
            finally:
                conn.close()

    def load_all(self, kind: str) -> Dict[str, Dict]:
        """Load all blobs for a given kind. Returns {key: payload_dict}.

        Corrupt rows raise ``StoreIntegrityError`` so startup fails loudly
        instead of silently dropping governance state.
        """
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT key, payload FROM kv WHERE kind = ? ORDER BY key",
                    (kind,),
                ).fetchall()
                return {k: self._parse_payload(kind, k, p) for k, p in rows}
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