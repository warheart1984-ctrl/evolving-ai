"""P6: persistence hardening — schema versioning, explicit errors, atomic batches."""
import sqlite3

import pytest

from app.storage.store import (
    SCHEMA_VERSION,
    SchemaVersionError,
    StateStore,
    StateStoreError,
    StoreIntegrityError,
)


class TestSchemaVersioning:
    def test_new_store_writes_current_schema_version(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        conn = sqlite3.connect(str(tmp_path / "evolving.db"))
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        finally:
            conn.close()
        assert int(row[0]) == SCHEMA_VERSION

    def test_restart_with_matching_version_ok(self, tmp_path):
        StateStore(str(tmp_path / "evolving.db"))
        # Same schema -> second boot fine.
        StateStore(str(tmp_path / "evolving.db"))

    def test_newer_schema_refuses_boot(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        StateStore(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 100),))
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(SchemaVersionError):
            StateStore(db)

    def test_unreadable_schema_refuses_boot(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        StateStore(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", ("garbage",))
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(SchemaVersionError):
            StateStore(db)


class TestExplicitErrors:
    def test_corrupt_json_raises_on_load(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        store = StateStore(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                ("runtime", "runtime-v0", "{not-json"),
            )
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(StoreIntegrityError):
            store.load("runtime", "runtime-v0")
        with pytest.raises(StoreIntegrityError):
            store.load_all("runtime")

    def test_non_object_payload_raises(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        store = StateStore(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                ("telemetry", "run-1", "\"just-a-string\""),
            )
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(StoreIntegrityError):
            store.load("telemetry", "run-1")
        with pytest.raises(StoreIntegrityError):
            store.load_all("telemetry")

    def test_nonserializable_payload_rejected_on_save(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        with pytest.raises(StateStoreError):
            store.save("runtime", "bad", {"blob": object()})


class TestFailStartupOnCorruption:
    def _corrupt(self, db: str, kind: str, key: str, payload: str):
        store = StateStore(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv (kind, key, payload) VALUES (?, ?, ?)",
                (kind, key, payload),
            )
            conn.commit()
        finally:
            conn.close()
        return store

    def test_registry_startup_raises_on_corrupt_runtime(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        self._corrupt(db, "runtime", "runtime-v0", "{corrupt-json")
        from app.governance.registry import RuntimeRegistry
        with pytest.raises(StoreIntegrityError):
            RuntimeRegistry(persistence=StateStore(db))

    def test_telemetry_startup_raises_on_corrupt_trusted_telemetry(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        self._corrupt(db, "telemetry", "run-000001", "[1,2,3]")
        from app.telemetry._init import TelemetryStore
        with pytest.raises(StoreIntegrityError):
            TelemetryStore(persistence=StateStore(db))

    def test_amendment_startup_raises_on_invalid_row(self, tmp_path):
        """Invalid governance rows surface explicitly, never silently skipped."""
        db = str(tmp_path / "evolving.db")
        self._corrupt(db, "amendment", "prop-1", "{\"not\":\"an amendment\"}")
        from app.api import main as main

        original_store = main.store
        original_amendments = main._amendments_raw
        try:
            main.store = StateStore(db)
            main._amendments_raw = main.store.load_all("amendment")
            with pytest.raises(Exception):
                {k: main.Amendment(**v) for k, v in main._amendments_raw.items()}
        finally:
            main.store = original_store
            main._amendments_raw = original_amendments


class TestAtomicBatches:
    def test_save_many_writes_all_rows(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        store = StateStore(db)
        store.save_many([
            ("meta", "current_version", {"value": "v1"}),
            ("runtime", "runtime-v1", {"id": "runtime-v1", "version": "v1"}),
        ])
        store2 = StateStore(db)
        assert store2.load("meta", "current_version") == {"value": "v1"}
        assert store2.load("runtime", "runtime-v1")["version"] == "v1"

    def test_save_many_empty_noop(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        store.save_many([])
        assert store.count("runtime") == 0

    def test_save_many_validates_before_writing(self, tmp_path):
        db = str(tmp_path / "evolving.db")
        store = StateStore(db)
        with pytest.raises(StateStoreError):
            store.save_many([
                ("meta", "current_version", {"value": "v1"}),
                ("runtime", "x", {"blob": object()}),
            ])
        # Nothing was written (validation happens before any row is persisted).
        assert store.count("runtime") == 0
        assert len(store.load_all("meta")) == 0