"""P3: telemetry trustworthiness — auth, hashing, untrusted isolation, limits."""
import os

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, registry, telemetry_store
from app.evaluation._init import FailureClassRegistry
from app.operator.operator import Operator
from app.governance.registry import RuntimeRegistry
from app.telemetry._init import (
    ExecutionTelemetry,
    TelemetryStore,
)


@pytest.fixture()
def client():
    os.environ["APP_ENV"] = "development"
    return TestClient(app)


@pytest.fixture()
def v0():
    current = registry.get_current()
    if current is None:
        registry.create_runtime(
            version="v0", model_identifier="gpt-3.5-turbo", constitution_version="v1",
            created_by="system", description="v0",
        )
        current = registry.get_current()
    return current


def _telemetry_dict(runtime_version="v0", manifest_hash_val=None, **overrides):
    data = {
        "source": "operator",
        "runtime_version": runtime_version,
        "runtime_manifest_hash": manifest_hash_val,
        "task_id": "math-001",
        "input": {"expression": "2+2"},
        "output": "5",
        "success": False,
        "errors": ["Incorrect answer"],
        "tools_used": [],
        "latency_ms": 10.0,
        "cost": 0.0,
    }
    data.update(overrides)
    return data


class TestTelemetryAuth:
    def test_record_requires_api_key(self, client, v0):
        resp = client.post("/telemetry/record", json=_telemetry_dict())
        assert resp.status_code == 401

    def test_record_rejects_wrong_key(self, client, v0):
        resp = client.post(
            "/telemetry/record", json=_telemetry_dict(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_record_accepts_config_key(self, client, v0):
        resp = client.post(
            "/telemetry/record", json=_telemetry_dict(),
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 200


class TestTelemetryTrustScoring:
    def test_untrusted_record_isolated_from_aggregation(self, client, v0):
        """No matching manifest hash => untrusted; never appears in failure classes."""
        before = telemetry_store.get_failure_classes("v0")
        resp = client.post(
            "/telemetry/record", json=_telemetry_dict(manifest_hash_val=None),
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["trusted"] is False

        assert telemetry_store.get_failure_classes("v0") == before
        untrusted = telemetry_store.get_untrusted_records()
        assert any(r.task_id == "math-001" for r in untrusted)

    def test_matching_manifest_hash_is_trusted(self, client, v0):
        hm = v0.manifest_hash
        resp = client.post(
            "/telemetry/record",
            json=_telemetry_dict(manifest_hash_val=hm),
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["trusted"] is True

    def test_fake_hash_not_trusted(self, client, v0):
        resp = client.post(
            "/telemetry/record",
            json=_telemetry_dict(manifest_hash_val="0" * 64),
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["trusted"] is False

    def test_trusted_records_feed_failure_classes(self, client, v0):
        hm = v0.manifest_hash
        before = telemetry_store.get_failure_classes("v0")
        resp = client.post(
            "/telemetry/record",
            json=_telemetry_dict(manifest_hash_val=hm),
            headers={"X-API-Key": "test-governance-key"},
        )
        after = telemetry_store.get_failure_classes("v0")
        assert resp.json()["trusted"] is True
        assert after.get("math:incorrect-answer", 0) == before.get("math:incorrect-answer", 0) + 1


class TestTelemetryLimits:
    def test_payload_size_cap(self, client, v0):
        big_input = {"blob": "x" * (70 * 1024)}
        resp = client.post(
            "/telemetry/record",
            json=_telemetry_dict(input=big_input),
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 413

    def test_rate_limit(self, client, v0):
        from app.telemetry._init import _external_telemetry_limiter
        # Reset limiter to deterministic state, then exceed the 30/60s budget.
        _external_telemetry_limiter._buckets.clear()
        codes = set()
        for _ in range(40):
            r = client.post(
                "/telemetry/record", json=_telemetry_dict(),
                headers={"X-API-Key": "test-governance-key"},
            )
            codes.add(r.status_code)
        assert 429 in codes


class TestOperatorAttributedTelemetry:
    def test_operator_build_telemetry_has_exact_hashes(self, v0):
        from app.governance.governor import RuntimeManifest
        operator = Operator(registry=registry, current_runtime=v0)
        operator.set_constitution_hash("constitution-pin-hash")
        from app.operator.operator import TaskResult
        result = TaskResult(
            task_id="t1", input={"expression": "2+2"}, output="4",
            success=True, runtime_version=v0.version,
        )
        t = operator.build_telemetry(result)
        assert t.trusted is True
        assert t.source == "operator"
        # Telemetry carries the exact runtime manifest hash as computed by the
        # registry's canonical algorithm, bound to the running runtime.
        assert t.runtime_manifest_hash == RuntimeRegistry._with_hash(v0).manifest_hash
        assert t.constitution_hash == "constitution-pin-hash"
        assert t.operation == {"verb": "execute_task", "task_id": "t1"}
        assert t.source_identity["owner"] == "operator"

    def test_untrusted_not_reloaded_into_trusted_set(self, tmp_path):
        from app.storage.store import StateStore
        store = StateStore(str(tmp_path / "t.db"))
        ts = TelemetryStore(persistence=store)
        ts.record_execution(ExecutionTelemetry(
            run_id="", runtime_id="runtime-v0", runtime_version="v0",
            task_id="x", input={}, output="o", success=False,
            errors=["e"], source="external", trusted=False,
        ))
        ts.record_execution(ExecutionTelemetry(
            run_id="", runtime_id="runtime-v0", runtime_version="v0",
            task_id="y", input={}, output="o", success=False,
            errors=["e"], source="operator", trusted=True,
        ))
        ts2 = TelemetryStore(persistence=store)
        assert len(ts2.get_failure_records()) == 1
        assert ts2.get_failure_records()[0].task_id == "y"