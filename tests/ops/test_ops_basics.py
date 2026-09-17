"""P6 ops basics: SQLite persistence survives restarts; governance API key enforcement."""
import os

import pytest
from fastapi.testclient import TestClient

from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evaluation, Evidence
from app.storage.store import StateStore
from app.telemetry._init import ExecutionTelemetry, TelemetryStore
from constitution.constitution import Constitution


class TestSQLitePersistence:
    def test_state_store_roundtrip(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        store.save("amendment", "a1", {"id": "a1", "status": "review"})
        store.save("telemetry", "run-1", {"run_id": "run-1", "success": False})
        assert store.load("amendment", "a1") == {"id": "a1", "status": "review"}
        assert store.count("telemetry") == 1

        # New store instance on same path = "restart"
        store2 = StateStore(str(tmp_path / "evolving.db"))
        assert store2.load("amendment", "a1")["status"] == "review"
        assert store2.load_all("telemetry")["run-1"]["success"] is False

    def test_registry_reloads_runtime_history(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        registry = RuntimeRegistry(persistence=store)
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")

        # "Restart": new registry instance over the same store.
        registry2 = RuntimeRegistry(persistence=store)
        assert registry2.get_current().version == "v0"
        assert len(registry2.list_runtimes()) == 1

    def test_telemetry_reloads_after_restart(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        ts = TelemetryStore(persistence=store)
        ts.record_execution(ExecutionTelemetry(
            run_id="", runtime_id="runtime-v0", runtime_version="v0",
            task_id="t1", input={"expression": "2+2"}, output="5",
            success=False, errors=["Incorrect answer"],
        ))

        ts2 = TelemetryStore(persistence=store)
        records = ts2.get_failure_records()
        assert len(records) == 1
        assert records[0].failure_class == "math:incorrect-answer"

    def test_amendment_loads_regression_cases_after_reload(self, tmp_path):
        store = StateStore(str(tmp_path / "evolving.db"))
        registry = RuntimeRegistry(persistence=store)
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        governor = Governor(registry, Constitution(), persistence=store)

        from app.governance.models import RegressionCase

        amendment = Amendment(
            id="prop-persist", parent_version="v0", target=TargetType.PROMPT,
            description="Fix math", rationale="math failures",
            proposed_diff={"prompts": {"system": "verify"}},
            proposer="steward",
            regression_cases=[
                RegressionCase(id="reg-persist", failure_class="math:incorrect-answer",
                               task_id="reg-persist", task_type="math",
                               input_data={"expression": "2+2"}, expected_output="4"),
            ],
        )
        store.save("amendment", amendment.id, amendment.model_dump(mode="json"))

        loaded = Amendment(**store.load("amendment", "prop-persist"))
        assert loaded.regression_cases[0].failure_class == "math:incorrect-answer"
        assert loaded.regression_cases[0].expected_output == "4"


class TestGovernanceAuth:
    @pytest.fixture()
    def client(self):
        os.environ["APP_ENV"] = "development"
        from app.api.main import app
        return TestClient(app)

    def test_approve_requires_api_key(self, client):
        resp = client.post("/governance/approve/prop-x", params={"reviewer": "human"})
        assert resp.status_code == 401

    def test_approve_accepts_dev_key(self, client):
        resp = client.post(
            "/governance/approve/prop-x",
            headers={"X-API-Key": "governance-dev-key"},
            params={"reviewer": "human"},
        )
        # Amendment prop-x doesn't exist, but the auth gate must have passed
        # (404 not 401).
        assert resp.status_code == 404

    def test_reject_requires_api_key(self, client):
        resp = client.post("/governance/reject/prop-x", params={"reason": "nope"})
        assert resp.status_code == 401

    def test_rollback_requires_api_key(self, client):
        resp = client.post("/runtime/rollback/v0")
        assert resp.status_code == 401

    def test_rollback_accepts_dev_key(self, client):
        resp = client.post("/runtime/rollback/v0", headers={"X-API-Key": "governance-dev-key"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "rolled_back"

    def test_bad_key_rejected(self, client):
        resp = client.post("/runtime/rollback/v0", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_boot_strict_constitution(self):
        """The API module refuses to import (boot) when the pin mismatches,
        in a fresh interpreter so module-import side effects are real."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from unittest.mock import patch\n"
            "with patch('constitution.constitution._sha256_of_file', return_value='0'*64):\n"
            "    try:\n"
            "        import app.api.main\n"
            "        sys.exit(0)\n"
            "    except Exception as e:\n"
            "        sys.stderr.write(type(e).__name__)\n"
            "        sys.exit(42)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=os.getcwd(),
            env={**os.environ, "APP_ENV": "development"},
        )
        assert proc.returncode == 42, f"API booted despite pin mismatch: {proc.stdout} {proc.stderr}"
        assert "ConstitutionIntegrityError" in proc.stderr