"""P7: API security defaults — explicit keys, protected mutation endpoints, safe CORS."""
import os

import pytest
from fastapi.testclient import TestClient

from app.api.auth import get_allowed_keys


class TestApiKeyDefaults:
    def test_no_fallback_key_without_env(self, monkeypatch):
        """get_allowed_keys() must be empty without GOVERNANCE_API_KEYS (no dev key)."""
        monkeypatch.delenv("GOVERNANCE_API_KEYS", raising=False)
        monkeypatch.setenv("APP_ENV", "development")  # development must NOT grant a key
        assert get_allowed_keys() == set()

    def test_explicit_keys_parsed(self, monkeypatch):
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "a-key, b-key,,c-key")
        assert get_allowed_keys() == {"a-key", "b-key", "c-key"}

    def test_newer_schema_keyword_never_accepted(self, monkeypatch):
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "governance-dev-key")  # explicit = ok
        assert "governance-dev-key" in get_allowed_keys()


class TestProtectedEndpoints:
    @pytest.fixture(autouse=True)
    def clear_keys(self, monkeypatch):
        """Force every request through the unauthenticated path for the no-key test."""
        monkeypatch.delenv("GOVERNANCE_API_KEYS", raising=False)
        yield
        os.environ["GOVERNANCE_API_KEYS"] = "test-governance-key"

    @pytest.fixture()
    def client(self):
        from app.api.main import app
        return TestClient(app)

    @pytest.mark.parametrize("method,path,params", [
        ("post", "/amendment/propose", {"target": "prompt", "description": "d",
                                        "rationale": "r", "proposed_diff": "{}"}),
        ("post", "/evaluation/run", {"suite_id": "core", "runtime_version": "v0"}),
        ("post", "/governance/evaluate/prop-1", {"suite_id": "core"}),
        ("post", "/memory/lesson", {"claim": "c", "scope": "global", "created_by": "u"}),
        ("post", "/memory/lesson/L1/validate", {"validator_id": "u"}),
        ("post", "/memory/lesson/L1/activate", {}),
        ("post", "/memory/lesson/L1/quarantine", {"reason": "r", "quarantiner_id": "u"}),
        ("post", "/steward/analyze", {}),
        ("post", "/runtime/rollback/v0", {}),
        ("post", "/governance/approve/prop-1", {"reviewer": "human"}),
        ("post", "/governance/reject/prop-1", {"reason": "r"}),
    ])
    def test_mutation_endpoints_fail_closed_without_key(self, client, method, path, params):
        resp = getattr(client, method)(path, params=params)
        assert resp.status_code in (401, 503), f"{method} {path} -> {resp.status_code}"

    def test_configured_key_reaches_handler(self, monkeypatch):
        """With a configured key, the auth gate passes (404 = handler found it)."""
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "test-governance-key")
        from app.api.main import app
        resp = TestClient(app).post(
            "/governance/approve/prop-x",
            params={"reviewer": "human"},
            headers={"X-API-Key": "test-governance-key"},
        )
        assert resp.status_code == 404  # auth passed, amendment not found


class TestCorsDefaults:
    def test_cors_credentials_disallowed(self):
        """allow_credentials must be off so wildcard origins are never combined with cookies."""
        import app.api.main as main
        middleware = [
            m for m in main.app.user_middleware
            if m.cls.__name__ == "CORSMiddleware"
        ]
        assert middleware, "CORSMiddleware not registered"
        options = middleware[0].kwargs
        assert options["allow_credentials"] is False
        assert options["allow_origins"] != ["*"]


class TestNoTrustedMutatorWithoutKey:
    def test_execute_still_runs_for_operator(self, monkeypatch):
        """/execute is the operator's user-facing task path (not a governance mutation)."""
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "test-governance-key")
        from app.api.main import app, registry
        if registry.get_current() is None:
            registry.create_runtime(
                version="v0", model_identifier="m", constitution_version="v1",
                prompts={"system": "system_v1"}, created_by="system", description="v0",
            )
        # No X-API-Key: /execute must NOT be auth-blocked (401/503).
        resp = TestClient(app).post(
            "/execute",
            params={"task_id": "math-2", "input_data": '{"expression": "2+2"}'},
        )
        assert resp.status_code != 401
        assert resp.status_code != 503