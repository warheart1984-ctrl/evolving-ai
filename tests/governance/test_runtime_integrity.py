"""P4 runtime integrity: immutability guarantee, rollback audit, concurrent handling."""
from datetime import datetime

import pytest

from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evaluation, Evidence
from app.operator.operator import Operator
from constitution.constitution import Constitution


@pytest.fixture()
def system():
    constitution = Constitution()
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v0",
        model_identifier="test-model",
        constitution_version=constitution.version,
        prompts={"system": "approved system prompt"},
        created_by="system",
        description="v0",
    )
    governor = Governor(registry, constitution)
    return constitution, registry, governor, registry.get_current()


def _review_amendment(amendment_id="prop-imm", proposer="steward", reviewer="human",
                     evaluation=None, parent_version="v0"):
    if evaluation is None:
        evaluation = Evaluation(
            id=f"eval-{amendment_id}", amendment_id=amendment_id,
            parent_runtime="v0", candidate_runtime="v1",
            correctness=0.95, instruction_following=0.92, robustness=0.9, safety=1.0,
            regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id=f"ev-{amendment_id}", type="replay",
                               description="replay", runtime_version="v1")],
        )
    return Amendment(
        id=amendment_id, parent_version=parent_version, target=TargetType.PROMPT,
        description="Improve prompt", rationale="Regression",
        proposed_diff={"prompts": {"system": "new prompt"}},
        proposer=proposer, reviewer=reviewer,
        evaluation=evaluation, status=AmendmentStatus.REVIEW,
    )


def _bind_hash(governor, amendment):
    """Stamp the exact candidate manifest hash onto the evaluation (P5)."""
    amendment.evaluation.candidate_manifest_hash = governor.build_candidate(amendment).manifest_hash
    return amendment


class TestImmutabilityGuarantee:
    def test_nested_dict_mutation_blocked(self, system):
        """frozen=True guards attributes; deep-freeze guards nested containers."""
        _, _, _, runtime = system

        # Attribute-level (covered by frozen=True)
        with pytest.raises(Exception):
            runtime.model_identifier = "rogue"
        with pytest.raises(Exception):
            runtime.prompts = {"system": "rogue"}

        # Nested-container-level (deep freeze) — the prior hole
        with pytest.raises(TypeError):
            runtime.prompts["system"] = "rogue"
        with pytest.raises(TypeError):
            runtime.prompts.update({"system": "rogue"})
        with pytest.raises(TypeError):
            runtime.prompts.pop("system")
        with pytest.raises(TypeError):
            runtime.prompts.clear()

        assert runtime.prompts["system"] == "approved system prompt"
        assert runtime.prompts == {"system": "approved system prompt"}

    def test_operator_cannot_mutate_promoted_runtime(self, system):
        """The Operator holds the current runtime; even so, nested mutation fails."""
        _, registry, _, runtime = system
        operator = Operator(registry=registry, current_runtime=runtime)

        with pytest.raises(TypeError):
            operator.current_runtime.prompts["system"] = "rogue"

        stored = registry.get_runtime("runtime-v0")
        assert stored.prompts["system"] == "approved system prompt"

    def test_applied_diff_creates_new_runtime_not_mutation(self, system):
        """Apply a diff via amendment produces a new frozen runtime; parent untouched."""
        _, registry, governor, runtime = system
        amendment = _bind_hash(governor, _review_amendment("prop-diff", reviewer="human"))
        result = governor.approve_amendment(amendment, evidence_ids=["ev-prop-diff"])
        assert result.success is True

        new_runtime = registry.get_runtime(result.new_runtime_id)
        assert new_runtime.prompts["system"] == "new prompt"
        assert runtime.prompts["system"] == "approved system prompt"
        assert new_runtime.id != runtime.id


class TestRollbackAudit:
    def test_rollback_writes_full_audit_entry(self, system):
        """Rollback records who, when, from/to — not just that it happened."""
        _, registry, governor, _ = system
        amendment = _bind_hash(governor, _review_amendment("prop-rb", reviewer="human"))
        result = governor.approve_amendment(amendment, evidence_ids=["ev-prop-rb"])
        assert result.success is True
        assert registry.get_current().version == "v1"

        target = governor.rollback("v0", initiated_by="ops-human")
        assert target is not None
        assert registry.get_current().version == "v0"

        rollbacks = governor.rollback_log
        assert len(rollbacks) == 1
        entry = rollbacks[0]
        assert entry["action"] == "rollback"
        assert entry["from_runtime"] == result.new_runtime_id
        assert entry["to_runtime"] == "runtime-v0"
        assert entry["from_version"] == "v1"
        assert entry["to_version"] == "v0"
        assert entry["initiated_by"] == "ops-human"
        assert "timestamp" in entry

    def test_rollback_audit_in_trail_and_persisted(self, tmp_path):
        """Audit trail shows both promote and rollback chronologically."""
        from app.storage.store import StateStore

        store = StateStore(str(tmp_path / "evolving.db"))
        constitution = Constitution()
        registry = RuntimeRegistry(persistence=store)
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        governor = Governor(registry, constitution, persistence=store)

        amendment = _bind_hash(governor, _review_amendment("prop-audit", reviewer="human"))
        result = governor.approve_amendment(amendment, evidence_ids=["ev-prop-audit"])
        governor.rollback("v0", initiated_by="ops")

        trail = governor.audit_trail()
        assert [e["action"] for e in trail] == ["promote", "rollback"]

        # Persistence: reload governor from store and see the same trail
        governor2 = Governor(RuntimeRegistry(persistence=store), Constitution(), persistence=store)
        assert [e["action"] for e in governor2.audit_trail()] == ["promote", "rollback"]


class TestConcurrentAmendmentHandling:
    def test_second_promotion_blocked_while_one_in_progress(self, system):
        """Only one promotion may be in progress; a second is rejected."""
        _, registry, governor, _ = system
        a1 = _bind_hash(governor, _review_amendment("prop-a", reviewer="human"))
        a2 = _review_amendment("prop-b", reviewer="human")

        # Simulate a promotion in progress by forking one first
        result1 = governor.approve_amendment(a1, evidence_ids=["ev-prop-a"])
        assert result1.success is True

        # Now try concurrent: lock is held during the in-band call, but here
        # simulate the lock being held explicitly and confirm rejection.
        governor._promoting_amendment = "prop-a"
        a2.evaluation = Evaluation(
            id="eval-b", amendment_id="prop-b", parent_runtime="v1", candidate_runtime="v2",
            correctness=0.95, instruction_following=0.92, robustness=0.9, safety=1.0,
            regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-prop-b", type="replay", description="r", runtime_version="v2")],
        )
        a2.parent_version = "v1"
        a2 = _bind_hash(governor, a2)
        result2 = governor.approve_amendment(a2, evidence_ids=["ev-prop-b"])
        governor._promoting_amendment = None  # release lock

        assert result2.success is False
        assert "Concurrent promotion" in result2.reason
        assert registry.get_current().version == "v1"


class TestRegistryReleasedSandbox:
    def _make_candidate(self, registry, cid, parent="v0", ts="2026-01-01T00:00:01"):
        from app.governance.models import RuntimeManifest
        return RuntimeManifest(
            id=f"runtime-{parent}-candidate-{cid}",
            version=f"{parent}-candidate-{cid}",
            parent_version=parent,
            model_identifier="m",
            constitution_version="v1",
            prompts={},
            created_by="governor:sandbox",
            created_at=datetime.fromisoformat(ts),
        )

    def test_list_runtimes_with_multiple_candidates(self):
        """/runtime/list must not crash when sandbox candidates exist."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        registry.create_runtime(version="v1", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v1")

        registry.register_candidate(self._make_candidate(registry, "a"))
        registry.register_candidate(self._make_candidate(registry, "b"))

        runtimes = registry.list_runtimes()  # must not raise
        released = [r for r in runtimes if r.kind == "released"]
        sandbox = [r for r in runtimes if r.kind == "sandbox"]

        assert [r.version for r in released] == ["v0", "v1"]
        assert len(sandbox) == 2
        assert [r.version for r in sandbox] == ["v0-candidate-a", "v0-candidate-b"]

    def test_api_runtime_list_survives_candidates(self):
        """End-to-end: /runtime/list returns releases plus sandbox candidates."""
        from fastapi.testclient import TestClient
        from unittest import mock
        import app.api.main as main

        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        registry.register_candidate(self._make_candidate(registry, "alpha"))

        with mock.patch.object(main, "registry", registry):
            client = TestClient(main.app)
            resp = client.get("/runtime/list")
        assert resp.status_code == 200
        kinds = [r["kind"] for r in resp.json()]
        assert "released" in kinds and "sandbox" in kinds

    def test_candidate_registration_never_changes_current(self):
        """Registering candidates must not flip the current released runtime."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        current_before = registry.get_current()

        registry.register_candidate(self._make_candidate(registry, "a"))
        registry.register_candidate(self._make_candidate(registry, "b"))

        assert registry.get_current().id == current_before.id == "runtime-v0"
        assert registry.get_current().kind == "released"

    def test_sandbox_candidate_ids_unique_and_immutable(self):
        """Same candidate content is idempotent; different content conflicts."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")

        c1 = self._make_candidate(registry, "a")
        c2 = self._make_candidate(registry, "a")
        c3 = self._make_candidate(registry, "a")
        c3.model_copy(update={})  # model_copy is safe on frozen model

        first = registry.register_candidate(c1)
        # Identical content and id -> idempotent (same hash)
        duplicate = registry.register_candidate(c2)
        assert duplicate.id == first.id
        assert duplicate.manifest_hash == first.manifest_hash

        # Different content under same id -> conflict
        different = self._make_candidate(registry, "a")
        different = different.model_copy(update={"prompts": {"system": "rogue"}})
        with pytest.raises(ValueError):
            registry.register_candidate(different)

    def test_released_not_sandbox_rollback(self):
        """rollback_to only targets released runtimes; candidates are never current."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        registry.register_candidate(self._make_candidate(registry, "a"))

        assert registry.rollback_to("v0") is not None
        # A sandbox candidate can't be addressed as a release version.
        assert registry.get_released("v0-candidate-a") is None