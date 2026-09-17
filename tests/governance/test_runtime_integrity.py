"""P4 runtime integrity: immutability guarantee, rollback audit, concurrent handling."""
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
            correctness=0.95, instruction_following=0.92, safety=1.0, regressions=0,
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
        amendment = _review_amendment("prop-diff", reviewer="human")
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
        amendment = _review_amendment("prop-rb", reviewer="human")
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

        amendment = _review_amendment("prop-audit", reviewer="human")
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
        a1 = _review_amendment("prop-a", reviewer="human")
        a2 = _review_amendment("prop-b", reviewer="human")

        # Simulate a promotion in progress by forking one first
        result1 = governor.approve_amendment(a1, evidence_ids=["ev-prop-a"])
        assert result1.success is True

        # Now try concurrent: lock is held during the in-band call, but here
        # simulate the lock being held explicitly and confirm rejection.
        governor._promoting_amendment = "prop-a"
        a2.evaluation = Evaluation(
            id="eval-b", amendment_id="prop-b", parent_runtime="v1", candidate_runtime="v2",
            correctness=0.95, instruction_following=0.92, safety=1.0, regressions=0,
            evidence=[Evidence(id="ev-prop-b", type="replay", description="r", runtime_version="v2")],
        )
        a2.parent_version = "v1"
        result2 = governor.approve_amendment(a2, evidence_ids=["ev-prop-b"])
        governor._promoting_amendment = None  # release lock

        assert result2.success is False
        assert "Concurrent promotion" in result2.reason
        assert registry.get_current().version == "v1"