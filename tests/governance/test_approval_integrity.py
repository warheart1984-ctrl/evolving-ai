"""P3 approval integrity: evidence-linked approval, reviewer != proposer, diff surfacing."""
import pytest

from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evaluation, Evidence
from constitution.constitution import Constitution


@pytest.fixture()
def governor():
    constitution = Constitution()
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v0", model_identifier="m", constitution_version="v1",
        prompts={"system": "old"}, created_by="system", description="v0",
    )
    return Governor(registry, constitution), registry


def _bind_hash(governor, amendment):
    """Stamp the exact candidate manifest hash onto the evaluation (P5)."""
    amendment.evaluation.candidate_manifest_hash = governor.build_candidate(amendment).manifest_hash
    return amendment


def _evaluation(amendment_id, evidence_ids=("ev-1",)):
    return Evaluation(
        id=f"eval-{amendment_id}", amendment_id=amendment_id,
        parent_runtime="v0", candidate_runtime="v1",
        correctness=0.95, instruction_following=0.92, robustness=0.9,
        safety=1.0, regressions=0,
        latency_ms=100.0, cost_per_task=0.01,
        parent_latency_ms=100.0, parent_cost_per_task=0.01,
        evidence=[Evidence(id=eid, type="replay", description="replay", runtime_version="v1")
                  for eid in evidence_ids],
    )


def _amendment(amendment_id, proposer="steward", reviewer="human", evidence_ids=("ev-1",)):
    return Amendment(
        id=amendment_id, parent_version="v0", target=TargetType.PROMPT,
        description="Improve prompt", rationale="Regression",
        proposed_diff={"prompts": {"system": "new"}},
        proposer=proposer, reviewer=reviewer,
        evaluation=_evaluation(amendment_id, evidence_ids),
        status=AmendmentStatus.REVIEW,
    )


class TestEvidenceLinkedApproval:
    def test_approval_requires_evidence_ids(self, governor):
        gov, registry = governor
        amendment = _amendment("prop-ev")
        result = gov.approve_amendment(amendment, evidence_ids=None, reviewer="human")
        assert result.success is False
        assert "evidence" in result.reason.lower() and "id" in result.reason.lower()

    def test_approval_rejects_unknown_evidence_ids(self, governor):
        gov, registry = governor
        amendment = _amendment("prop-ev")
        result = gov.approve_amendment(amendment, evidence_ids=["ev-999"], reviewer="human")
        assert result.success is False
        assert "unknown evidence" in result.reason.lower()

    def test_approval_succeeds_with_valid_evidence_id(self, governor):
        gov, registry = governor
        amendment = _bind_hash(gov, _amendment("prop-ev"))
        result = gov.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is True

    def test_audit_entry_records_evidence_ids(self, governor):
        gov, registry = governor
        amendment = _bind_hash(gov, _amendment("prop-ev"))
        result = gov.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is True
        entry = result.audit_log[0]
        assert entry["evidence_ids"] == ["ev-1"]


class TestReviewerNotProposer:
    def test_self_approval_blocked(self, governor):
        gov, registry = governor
        amendment = _bind_hash(gov, _amendment("prop-self", proposer="steward", reviewer="steward"))
        result = gov.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="steward")
        assert result.success is False
        assert "Self-approval blocked" in result.reason

    def test_reviewer_must_be_explicit(self, governor):
        gov, registry = governor
        amendment = _amendment("prop-norev")
        amendment.reviewer = None
        result = gov.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer=None)
        assert result.success is False
        assert "reviewer" in result.reason.lower()


class TestDiffSurfacing:
    def test_amendment_diff_lists_changes(self, governor):
        gov, registry = governor
        amendment = _amendment("prop-diff")
        diff = gov.amendment_diff(amendment)
        assert diff["parent_runtime"] == "runtime-v0"
        assert diff["has_changes"] is True
        prompts_changes = [c for c in diff["changes"]["prompts"] if c["field"] == "prompts"]
        assert prompts_changes

    def test_diff_no_change_for_identical_amendment(self, governor):
        gov, registry = governor
        amend = _amendment("prop-same")
        amend.proposed_diff = {}
        diff = gov.amendment_diff(amend)
        assert diff["has_changes"] is False