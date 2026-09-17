"""P5: approval bound to exact evidence and exact candidate manifest.

The governor must approve the *exact* sandbox candidate that was evaluated
and only on evidence that remains canonically intact. Tampered evidence or
a changed candidate (proposed_diff mutated after evaluation) must be
rejected even when all gates pass.
"""
import pytest

from app.evaluation._init import FailureClassRegistry
from app.governance.governor import (
    Amendment,
    AmendmentStatus,
    Governor,
    RuntimeRegistry,
    TargetType,
)
from app.governance.models import Evaluation, Evidence
from constitution.constitution import Constitution


def _evidence(**overrides):
    data = dict(
        id="ev-1",
        type="replay",
        description="replay parent vs candidate",
        results={"candidate": {"passed": 8, "total": 8}},
        runtime_version="v0-candidate-prop-1",
    )
    data.update(overrides)
    return Evidence(**data)


def _evaluation(**overrides):
    data = dict(
        id="eval-1",
        amendment_id="prop-1",
        parent_runtime="v0",
        candidate_runtime="v0-candidate-prop-1",
        correctness=0.95,
        instruction_following=0.93,
        robustness=0.92,
        safety=1.0,
        regressions=0,
        latency_ms=100.0,
        cost_per_task=0.01,
        parent_latency_ms=100.0,
        parent_cost_per_task=0.01,
        evidence=[_evidence()],
    )
    data.update(overrides)
    return Evaluation(**data)


@pytest.fixture()
def env():
    constitution = Constitution()
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v0",
        model_identifier="m",
        constitution_version="v1",
        prompts={"system": "old-system"},
        created_by="system",
        description="v0",
    )
    governor = Governor(registry, constitution)
    return governor, registry, constitution


def _proposed(env, **overrides):
    _, registry, _ = env
    amendment = Amendment(
        id="prop-1",
        parent_version="v0",
        target=TargetType.PROMPT,
        description="Improve system prompt",
        rationale="Fix observed regressions",
        proposed_diff={"prompts": {"system": "new-system"}},
        proposer="steward",
    )
    return amendment


class TestExactCandidateBinding:
    def test_approval_succeeds_when_candidate_unchanged(self, env):
        governor, _, _ = env
        amendment = _proposed(env)
        # Materialize the candidate, evaluating THE SAME proposed_diff it carries.
        candidate = governor.materialize_candidate(amendment)
        amendment.evaluation = _evaluation(
            candidate_runtime=candidate.version,
            evidence=[_evidence(runtime_version=candidate.version)],
            candidate_manifest_hash=candidate.manifest_hash,
        )
        amendment.status = AmendmentStatus.REVIEW
        result = governor.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is True, result.reason

    def test_approval_rejected_when_candidate_diff_changes(self, env):
        governor, _, _ = env
        amendment = _proposed(env)
        candidate = governor.materialize_candidate(amendment)
        amendment.evaluation = _evaluation(
            candidate_runtime=candidate.version,
            evidence=[_evidence(runtime_version=candidate.version)],
            candidate_manifest_hash=candidate.manifest_hash,
        )
        amendment.status = AmendmentStatus.REVIEW
        # The diff is changed AFTER evaluation -> must fail even though gates pass.
        amendment.proposed_diff = {"prompts": {"system": "tampered-system"}}
        result = governor.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is False
        assert "candidate changed since evaluation" in result.reason

    def test_approval_rejected_when_candidate_hash_missing(self, env):
        governor, _, _ = env
        amendment = _proposed(env)
        amendment.evaluation = _evaluation()  # no candidate_manifest_hash
        amendment.status = AmendmentStatus.REVIEW
        result = governor.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is False
        assert "no candidate manifest hash" in result.reason

    def test_materialize_candidate_is_idempotent_by_hash(self, env):
        governor, registry, _ = env
        amendment = _proposed(env)
        c1 = governor.materialize_candidate(amendment)
        c2 = governor.build_candidate(amendment)
        # Same proposed_diff materializes to the same canonical manifest hash.
        assert c1.manifest_hash == c2.manifest_hash
        # register_candidate is idempotent for identical immutable candidates
        # (registering the same amendment twice returns the same SANDBOX).
        c3 = governor.materialize_candidate(amendment)
        assert c3.id == c1.id and c3.manifest_hash == c1.manifest_hash


class TestEvidenceIntegrity:
    def test_approval_rejected_when_evidence_modified(self, env):
        governor, _, _ = env
        amendment = _proposed(env)
        candidate = governor.materialize_candidate(amendment)
        amendment.evaluation = _evaluation(
            candidate_runtime=candidate.version,
            evidence=[_evidence(runtime_version=candidate.version)],
            candidate_manifest_hash=candidate.manifest_hash,
        )
        amendment.status = AmendmentStatus.REVIEW
        # Tamper with evidence results AFTER evaluation.
        amendment.evaluation.evidence[0].results["candidate"]["passed"] = 999
        result = governor.approve_amendment(amendment, evidence_ids=["ev-1"], reviewer="human")
        assert result.success is False
        assert "evidence modified" in result.reason

    def test_evidence_hash_detects_rewritten_payload(self):
        e = _evidence()
        h = e.canonical_hash
        e.results["candidate"]["passed"] = 0
        assert e.is_intact() is False
        # The field keeps the original stamp; recomputation detects the rewrite.
        assert e.canonical_hash == h

    def test_untouched_evidence_is_intact(self):
        e = _evidence()
        assert e.is_intact() is True


class TestGovernorFailsClosed:
    def test_failure_class_registry_wired(self, env):
        governor, _, _ = env
        assert isinstance(governor.failure_class_registry, FailureClassRegistry)