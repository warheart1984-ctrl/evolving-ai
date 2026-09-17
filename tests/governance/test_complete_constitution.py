"""P4: complete constitution enforcement — every declared gate bound, fail closed."""
import pytest

from app.governance.models import ConstitutionGates, Evaluation, Evidence
from app.governance.governor import (
    Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType,
)
from constitution.constitution import Constitution


def _evaluation(**overrides):
    data = dict(
        id="eval-1", amendment_id="prop-1", parent_runtime="v0", candidate_runtime="v1",
        correctness=0.95, instruction_following=0.93, robustness=0.92,
        safety=1.0, regressions=0,
        latency_ms=100.0, cost_per_task=0.01,
        parent_latency_ms=100.0, parent_cost_per_task=0.01,
        evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
    )
    data.update(overrides)
    return Evaluation(**data)


def _bind_hash(governor, amendment):
    """Stamp the exact candidate manifest hash onto the evaluation (P5)."""
    amendment.evaluation.candidate_manifest_hash = governor.build_candidate(amendment).manifest_hash
    return amendment


class TestEveryDeclaredGateBound:
    def test_all_promotion_gates_are_evaluated(self):
        """Every gate in constitution.promotion_gates must appear in results."""
        c = Constitution()
        declared = set(c.promotion_gates.keys())
        gates = ConstitutionGates.check_all_gates(_evaluation(), c)
        assert declared <= set(gates.keys())

    def test_new_declared_gate_fails_closed(self):
        """A gate added to the constitution with no handler cannot pass."""

        class Widened(Constitution):
            pass

        widened = Widened()
        # Simulate a widened constitution with an unenforced gate
        data = widened.model_dump(mode="json")
        data["promotion_gates"]["quantum_verified"] = True
        widened = Widened(**data)

        gates = ConstitutionGates.check_all_gates(_evaluation(), widened)
        assert "quantum_verified" in gates
        assert gates["quantum_verified"].passed is False
        assert "unenforced" in gates["quantum_verified"].details["reason"]

    def test_gates_recorded_on_step_summary_for_good_evaluation(self):
        c = Constitution()
        gates = ConstitutionGates.check_all_gates(_evaluation(), c)
        failed = [name for name, g in gates.items() if not g.passed]
        assert failed == []

    def test_healthcheck_evaluation_evidence_gate(self):
        gates = ConstitutionGates.check_all_gates(_evaluation(evidence=[]), Constitution())
        assert gates["evaluation_evidence_exists"].passed is False


class TestParentRelativeDeltaGates:
    def test_latency_delta_over_budget_fails(self):
        c = Constitution()
        evaluation = _evaluation(latency_ms=150.0, parent_latency_ms=100.0)  # +50% > 20%
        gates = ConstitutionGates.check_all_gates(evaluation, c)
        assert gates["latency_within_budget"].passed is False

    def test_cost_delta_over_budget_fails(self):
        c = Constitution()
        evaluation = _evaluation(cost_per_task=0.05, parent_cost_per_task=0.01)  # +400% > 15%
        gates = ConstitutionGates.check_all_gates(evaluation, c)
        assert gates["cost_within_budget"].passed is False

    def test_delta_improvement_passes(self):
        c = Constitution()
        evaluation = _evaluation(latency_ms=80.0, parent_latency_ms=100.0,
                                 cost_per_task=0.005, parent_cost_per_task=0.01)
        gates = ConstitutionGates.check_all_gates(evaluation, c)
        assert gates["latency_within_budget"].passed is True
        assert gates["cost_within_budget"].passed is True


class TestGovernorFailsClosed:
    @pytest.fixture()
    def governor(self):
        constitution = Constitution()
        registry = RuntimeRegistry()
        registry.create_runtime(
            version="v0", model_identifier="m", constitution_version="v1",
            prompts={"system": "old"}, created_by="system", description="v0",
        )
        return Governor(registry, constitution), registry

    def _amendment(self, evaluation):
        return Amendment(
            id="prop-1", parent_version="v0", target=TargetType.PROMPT,
            description="Improve prompt", rationale="Regression",
            proposed_diff={"prompts": {"system": "new"}},
            proposer="steward", reviewer="human",
            evaluation=evaluation, status=AmendmentStatus.REVIEW,
        )

    def test_approval_rejected_without_parent_metrics(self, governor):
        """Missing parent-relative metrics must fail closed at approval."""
        gov, _ = governor
        evaluation = _evaluation(
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=None, parent_cost_per_task=None,
        )
        result = gov.approve_amendment(self._amendment(evaluation), evidence_ids=["ev-1"])
        assert result.success is False
        assert "latency_within_budget" in result.reason or "cost_within_budget" in result.reason

    def test_approval_rejected_when_latency_budget_exceeded(self, governor):
        gov, _ = governor
        evaluation = _evaluation(latency_ms=300.0, parent_latency_ms=100.0)  # +200%
        result = gov.approve_amendment(self._amendment(evaluation), evidence_ids=["ev-1"])
        assert result.success is False
        assert "latency_within_budget" in result.reason

    def test_approval_rejected_when_cost_budget_exceeded(self, governor):
        gov, _ = governor
        evaluation = _evaluation(cost_per_task=0.1, parent_cost_per_task=0.01)  # +900%
        result = gov.approve_amendment(self._amendment(evaluation), evidence_ids=["ev-1"])
        assert result.success is False
        assert "cost_within_budget" in result.reason

    def test_approval_rejected_without_robustness(self, governor):
        """Fail closed: no robustness score means required_tests_pass fails."""
        gov, _ = governor
        evaluation = _evaluation(robustness=0.0)
        result = gov.approve_amendment(self._amendment(evaluation), evidence_ids=["ev-1"])
        assert result.success is False
        assert "required_tests_pass" in result.reason

    def test_approval_succeeds_with_complete_metrics(self, governor):
        gov, _ = governor
        amendment = _bind_hash(gov, self._amendment(_evaluation()))
        result = gov.approve_amendment(amendment, evidence_ids=["ev-1"])
        assert result.success is True