"""Unauthorized self-modification attempts must be blocked and auditable."""
import pytest

from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evaluation, Evidence
from app.operator.operator import Operator
from app.steward._init import Steward
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
    steward = Steward(registry, constitution)
    return constitution, registry, governor, steward, registry.get_current()


class TestUnauthorizedSelfModification:
    def test_operator_cannot_mutate_runtime_manifest(self, system):
        """Runtime manifests are immutable: even the Operator holding one cannot change it."""
        _, registry, _, _, runtime = system
        operator = Operator(registry=registry, current_runtime=runtime)

        with pytest.raises(Exception):
            operator.current_runtime.model_identifier = "rogue-model"
        with pytest.raises(Exception):
            operator.current_runtime.prompts = {"system": "rogue prompt"}

        stored = registry.get_runtime("runtime-v0")
        assert stored.model_identifier == "test-model"
        assert stored.prompts["system"] == "approved system prompt"

    def test_steward_proposal_does_not_change_runtime(self, system):
        """A Steward proposal alone never changes the runtime."""
        _, registry, _, steward, _ = system
        pattern = steward.analyze_telemetry([
            {"task_id": "t1", "success": False, "errors": ["e"], "runtime_version": "v0"},
            {"task_id": "t1", "success": False, "errors": ["e"], "runtime_version": "v0"},
            {"task_id": "t1", "success": False, "errors": ["e"], "runtime_version": "v0"},
        ])[0]
        hypothesis = steward.generate_hypothesis(pattern, TargetType.PROMPT)
        proposal = steward.propose_amendment(
            hypothesis=hypothesis,
            target_component="prompt",
            description="Improve prompt",
            rationale="Repeated failures",
            proposed_diff={"prompts": {"system": "new prompt"}},
        )
        assert proposal.status == AmendmentStatus.PROPOSED
        assert registry.get_current().version == "v0"

    def test_promotion_requires_human_approval(self, system):
        """Even with a passing evaluation, promotion without human approval is refused."""
        _, registry, governor, _, _ = system
        evaluation = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0", candidate_runtime="v1",
            correctness=0.95, instruction_following=0.92, safety=1.0, regressions=0,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
        )
        amendment = Amendment(
            id="prop-1", parent_version="v0", target=TargetType.PROMPT,
            description="Improve prompt", rationale="Better failures",
            proposed_diff={"prompts": {"system": "new prompt"}},
            proposer="steward",
            evaluation=evaluation,
            status=AmendmentStatus.PROPOSED,
        )
        result = governor.approve_amendment(amendment)
        assert result.success is False
        assert "Human approval" in result.reason
        assert registry.get_current().version == "v0"

    def test_promotion_blocked_without_evaluation(self, system):
        """No evaluation evidence → no promotion."""
        _, registry, governor, _, _ = system
        amendment = Amendment(
            id="prop-2", parent_version="v0", target=TargetType.PROMPT,
            description="Improve prompt", rationale="Skipping evaluation",
            proposed_diff={"prompts": {"system": "new prompt"}},
            proposer="steward",
            evaluation=None,
            status=AmendmentStatus.APPROVED,
        )
        result = governor.approve_amendment(amendment)
        assert result.success is False
        assert registry.get_current().version == "v0"

    def test_promotion_blocked_by_failing_gates(self, system):
        """Failing governance gates → no promotion, even with human approval."""
        _, registry, governor, _, _ = system
        evaluation = Evaluation(
            id="eval-3", amendment_id="prop-3", parent_runtime="v0", candidate_runtime="v1",
            correctness=0.40, instruction_following=0.50, safety=0.5, regressions=3,
            evidence=[Evidence(id="ev-3", type="replay", description="replay", runtime_version="v1")],
        )
        amendment = Amendment(
            id="prop-3", parent_version="v0", target=TargetType.PROMPT,
            description="Risky change", rationale="Overfitting",
            proposed_diff={"prompts": {"system": "risky"}},
            proposer="steward",
            evaluation=evaluation,
            status=AmendmentStatus.APPROVED,
        )
        result = governor.approve_amendment(amendment)
        assert result.success is False
        assert "Governance gates failed" in result.reason
        assert registry.get_current().version == "v0"

    def test_v0_mutation_scope_enforced(self, system):
        """v0 only allows prompt and memory-rule changes."""
        from fastapi import HTTPException

        import asyncio
        from app.api.main import propose_amendment

        with pytest.raises(HTTPException) as exc:
            asyncio.run(propose_amendment(
                target="model",
                description="Swap model",
                rationale="Faster model",
                proposed_diff={"model_identifier": "rogue-model"},
            ))
        assert exc.value.status_code == 403
