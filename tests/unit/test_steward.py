"""Tests for steward module - analyzes failures and proposes amendments."""
import pytest

from app.governance.models import AmendmentStatus, TargetType
from app.governance.registry import RuntimeRegistry
from app.steward._init import Steward


@pytest.fixture()
def steward():
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v0", model_identifier="m", constitution_version="v1",
        created_by="system", description="v0",
    )
    return Steward(registry, None)


def _telemetry(task_id="task-001", count=4):
    return [
        {"task_id": task_id, "success": False, "errors": ["Incorrect answer"], "runtime_version": "v0"}
        for _ in range(count)
    ]


class TestSteward:
    def test_analyze_telemetry_finds_patterns(self, steward):
        patterns = steward.analyze_telemetry(_telemetry())
        assert len(patterns) == 1
        assert patterns[0].frequency == 4

    def test_analyze_telemetry_ignores_isolated_failures(self, steward):
        patterns = steward.analyze_telemetry(_telemetry(count=1))
        assert len(patterns) == 0

    def test_generate_hypothesis(self, steward):
        pattern = steward.analyze_telemetry(_telemetry())[0]
        hypothesis = steward.generate_hypothesis(pattern, TargetType.PROMPT)
        assert hypothesis.target == TargetType.PROMPT
        assert hypothesis.supporting_evidence == [pattern.id]

    def test_propose_amendment(self, steward):
        pattern = steward.analyze_telemetry(_telemetry())[0]
        hypothesis = steward.generate_hypothesis(pattern, TargetType.PROMPT)
        proposal = steward.propose_amendment(
            hypothesis=hypothesis,
            target_component="prompt",
            description="Improve prompt",
            rationale="Repeated failures",
            proposed_diff={"prompts": {"system": "new"}},
        )
        assert proposal.status == AmendmentStatus.PROPOSED
        assert proposal.parent_version == "v0"
        assert proposal.proposer == "steward"

    def test_steward_cannot_promote(self, steward):
        """The Steward is creative, not authoritative: it has no promotion powers."""
        assert not hasattr(steward, "promote")
        assert not hasattr(steward, "approve")
