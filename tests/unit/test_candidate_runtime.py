import pytest

from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evidence, Evaluation
from constitution.constitution import Constitution


def test_candidate_is_materialized_without_flipping_current():
    registry = RuntimeRegistry()
    registry.create_runtime("v0", "model", "v1", prompts={"system": "old"})
    governor = Governor(registry, Constitution())
    amendment = Amendment(
        id="prop-candidate",
        parent_version="v0",
        target=TargetType.PROMPT,
        description="change prompt",
        rationale="test",
        proposed_diff={"prompts": {"system": "new"}},
        proposer="steward",
        status=AmendmentStatus.PROPOSED,
    )

    candidate = governor.materialize_candidate(amendment)

    assert candidate.id == "runtime-v0-candidate-prop-candidate"
    assert candidate.manifest_hash
    assert candidate.prompts["system"] == "new"
    assert registry.get_current().version == "v0"


def test_governor_rejects_model_diff_even_if_called_directly():
    registry = RuntimeRegistry()
    registry.create_runtime("v0", "model", "v1")
    governor = Governor(registry, Constitution())
    amendment = Amendment(
        id="prop-illegal",
        parent_version="v0",
        target=TargetType.PROMPT,
        description="illegal model change",
        rationale="test",
        proposed_diff={"model_identifier": "new-model"},
        proposer="steward",
    )

    result = governor.approve_amendment(amendment)

    assert not result.success
    assert "Illegal v0 diff fields" in result.reason
