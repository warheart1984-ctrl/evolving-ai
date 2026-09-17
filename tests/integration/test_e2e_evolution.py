"""Integration test demonstrating the full end-to-end evolution scenario.

failure → Steward proposal → sandbox → replay evaluation → evidence
→ human approval → new immutable runtime → rollback → audit trail
"""
from datetime import datetime

import pytest

from app.evaluation._init import Evaluator, ReplaySuite
from app.governance.governor import (
    Amendment,
    AmendmentStatus,
    Governor,
    RuntimeRegistry,
    TargetType,
)
from app.governance.models import Evaluation, Evidence
from app.memory._init import MemoryStore
from app.operator.operator import Operator
from app.steward._init import Steward
from app.telemetry._init import ExecutionTelemetry, TelemetryStore
from constitution.constitution import Constitution


class TestEndToEndEvolution:
    """Integration test: failure → Steward proposal → sandbox →
    replay evaluation → evidence → human approval → new runtime."""

    @pytest.fixture(autouse=True)
    def setup_components(self):
        self.constitution = Constitution()
        self.registry = RuntimeRegistry()
        self.governor = Governor(self.registry, self.constitution)
        self.steward = Steward(self.registry, self.constitution)
        self.memory_store = MemoryStore()
        self.telemetry_store = TelemetryStore()
        self.evaluator = Evaluator(self.registry, {"core-suite": self._create_default_suite()})

    def _create_default_suite(self) -> ReplaySuite:
        suite = ReplaySuite(
            id="core-suite",
            name="Core Evaluation Suite",
            description="Historical tasks for regression testing",
            tags=["core", "historical"],
        )
        for i in range(5):
            suite.add_task({
                "id": f"hist-task-{i:03d}",
                "type": "math",
                "input": {"expression": f"{i + 1}+{i + 1}"},
                "expected_output": str((i + 1) * 2),
                "expected_latency_ms": 150.0,
                "expected_cost": 0.01,
            })
        return suite

    def _create_v0(self):
        return self.registry.create_runtime(
            version="v0",
            model_identifier="gpt-3.5-turbo",
            constitution_version="v1",
            prompts={"system": "You are a helpful math assistant."},
            created_by="system",
            description="Initial runtime v0",
        )

    def _record_failures(self, count: int = 4):
        """Record trusted, attributed telemetry for a repeated failure class."""
        for i in range(count):
            telemetry = ExecutionTelemetry(
                run_id="",
                runtime_id="runtime-v0",
                runtime_version="v0",
                task_id="math-failures",
                input={"expression": f"{i + 1}+{i + 1}"},
                output="incorrect_result",
                success=False,
                errors=["Incorrect answer produced"],
                tools_used=[],
                latency_ms=150.0,
                cost=0.01,
            )
            # Attribute as internal operator telemetry so it is eligible for analysis.
            telemetry = self._attribute_telemetry(telemetry)
            self.telemetry_store.record_execution(telemetry)

    def _attribute_telemetry(self, telemetry):
        """Mark telemetry as trusted operator-produced attribution."""
        telemetry.source = "operator"
        telemetry.trusted = True
        telemetry.runtime_manifest_hash = (
            self.registry.get_current().manifest_hash
            if self.registry.get_current()
            else None
        )
        telemetry.operation = {"verb": "execute_task", "task_id": telemetry.task_id}
        return telemetry

    def test_operator_repeated_failures_detected(self):
        """Step 1: Operator repeatedly fails a class of tasks; telemetry detects it."""
        self._create_v0()

        operator = Operator(registry=self.registry, current_runtime=self.registry.get_current())
        for i in range(4):
            result = operator.execute_task(
                task_id=f"fail-task-{i:03d}",
                input_data={"expression": f"{i + 1}+{i + 1}"},
            )
            result.success = False
            result.errors.append("Incorrect answer produced")

        self._record_failures(4)
        assert len(self.telemetry_store.get_failure_records("v0")) == 4

    def test_steward_analyzes_failures_and_proposes(self):
        """Steps 2-3: Steward analyzes failures and proposes an amendment."""
        self._create_v0()
        self._record_failures(4)

        failure_records = self.telemetry_store.get_failure_records("v0")
        assert len(failure_records) == 4

        proposals = self.steward.recommend_amendments([dict(r) for r in failure_records])
        assert len(proposals) >= 1

        proposal = proposals[0]
        assert proposal.id.startswith("prop-")
        assert proposal.parent_version == "v0"
        assert proposal.target in [TargetType.PROMPT, TargetType.MEMORY]
        assert proposal.description
        assert proposal.rationale
        assert proposal.proposed_diff is not None
        assert proposal.status == AmendmentStatus.PROPOSED
        assert proposal.proposer == "steward"

        # Nothing evolves in place: the runtime is untouched by a proposal
        assert self.registry.get_current().version == "v0"

    def test_evaluator_tests_candidate_against_suite(self):
        """Step 4: Evaluator tests candidate against the replay suite."""
        self._create_v0()

        parent_result = self.evaluator.run_suite_against_runtime("core-suite", "v0")
        candidate_result = self.evaluator.run_suite_against_runtime("core-suite", "candidate-v1")

        assert parent_result.total_tasks == 5
        assert candidate_result.total_tasks == 5
        assert parent_result.passed_tasks == 5

        evaluation = Evaluation(
            id="eval-001",
            amendment_id="prop-001",
            parent_runtime=parent_result.runtime_version,
            candidate_runtime=candidate_result.runtime_version,
            correctness=candidate_result.correctness_avg,
            instruction_following=candidate_result.instruction_following_avg,
            robustness=candidate_result.robustness_score,
            safety=candidate_result.safety_score,
            latency_ms=candidate_result.latency_avg_ms,
            cost_per_task=candidate_result.cost_avg,
            parent_latency_ms=parent_result.latency_avg_ms,
            parent_cost_per_task=parent_result.cost_avg,
            regressions=candidate_result.regressions,
            evidence=[Evidence(
                id="ev-001",
                type="replay",
                description=f"Replay of core-suite: {parent_result.runtime_version} vs candidate",
                results={
                    "parent": {"passed": parent_result.passed_tasks, "total": parent_result.total_tasks},
                    "candidate": {"passed": candidate_result.passed_tasks, "total": candidate_result.total_tasks},
                },
                runtime_version=candidate_result.runtime_version,
            )],
            evaluator_id="evaluator:v0",
        )
        amendment = Amendment(
            id="prop-001",
            parent_version="v0",
            target=TargetType.PROMPT,
            description="Improve math reasoning prompts",
            rationale="Operator repeatedly fails math tasks",
            proposed_diff={
                "prompts": {"system": "You are a helpful math assistant. Show step-by-step reasoning."}
            },
            proposer="steward",
            evaluation=evaluation,
        )

        gate_result = self.governor.evaluate_amendment(amendment)
        for gate in [
            "required_tests_pass",
            "no_safety_failures",
            "no_unexplained_regressions",
            "evaluation_evidence_exists",
        ]:
            assert gate in gate_result["gates_passed"]
        assert gate_result["success"] is True

    def _bind_evaluation_candidate(self, amendment):
        """Stamp the exact evaluated candidate's manifest hash onto the evaluation (P5)."""
        candidate_manifest = self.governor.build_candidate(amendment)
        amendment.evaluation.candidate_manifest_hash = candidate_manifest.manifest_hash
        return amendment

    def test_governor_approval_creates_new_runtime(self):
        """Steps 7-8: Human approves; new immutable runtime created; old stays for rollback."""
        self._create_v0()

        evaluation = Evaluation(
            id="eval-002",
            amendment_id="prop-002",
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=0.91,
            instruction_following=0.92,
            robustness=0.9,
            safety=1.0,
            regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-002", type="replay", description="Replay evidence", runtime_version="v1")],
            evaluator_id="evaluator:v0",
        )
        amendment = Amendment(
            id="prop-002",
            parent_version="v0",
            target=TargetType.PROMPT,
            description="Improved math system prompt",
            rationale="Better step-by-step reasoning reduces errors",
            proposed_diff={
                "prompts": {"system": "You are a helpful math assistant. Show step-by-step reasoning."}
            },
            proposer="steward",
            evaluation=evaluation,
            status=AmendmentStatus.APPROVED,
            reviewer="human:approver-1",
            decided_at=datetime.utcnow(),
        )

        self._bind_evaluation_candidate(amendment)

        promotion_result = self.governor.approve_amendment(amendment, evidence_ids=["ev-002"])
        assert promotion_result.success is True

        new_runtime = self.registry.get_runtime(promotion_result.new_runtime_id)
        assert new_runtime.version == "v1"
        assert new_runtime.parent_version == "v0"
        assert new_runtime.prompts["system"].startswith("You are a helpful math assistant. Show")
        assert amendment.status == AmendmentStatus.PROMOTED
        assert amendment.resulting_runtime == promotion_result.new_runtime_id

        # Old runtime remains intact and available for rollback
        old_runtime = self.registry.get_runtime("runtime-v0")
        assert old_runtime is not None
        assert old_runtime.prompts["system"] == "You are a helpful math assistant."
        assert self.registry.get_current().version == "v1"

    def test_rollback_preserves_old_runtime(self):
        """Step 9: Old runtime remains available for rollback."""
        self.registry.create_runtime(
            version="v0", model_identifier="gpt-3.5-turbo", constitution_version="v1",
            created_by="system", description="v0 original",
        )
        self.registry.create_runtime(
            version="v1", model_identifier="gpt-3.5-turbo", constitution_version="v1",
            created_by="system", description="v1 promoted",
        )

        assert self.registry.get_current().version == "v1"

        rolled_back = self.governor.rollback("v0")
        assert rolled_back is not None
        assert rolled_back.version == "v0"
        assert self.registry.get_current().version == "v0"

        # v1 still exists in the registry
        assert self.registry.get_runtime("runtime-v1") is not None

    def test_audit_trail_recorded(self):
        """Step 10: The entire evolution is recorded in an audit trail."""
        self._create_v0()

        evaluation = Evaluation(
            id="eval-audit-1",
            amendment_id="prop-audit-1",
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=0.92,
            instruction_following=0.94,
            robustness=0.9,
            safety=1.0,
            regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-audit-1", type="replay", description="Replay evidence", runtime_version="v1")],
            evaluator_id="evaluator:v0",
        )
        amendment = Amendment(
            id="prop-audit-1",
            parent_version="v0",
            target=TargetType.PROMPT,
            description="Audit trail test amendment",
            rationale="Testing audit trail completeness",
            proposed_diff={"prompts": {"system": "New prompt for audit testing"}},
            proposer="steward",
            evaluation=evaluation,
            status=AmendmentStatus.APPROVED,
            reviewer="human:auditor",
            decided_at=datetime.utcnow(),
        )

        self._bind_evaluation_candidate(amendment)

        promotion_result = self.governor.approve_amendment(amendment, evidence_ids=["ev-audit-1"])
        assert promotion_result.success is True
        assert len(promotion_result.audit_log) > 0

        audit_entry = promotion_result.audit_log[0]
        assert audit_entry["action"] == "promote"
        assert audit_entry["amendment_id"] == "prop-audit-1"
        assert audit_entry["from_runtime"] == "runtime-v0"
        assert audit_entry["to_runtime"] == f"runtime-{amendment.evaluation.candidate_runtime}"
        assert "timestamp" in audit_entry
        assert "gates_passed" in audit_entry

        # Rollback is also auditable
        self.governor.rollback("v0")
        trail = self.governor.audit_trail()
        actions = [e["action"] for e in trail]
        assert "promote" in actions
        assert "rollback" in actions

    def test_full_evolution_pipeline(self):
        """The complete pipeline: failure → telemetry → Steward → evaluation
        → evidence → human approval → new runtime → rollback → audit."""
        # 1. Operator fails repeatedly on the same task class
        self._create_v0()
        operator = Operator(registry=self.registry, current_runtime=self.registry.get_current())
        for i in range(4):
            result = operator.execute_task(
                task_id="math-failures",
                input_data={"expression": f"{i + 1}+{i + 1}"},
            )
            result.success = False
            result.errors.append("Incorrect answer produced")

        self._record_failures(4)
        assert len(self.telemetry_store.get_failure_records("v0")) == 4

        # 2. Steward analyzes and proposes
        proposals = self.steward.recommend_amendments(
            [dict(r) for r in self.telemetry_store.get_failure_records("v0")]
        )
        assert len(proposals) >= 1
        proposal = proposals[0]

        # 3. Sandbox + replay evaluation (parent vs candidate)
        parent_result = self.evaluator.run_suite_against_runtime("core-suite", "v0")
        candidate_result = self.evaluator.run_suite_against_runtime("core-suite", "candidate-v1")

        evaluation = Evaluation(
            id="eval-pipeline",
            amendment_id=proposal.id,
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=candidate_result.correctness_avg,
            instruction_following=candidate_result.instruction_following_avg,
            robustness=candidate_result.robustness_score,
            safety=candidate_result.safety_score,
            latency_ms=candidate_result.latency_avg_ms,
            cost_per_task=candidate_result.cost_avg,
            parent_latency_ms=parent_result.latency_avg_ms,
            parent_cost_per_task=parent_result.cost_avg,
            regressions=candidate_result.regressions,
            evidence=[Evidence(
                id="ev-pipeline",
                type="replay",
                description=(
                    f"Replay of core-suite: v0 vs v1 "
                    f"({parent_result.passed_tasks}/{parent_result.total_tasks} vs "
                    f"{candidate_result.passed_tasks}/{candidate_result.total_tasks})"
                ),
                results={
                    "parent": {"passed": parent_result.passed_tasks, "total": parent_result.total_tasks},
                    "candidate": {"passed": candidate_result.passed_tasks, "total": candidate_result.total_tasks},
                },
                runtime_version="v1",
            )],
            evaluator_id="evaluator:v0",
        )
        amendment = Amendment(
            id=proposal.id,
            parent_version="v0",
            target=proposal.target,
            description=proposal.description,
            rationale=proposal.rationale,
            proposed_diff=proposal.proposed_diff,
            proposer="steward",
            evaluation=evaluation,
            status=AmendmentStatus.APPROVED,
            reviewer="human:approver-1",
            decided_at=datetime.utcnow(),
        )

        self._bind_evaluation_candidate(amendment)

        # 4. Governance gate + human approval
        promotion = self.governor.approve_amendment(amendment, evidence_ids=["ev-pipeline"])
        assert promotion.success is True

        # 5. New immutable runtime exists; old one preserved
        new_runtime = self.registry.get_runtime(promotion.new_runtime_id)
        assert new_runtime.version == "v1"
        assert new_runtime.parent_version == "v0"
        assert self.registry.get_runtime("runtime-v0") is not None

        # 6. Rollback works
        rolled = self.governor.rollback("v0")
        assert rolled.version == "v0"
        assert self.registry.get_current().version == "v0"

        # 7. Full audit trail
        trail = self.governor.audit_trail()
        actions = [e["action"] for e in trail]
        assert "promote" in actions
        assert "rollback" in actions
