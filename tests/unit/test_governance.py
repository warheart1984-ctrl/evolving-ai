"""Tests for governance module - constitution, gates, approvals."""
import pytest
from datetime import datetime
from constitution.constitution import Constitution
from app.governance.governor import (
    Amendment, AmendmentStatus, TargetType, ConstitutionGates, 
    Evaluation, Evidence, GateResult, RuntimeManifest, PromotionResult
)
from app.governance.registry import RuntimeRegistry


class TestConstitution:
    """Test constitution model and gates."""
    
    def test_constituion_creation(self):
        """Test basic constitution creation."""
        c = Constitution()
        assert c.version == "v1"
        assert c.promotion_gates["human_approval_mandatory"] == True
    
    def test_gates_check_required_tests(self):
        """Test required_tests gate passes with good evaluation."""
        eval_pass = Evaluation(
            id="eval-1",
            amendment_id="prop-1",
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=0.90,
            instruction_following=0.88,
            robustness=0.90,
            safety=1.0,
            regressions=0,
            evidence=[],
            passed_gates=["required_tests_pass"],
            failed_gates=[],
        )
        result = ConstitutionGates.check_required_tests(eval_pass)
        assert result.passed == True
    
    def test_gates_check_required_tests_fail(self):
        """Test required_tests gate fails with poor evaluation."""
        eval_fail = Evaluation(
            id="eval-1",
            amendment_id="prop-1",
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=0.50,
            instruction_following=0.40,
            robustness=0.50,
            safety=1.0,
            regressions=0,
            evidence=[],
            passed_gates=[],
            failed_gates=[],
        )
        result = ConstitutionGates.check_required_tests(eval_fail)
        assert result.passed == False

    def test_required_tests_fails_closed_without_robustness(self):
        """A missing robustness score must fail the gate (fail closed)."""
        eval_no_robustness = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.95, instruction_following=0.95,
            safety=1.0, regressions=0, evidence=[],
        )
        result = ConstitutionGates.check_required_tests(eval_no_robustness)
        assert result.passed == False
    
    def test_gates_check_no_safety_failures(self):
        """Test no_safety_failures gate."""
        eval_safe = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.90, instruction_following=0.88,
            safety=1.0, regressions=0, evidence=[], passed_gates=[], failed_gates=[],
        )
        result = ConstitutionGates.check_no_safety_failures(eval_safe)
        assert result.passed == True
        
        eval_unsafe = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.90, instruction_following=0.88,
            safety=0.5, regressions=0, evidence=[], passed_gates=[], failed_gates=[],
        )
        result = ConstitutionGates.check_no_safety_failures(eval_unsafe)
        assert result.passed == False
    
    def test_gates_check_no_unexplained_regressions(self):
        """Test no_unexplained_regressions gate."""
        eval_no_reg = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.90, instruction_following=0.88,
            safety=1.0, regressions=0, evidence=[], passed_gates=[], failed_gates=[],
        )
        result = ConstitutionGates.check_no_unexplained_regressions(eval_no_reg)
        assert result.passed == True
        
        eval_with_reg = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.90, instruction_following=0.88,
            safety=1.0, regressions=3, evidence=[], passed_gates=[], failed_gates=[],
        )
        result = ConstitutionGates.check_no_unexplained_regressions(eval_with_reg)
        assert result.passed == False
    
    def test_gates_all_checks(self):
        """Test all gates together."""
        eval_good = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.92, instruction_following=0.89,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
            passed_gates=[], failed_gates=[],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_good)
        passed = [name for name, gate in all_gates.items() if gate.passed]
        assert "required_tests_pass" in passed
        assert "no_safety_failures" in passed
        assert "no_unexplained_regressions" in passed
        assert "evaluation_evidence_exists" in passed

    def test_all_gates_bind_all_constitution_gates(self):
        """Every gate declared in the constitution must be evaluated."""
        from constitution.constitution import Constitution
        c = Constitution()
        declared = set(c.promotion_gates.keys())
        eval_good = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.92, instruction_following=0.89,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_good, c)
        assert set(all_gates.keys()) == declared | {"latency_within_budget", "cost_within_budget"}

    def test_delta_gates_fail_closed_without_parent_metrics(self):
        """Missing parent-relative metrics must fail the delta gates closed."""
        eval_missing = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.92, instruction_following=0.89,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
        )
        gates = ConstitutionGates.check_all_gates(eval_missing)
        assert gates["latency_within_budget"].passed is False
        assert gates["cost_within_budget"].passed is False

    def test_delta_gates_pass_within_budget(self):
        eval_budget = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.92, instruction_following=0.89,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=110.0, cost_per_task=0.011,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
        )
        gates = ConstitutionGates.check_all_gates(eval_budget)
        assert gates["latency_within_budget"].passed is True   # +10% <= 20%
        assert gates["cost_within_budget"].passed is True      # +10% <= 15%

    def test_delta_gates_fail_over_budget(self):
        eval_over = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.92, instruction_following=0.89,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=150.0, cost_per_task=0.03,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
        )
        gates = ConstitutionGates.check_all_gates(eval_over)
        assert gates["latency_within_budget"].passed is False   # +50% > 20%
        assert gates["cost_within_budget"].passed is False      # +200% > 15%


class TestAmendment:
    """Test amendment model."""
    
    def test_amendment_creation(self):
        """Test basic amendment creation."""
        amend = Amendment(
            id="prop-001",
            parent_version="v0",
            target=TargetType.PROMPT,
            description="Test prompt change",
            rationale="Improve response quality",
            proposed_diff={"prompt_system": "new_system_prompt"},
            proposer="steward:v0",
        )
        assert amend.id == "prop-001"
        assert amend.parent_version == "v0"
        assert amend.target == TargetType.PROMPT
        assert amend.status == AmendmentStatus.PROPOSED
    
    def test_amendment_with_evaluation(self):
        """Test amendment with evaluation results."""
        eval_result = Evaluation(
            id="eval-001",
            amendment_id="prop-001",
            parent_runtime="v0",
            candidate_runtime="v1",
            correctness=0.93,
            instruction_following=0.90,
            safety=1.0,
            regressions=0,
            evidence=[],
            passed_gates=["required_tests_pass", "no_safety_failures", "no_unexplained_regressions", "evaluation_evidence_exists"],
            failed_gates=[],
        )
        amend = Amendment(
            id="prop-001",
            parent_version="v0",
            target=TargetType.PROMPT,
            description="Test prompt change",
            rationale="Improve response quality",
            proposed_diff={"prompt_system": "new_system_prompt"},
            proposer="steward:v0",
            evaluation=eval_result,
        )
        assert amend.evaluation is not None
        assert amend.evaluation.correctness == 0.93


class TestRuntimeManifest:
    """Test runtime manifest model."""
    
    def test_runtime_creation(self):
        """Test runtime manifest creation."""
        runtime = RuntimeManifest(
            id="runtime-v0",
            version="v0",
            model_identifier="gpt-4",
            constitution_version="v1",
            created_by="system",
            description="Initial runtime",
        )
        assert runtime.version == "v0"
        assert runtime.model_identifier == "gpt-4"
        assert runtime.id == "runtime-v0"
    
    def test_runtime_with_prompts(self):
        """Test runtime with prompt versions."""
        runtime = RuntimeManifest(
            id="runtime-v1",
            version="v1",
            model_identifier="gpt-4",
            constitution_version="v1",
            prompts={"system": "prompt-42", "planner": "prompt-18"},
            created_by="steward:v0",
            description="Runtime with prompts",
        )
        assert runtime.prompts["system"] == "prompt-42"
        assert runtime.prompts["planner"] == "prompt-18"


class TestConstitutionGatesAll:
    """Comprehensive governance gate tests."""
    
    def test_all_gates_pass_with_good_evaluation(self):
        """When all metrics are good, all gates pass."""
        eval_result = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.95, instruction_following=0.92,
            robustness=0.9, safety=1.0, regressions=0,
            latency_ms=100.0, cost_per_task=0.01,
            parent_latency_ms=100.0, parent_cost_per_task=0.01,
            evidence=[Evidence(id="ev-1", type="replay", description="replay", runtime_version="v1")],
            passed_gates=[],
            failed_gates=[],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_result)
        assert all(gate.passed for gate in all_gates.values())
    
    def test_fails_on_low_correctness(self):
        """Gate fails when correctness is too low."""
        eval_result = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.50, instruction_following=0.80,
            safety=1.0, regressions=0, evidence=[], passed_gates=[],
            failed_gates=[],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_result)
        assert not all_gates["required_tests_pass"].passed
    
    def test_fails_on_safety_violation(self):
        """Gate fails when safety is violated."""
        eval_result = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.95, instruction_following=0.92,
            safety=0.0, regressions=0, evidence=[], passed_gates=[],
            failed_gates=[],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_result)
        assert not all_gates["no_safety_failures"].passed
    
    def test_fails_on_regressions(self):
        """Gate fails when there are regressions."""
        eval_result = Evaluation(
            id="eval-1", amendment_id="prop-1", parent_runtime="v0",
            candidate_runtime="v1", correctness=0.95, instruction_following=0.92,
            safety=1.0, regressions=5, evidence=[], passed_gates=[],
            failed_gates=[],
        )
        all_gates = ConstitutionGates.check_all_gates(eval_result)
        assert not all_gates["no_unexplained_regressions"].passed


class TestPromotion:
    """Test promotion result and governor logic."""
    
    def test_promotion_result_success(self):
        """Test successful promotion result."""
        result = PromotionResult(success=True, new_runtime_id="runtime-v1")
        assert result.success == True
        assert result.new_runtime_id == "runtime-v1"
    
    def test_promotion_result_failure(self):
        """Test failed promotion result."""
        result = PromotionResult(success=False, reason="Gates not passed")
        assert result.success == False
        assert result.reason == "Gates not passed"


class TestRuntimeRegistry:
    """Test runtime registry management."""
    
    def test_registry_creates_runtimes(self):
        """Test that registry creates new runtimes."""
        reg = RuntimeRegistry()
        runtime = reg.create_runtime(
            version="v1",
            model_identifier="gpt-4",
            constitution_version="v1",
            created_by="steward:v0",
            description="First runtime",
        )
        assert runtime.version == "v1"
        assert runtime.id == "runtime-v1"
        assert reg._current_version == "v1"
    
    def test_registry_creates_version_chain(self):
        """Test that runtimes form a version chain."""
        reg = RuntimeRegistry()
        
        r1 = reg.create_runtime(
            version="v1", model_identifier="gpt-4", constitution_version="v1",
            created_by="steward:v0", description="First",
        )
        r2 = reg.create_runtime(
            version="v2", model_identifier="gpt-4", constitution_version="v1",
            created_by="steward:v1", description="Second",
        )
        r3 = reg.create_runtime(
            version="v3", model_identifier="gpt-4", constitution_version="v1",
            created_by="steward:v2", description="Third",
        )
        
        # Check current is v3
        current = reg.get_current()
        assert current.version == "v3"
        
        # Check chain
        chain = reg.get_amendment_chain("runtime-v3")
        assert len(chain) == 3
        assert chain[0].version == "v1"
        assert chain[1].version == "v2"
        assert chain[2].version == "v3"
    
    def test_registry_rollback(self):
        """Test rollback functionality."""
        reg = RuntimeRegistry()
        
        r1 = reg.create_runtime(version="v1", model_identifier="gpt-4", constitution_version="v1",
                               created_by="system", description="v1")
        r2 = reg.create_runtime(version="v2", model_identifier="gpt-4", constitution_version="v1",
                               created_by="system", description="v2")
        r3 = reg.create_runtime(version="v3", model_identifier="gpt-4", constitution_version="v1",
                               created_by="system", description="v3")
        
        # Rollback to v2
        rolled_back = reg.rollback_to("v2")
        assert rolled_back.version == "v2"
        assert reg._current_version == "v2"
        
        # Rollback to v1
        rolled_back = reg.rollback_to("v1")
        assert rolled_back.version == "v1"
        assert reg._current_version == "v1"