"""Tests for evaluation/replay system."""
import pytest
from datetime import datetime
from app.evaluation._init import ReplaySuite, ReplayResult, TaskOutcome, Evaluator
from app.governance.registry import RuntimeRegistry


class TestReplaySuite:
    """Test replay suite management."""
    
    def test_suite_creation(self):
        """Test basic suite creation."""
        suite = ReplaySuite(
            id="suite-001",
            name="Core Tests",
            description="Core evaluation suite",
            tags=["core", "regression"],
        )
        assert suite.id == "suite-001"
        assert suite.name == "Core Tests"
        assert len(suite.tasks) == 0
    
    def test_suite_add_task(self):
        """Test adding tasks to suite."""
        suite = ReplaySuite(id="suite-001", name="Tests", description="")
        task = {
            "id": "task-001",
            "type": "math",
            "input": {"expression": "2+2"},
            "expected_output": "4",
            "expected_latency_ms": 100,
        }
        suite.add_task(task)
        assert len(suite.tasks) == 1
        assert suite.tasks[0]["id"] == "task-001"
    
    def test_suite_multiple_tasks(self):
        """Test suite with multiple tasks."""
        suite = ReplaySuite(id="suite-001", name="Tests", description="")
        for i in range(5):
            task = {
                "id": f"task-{i:03d}",
                "type": "math",
                "input": {"expression": f"{i}+{i}"},
                "expected_output": str(i * 2),
            }
            suite.add_task(task)
        assert len(suite.tasks) == 5


class TestTaskOutcome:
    """Test task outcome model."""
    
    def test_outcome_creation(self):
        """Test basic task outcome creation."""
        outcome = TaskOutcome(
            task_id="task-001",
            input={"question": "What is 2+2?"},
            expected_output="4",
            actual_output="4",
            runtime_version="v1",
            correctness=1.0,
            instruction_following=1.0,
            safety_violation=False,
        )
        assert outcome.task_id == "task-001"
        assert outcome.correctness == 1.0
        assert outcome.safety_violation == False
    
    def test_outcome_incorrect(self):
        """Test outcome with incorrect answer."""
        outcome = TaskOutcome(
            task_id="task-001",
            input={"question": "What is 2+2?"},
            expected_output="4",
            actual_output="5",
            runtime_version="v1",
            correctness=0.0,
            instruction_following=0.5,
            safety_violation=False,
        )
        assert outcome.correctness == 0.0


class TestReplayResult:
    """Test replay result model."""
    
    def test_result_creation(self):
        """Test basic replay result creation."""
        outcome1 = TaskOutcome(
            task_id="task-001", input={}, expected_output="a",
            actual_output="a", runtime_version="v1",
            correctness=1.0, instruction_following=1.0,
            safety_violation=False,
        )
        outcome2 = TaskOutcome(
            task_id="task-002", input={}, expected_output="b",
            actual_output="b", runtime_version="v1",
            correctness=0.75, instruction_following=0.9,
            safety_violation=False,
        )
        
        result = ReplayResult(
            suite_id="suite-001",
            runtime_version="v1",
            outcomes=[outcome1, outcome2],
            total_tasks=2,
        )
        assert result.total_tasks == 2
        assert result.passed_tasks == 1  # correctness 1.0 passes; 0.75 fails the 0.80 threshold
        assert result.failed_tasks == 1
        assert result.correctness_avg == 0.875  # (1.0 + 0.75) / 2
    
    def test_result_with_regressions(self):
        """Test regression detection."""
        outcome1 = TaskOutcome(
            task_id="task-001", input={}, expected_output="a",
            actual_output="a", runtime_version="v1",
            correctness=1.0, instruction_following=1.0,
            safety_violation=False,
        )
        outcome2 = TaskOutcome(
            task_id="task-002", input={}, expected_output="b",
            actual_output="wrong", runtime_version="v1",
            correctness=0.3, instruction_following=0.3,
            safety_violation=False,
        )
        
        result = ReplayResult(
            suite_id="suite-001",
            runtime_version="v1",
            outcomes=[outcome1, outcome2],
            total_tasks=2,
        )
        # outcome2 has low correctness - counts as regression
        assert result.regressions == 1


class TestEvaluator:
    """Test evaluator replay execution."""
    
    def test_evaluator_runs_suite(self):
        """Test evaluator runs a replay suite."""
        # Create a suite
        suite = ReplaySuite(
            id="suite-001",
            name="Math Tests",
            description="Math evaluation",
            tags=["math"],
        )
        task = {
            "id": "task-001",
            "type": "math",
            "input": {"expression": "2+2"},
            "expected_output": "4",
        }
        suite.add_task(task)
        
        # Create evaluator
        registry = RuntimeRegistry()
        # Add a runtime
        reg_runtime = registry.create_runtime(
            version="v1", model_identifier="test", constitution_version="v1",
            created_by="system", description="v1",
        )
        evaluator = Evaluator(registry, {"suite-001": suite})
        
        # Run evaluation
        result = evaluator.run_suite_against_runtime("suite-001", "v1")
        
        assert result.suite_id == "suite-001"
        assert result.runtime_version == "v1"
        assert result.total_tasks == 1
        assert len(result.outcomes) == 1
    
    def test_evaluator_metrics_calculation(self):
        """Test that evaluator calculates metrics correctly."""
        suite = ReplaySuite(
            id="suite-001",
            name="Mixed Tests",
            description="",
            tags=["mixed"],
        )
        
        # Add 3 tasks: 2 passing, 1 failing (task-002 has a wrong expected output)
        for i, expected in enumerate(["0", "2", "wrong"]):
            task = {
                "id": f"task-{i:03d}",
                "type": "math",
                "input": {"expression": f"{i}+{i}"},
                "expected_output": expected,
            }
            suite.add_task(task)
        
        registry = RuntimeRegistry()
        evaluator = Evaluator(registry, {"suite-001": suite})
        
        result = evaluator.run_suite_against_runtime("suite-001", "v1")
        
        assert result.total_tasks == 3
        assert result.passed_tasks >= 1  # At least some pass
        assert result.failed_tasks >= 1  # At least some fail
        assert 0 <= result.correctness_avg <= 1.0
        assert 0 <= result.latency_avg_ms