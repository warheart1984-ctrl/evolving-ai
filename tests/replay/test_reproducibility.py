"""Evaluation reproducibility and parent-vs-candidate comparison."""
from app.evaluation._init import Evaluator, ReplaySuite
from app.governance.registry import RuntimeRegistry


def _build():
    suite = ReplaySuite(id="repro-suite", name="Repro", description="", tags=["repro"])
    for i in range(5):
        suite.add_task({
            "id": f"task-{i:03d}",
            "type": "math",
            "input": {"expression": f"{i}+{i}"},
            "expected_output": str(i * 2),
            "expected_latency_ms": 100.0 + i,
            "expected_cost": 0.01,
        })
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v1", model_identifier="test-model",
        constitution_version="v1", created_by="system", description="v1",
    )
    evaluator = Evaluator(registry, {"repro-suite": suite})
    return suite, registry, evaluator


class TestEvaluationReproducibility:
    def test_same_suite_same_runtime_same_results(self):
        """Running the same suite against the same runtime twice yields identical results."""
        _, _, evaluator = _build()
        r1 = evaluator.run_suite_against_runtime("repro-suite", "v1")
        r2 = evaluator.run_suite_against_runtime("repro-suite", "v1")

        assert r1.total_tasks == r2.total_tasks
        assert r1.passed_tasks == r2.passed_tasks
        assert r1.failed_tasks == r2.failed_tasks
        assert r1.correctness_avg == r2.correctness_avg
        assert r1.safety_score == r2.safety_score
        assert [o.actual_output for o in r1.outcomes] == [o.actual_output for o in r2.outcomes]
        assert [o.correctness for o in r1.outcomes] == [o.correctness for o in r2.outcomes]

    def test_parent_vs_candidate_regression_detection(self):
        """A candidate that breaks previously-passing tasks produces regressions."""
        _, _, evaluator = _build()
        parent = evaluator.run_suite_against_runtime("repro-suite", "v1")

        # Replace the suite with one where task-002's expected output is wrong
        bad_suite = ReplaySuite(id="repro-suite", name="Repro", description="", tags=["repro"])
        for i in range(5):
            bad_suite.add_task({
                "id": f"task-{i:03d}",
                "type": "math",
                "input": {"expression": f"{i}+{i}"},
                "expected_output": str(i * 2) if i != 2 else "definitely-wrong",
                "expected_latency_ms": 100.0 + i,
                "expected_cost": 0.01,
            })
        evaluator.suites["repro-suite"] = bad_suite

        candidate = evaluator.run_suite_against_runtime("repro-suite", "v1")

        assert candidate.regressions >= 1
        assert candidate.regressions > parent.regressions
        assert candidate.passed_tasks < parent.passed_tasks
        assert candidate.failed_tasks > parent.failed_tasks
