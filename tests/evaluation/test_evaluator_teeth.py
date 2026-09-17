"""P1 evaluator teeth: auto-derived regression cases, suite coverage, adversarial growth."""
import pytest

from app.evaluation._init import Evaluator, FailureClassRegistry, ReplaySuite
from app.governance.governor import Amendment, AmendmentStatus, Governor, RuntimeRegistry, TargetType
from app.governance.models import Evaluation, Evidence
from app.steward._init import Steward
from constitution.constitution import Constitution


@pytest.fixture()
def evaluator():
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v0", model_identifier="m", constitution_version="v1",
        prompts={"system": "default"}, created_by="system", description="v0",
    )
    suite = ReplaySuite(
        id="core",
        name="core",
        tasks=[
            {"id": "t1", "type": "math", "input": {"expression": "2+2"}, "expected_output": "4"},
            {"id": "t2", "type": "math", "input": {"expression": "10/2"}, "expected_output": "5"},
            {"id": "t3", "type": "math", "input": {"expression": "7*3"}, "expected_output": "21"},
        ],
    )
    fcr = FailureClassRegistry()
    return registry, Evaluator(registry, {"core": suite}, failure_class_registry=fcr), fcr, suite


def _failures(task_id="task-001", count=4, expression="2+2"):
    return [
        {
            "task_id": task_id,
            "success": False,
            "errors": ["Incorrect answer"],
            "runtime_version": "v0",
            "input": {"expression": expression},
        }
        for _ in range(count)
    ]


class TestAutoDerivedRegressionCases:
    def test_regression_cases_attach_to_proposal(self):
        """The steward's proposals carry auto-derived cases reproducing the exact failure."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        steward = Steward(registry, None)

        proposals = steward.recommend_amendments(_failures(task_id="task-math", expression="2+2"))

        assert len(proposals) > 0
        for proposal in proposals:
            assert proposal.regression_cases, "proposal must carry auto-derived regression cases"
            rc = proposal.regression_cases[0]
            assert rc.source == "auto-derived"
            assert rc.source_reference  # pattern_id
            assert rc.failure_class
            assert rc.input_data.get("expression") == "2+2"
            assert rc.expected_output == "4"  # deterministic math expected output

    def test_evaluation_runs_regression_cases_against_candidate(self, evaluator):
        """Regression cases are replayed against the candidate runtime, not hand-waved."""
        registry, ev, fcr, suite = evaluator
        extra = [
            {
                "id": "reg-auto-abc", "type": "math",
                "input": {"expression": "2+2"}, "expected_output": "4",
                "failure_class": "math:incorrect-answer", "regression_case": True,
                "source": {"kind": "auto-derived", "reference": "pattern-task-001"},
            }
        ]
        result = ev.run_suite_against_runtime("core", "v0", extra_tasks=extra)
        assert result.regression_cases_total == 1
        assert result.regression_cases_passed == 1
        assert result.regression_cases_failed == 0


class TestSuiteCoverageMetric:
    def test_coverage_reports_fraction_of_known_classes(self, evaluator):
        """Coverage = (exercised known classes) / (all known classes)."""
        _, ev, fcr, suite = evaluator
        fcr.register_class("math:incorrect-answer")
        fcr.register_class("summarize:empty-output")
        fcr.register_class("api:timeout")

        tasks = list(suite.tasks) + [{"id": "xc-1", "failure_class": "math:incorrect-answer"}]
        coverage = fcr.coverage_for_tasks(tasks)

        assert coverage.total_known == 3
        assert coverage.total_exercised == 1
        assert coverage.fraction == pytest.approx(1 / 3)
        assert "math:incorrect-answer" in coverage.exercised_classes

    def test_coverage_zero_known_classes_is_full(self, evaluator):
        _, _, fcr, _ = evaluator
        coverage = fcr.coverage(set())
        assert coverage.total_known == 0
        assert coverage.fraction == 1.0

    def test_evaluation_reports_coverage(self, evaluator):
        """A suite run tagged with failure classes reports suite coverage."""
        registry, ev, fcr, suite = evaluator
        fcr.register_class("math:incorrect-answer")
        fcr.register_class("api:timeout")

        suite.add_task({"id": "t4", "type": "math", "input": {"expression": "2+2"},
                        "expected_output": "4", "failure_class": "math:incorrect-answer",
                        "regression_case": False})

        result = ev.run_suite_against_runtime("core", "v0")
        exercised = {o.failure_class for o in result.outcomes if o.failure_class}
        coverage = fcr.coverage(exercised)

        assert coverage.total_known == 2
        assert coverage.total_exercised >= 1
        assert coverage.fraction > 0 and coverage.fraction < 1.0
        assert result.total_tasks == len(suite.tasks)


class TestAdversarialSuiteGrowth:
    def test_rejection_grows_permanent_regression_cases(self):
        """Rejected amendments become permanent regression cases so past
        failures can't silently regress in a later amendment."""
        fcr = FailureClassRegistry()
        constitution = Constitution()
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        governor = Governor(registry, constitution, failure_class_registry=fcr)

        from app.governance.models import RegressionCase

        amendment = Amendment(
            id="prop-rej", parent_version="v0", target=TargetType.PROMPT,
            description="broken fix attempt", rationale="math failures",
            proposed_diff={"prompts": {"system": "bad fix"}},
            proposer="steward",
            regression_cases=[
                RegressionCase(id="reg-rej", failure_class="math:incorrect-answer",
                               task_id="reg-rej", task_type="math",
                               input_data={"expression": "5*5"}, expected_output="25"),
            ],
        )
        governor.reject_amendment(amendment, "Fails validation")

        assert "math:incorrect-answer" in fcr.known_classes()
        cases = fcr.rejection_cases_for_class("math:incorrect-answer")
        assert len(cases) == 1
        assert cases[0]["source"] == "rejection-derived"
        assert cases[0]["source_reference"] == "prop-rej"

    def test_registry_never_shrinks(self):
        """The failure class registry is monotonic: classes never get removed."""
        fcr = FailureClassRegistry()
        fcr.register_class("math:incorrect-answer")
        fcr.register_class("api:timeout")
        assert fcr.known_classes() == {"math:incorrect-answer", "api:timeout"}

        # Re-registering an existing class increments occurrences, never drops it.
        fcr.register_class("api:timeout")
        assert fcr.known_classes() == {"math:incorrect-answer", "api:timeout"}
        assert fcr.get_class("api:timeout").occurrence_count == 2