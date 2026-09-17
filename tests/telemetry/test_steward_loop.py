"""P5 telemetry → steward loop: failure classification tags and proposal rate limiting."""
import datetime

import pytest

from app.governance.models import TargetType
from app.governance.registry import RuntimeRegistry
from app.steward._init import Steward
from app.telemetry._init import TelemetryStore, classify_failure


class TestFailureClassification:
    def test_math_incorrect_answer(self):
        fc = classify_failure("task-001", {"expression": "2+2"}, ["Incorrect answer"], "5")
        assert fc == "math:incorrect-answer"

    def test_api_timeout(self):
        fc = classify_failure("task-002", {"url": "http://x"}, ["Timeout exceeded"], None)
        assert fc == "api:timeout"

    def test_summarize_empty_output(self):
        fc = classify_failure("task-003", {"text": "abc"}, ["Empty output"], "")
        assert fc == "summarize:empty-output"

    def test_classify_failure_defaults(self):
        fc = classify_failure("unseen", {}, ["mystery error"], None)
        assert fc.endswith("incorrect-answer")

    def test_telemetry_store_auto_tags_failures(self):
        store = TelemetryStore()
        run_id = store.record_execution(
            __import__("app.telemetry._init", fromlist=["ExecutionTelemetry"]).ExecutionTelemetry(
                run_id="", runtime_id="runtime-v0", runtime_version="v0",
                task_id="task-001", input={"expression": "2+2"},
                output="5", success=False, errors=["Incorrect answer"],
            )
        )
        rec = store.get_record(run_id)
        assert rec.failure_class == "math:incorrect-answer"

    def test_failure_class_summary_counts(self):
        from app.telemetry._init import ExecutionTelemetry
        store = TelemetryStore()
        for i in range(3):
            store.record_execution(ExecutionTelemetry(
                run_id="", runtime_id="runtime-v0", runtime_version="v0",
                task_id=f"t-{i}", input={"expression": "2+2"}, output="5",
                success=False, errors=["Incorrect answer"],
            ))
        counts = store.get_failure_classes()
        assert counts == {"math:incorrect-answer": 3}


@pytest.fixture()
def steward():
    registry = RuntimeRegistry()
    registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                            created_by="system", description="v0")
    return Steward(registry, None)


class TestProposalRateLimiting:
    def test_single_burst_allows_one_proposal(self, steward):
        """One analyze pass per failure class within the window is allowed."""
        failures = [
            {"task_id": "task-a", "success": False, "errors": ["Incorrect answer"],
             "runtime_version": "v0", "input": {"expression": "2+2"}}
        ] * 4
        proposals = steward.recommend_amendments(failures)


        # recommend_amendments makes 2 proposals per pattern (PROMPT + MEMORY)
        assert len(proposals) >= 2

    def test_second_burst_same_class_is_rate_limited(self, steward):
        """A second proposal request for the same failure class within the
        window is skipped (rate-limited)."""
        failures = [
            {"task_id": "task-a", "success": False, "errors": ["Incorrect answer"],
             "runtime_version": "v0", "input": {"expression": "2+2"}}
        ] * 4

        first = steward.recommend_amendments(failures)
        assert len(first) >= 2

        # Bump the recorded timestamps without modifying the caller's clock:
        # instead of waiting 1 hour, re-run immediately — rate limit must hold.
        second = steward.recommend_amendments(failures)
        assert len(second) == 0

    def test_rate_limit_respects_class(self, steward):
        """Different failure classes don't throttle each other."""
        failures_a = [
            {"task_id": "task-a", "success": False, "errors": ["Incorrect answer"],
             "runtime_version": "v0", "input": {"expression": "2+2"}}
        ] * 4
        failures_b = [
            {"task_id": "task-b", "success": False, "errors": ["Timeout exceeded"],
             "runtime_version": "v0", "input": {"url": "http://x"}}
        ] * 4

        first_a = steward.recommend_amendments(failures_a)
        # Use the same steward instance; failures_b is a different class.
        first_b = steward.recommend_amendments(failures_b)
        assert len(first_a) >= 2
        assert len(first_b) >= 2

    def test_timestamps_age_out_of_window(self, steward):
        """After the window elapses, proposals for that class are allowed again."""
        failures = [
            {"task_id": "task-a", "success": False, "errors": ["Incorrect answer"],
             "runtime_version": "v0", "input": {"expression": "2+2"}}
        ] * 4
        steward.recommend_amendments(failures)

        # Move recorded timestamps outside the window.
        old = datetime.datetime.utcnow() - datetime.timedelta(seconds=steward.RATE_LIMIT_WINDOW_SECONDS * 2)
        for key in steward._proposal_timestamps:
            steward._proposal_timestamps[key] = [old]

        again = steward.recommend_amendments(failures)
        assert len(again) >= 2