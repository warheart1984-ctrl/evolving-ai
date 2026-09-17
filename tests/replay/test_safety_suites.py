"""Causal replay tests for the constitution-required safety suites."""

import json
from pathlib import Path

from app.evaluation._init import Evaluator, ReplaySuite
from app.governance.registry import RuntimeRegistry


ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = ROOT / "evaluations" / "suites"


def _load_suite(filename: str) -> ReplaySuite:
    data = json.loads((SUITES_DIR / filename).read_text(encoding="utf-8"))
    return ReplaySuite(**data)


def _registry_with_runtime(prompt_policy: str) -> RuntimeRegistry:
    registry = RuntimeRegistry()
    registry.create_runtime(
        version="v1",
        model_identifier="mock-model",
        constitution_version="v1",
        prompts={"safety_policy": prompt_policy},
        created_by="test",
        description="safety replay fixture",
    )
    return registry


def test_safety_suites_are_valid_and_distinct():
    refuse = _load_suite("safety_refuse_v1.json")
    overrefuse = _load_suite("safety_overrefuse_v1.json")

    assert refuse.id == "safety_refuse"
    assert overrefuse.id == "safety_overrefuse"
    assert refuse.tasks and overrefuse.tasks
    assert all(task["type"] == "safety_refuse" for task in refuse.tasks)
    assert all(task["type"] == "safety_allow" for task in overrefuse.tasks)


def test_operator_prompt_policy_drives_refusal_behavior():
    suite = _load_suite("safety_refuse_v1.json")
    evaluator = Evaluator(_registry_with_runtime(
        "weapons_construction,explosives,self_harm,illegal_activity,fraud"
    ), {suite.id: suite})

    result = evaluator.run_suite_against_runtime("safety_refuse", "v1")

    assert result.total_tasks == len(suite.tasks)
    assert result.passed_tasks == result.total_tasks
    assert all(outcome.actual_output.startswith("REFUSED:") for outcome in result.outcomes)


def test_evaluator_observes_prompt_change_through_operator():
    suite = _load_suite("safety_refuse_v1.json")
    registry = _registry_with_runtime("")
    evaluator = Evaluator(registry, {suite.id: suite})

    result = evaluator.run_suite_against_runtime("safety_refuse", "v1")

    assert result.passed_tasks == 0
    assert all(outcome.actual_output.startswith("ALLOWED:") for outcome in result.outcomes)


def test_overrefusal_suite_passes_when_benign_categories_are_not_policy_entries():
    suite = _load_suite("safety_overrefuse_v1.json")
    evaluator = Evaluator(_registry_with_runtime("weapons_construction,explosives"), {suite.id: suite})

    result = evaluator.run_suite_against_runtime("safety_overrefuse", "v1")

    assert result.passed_tasks == result.total_tasks
    assert all(outcome.actual_output.startswith("ALLOWED:") for outcome in result.outcomes)
