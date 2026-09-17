"""FastAPI application for the governed evolving AI runtime."""
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.evaluation._init import Evaluator, ReplaySuite
from app.governance.governor import (
    Amendment,
    AmendmentStatus,
    Governor,
    RuntimeRegistry,
    TargetType,
)
from app.governance.models import Evidence, Evaluation
from app.memory._init import LessonStatus, MemoryStore
from app.operator.operator import Operator
from app.steward._init import Hypothesis, Steward
from app.telemetry._init import ExecutionTelemetry, TelemetryStore
from constitution.constitution import Constitution

# --- Components ---

_constitution_dir = Path(__file__).resolve().parents[2] / "constitution"
constitution = Constitution.load_pinned(
    _constitution_dir / "constitution.yaml",
    _constitution_dir / "constitution.yaml.sha256",
)
registry = RuntimeRegistry()
governor = Governor(registry, constitution)
steward = Steward(registry, constitution)
memory_store = MemoryStore()
telemetry_store = TelemetryStore()

# In-memory amendment registry (v0 prototype; PostgreSQL in later phases)
amendments: Dict[str, Amendment] = {}


def _load_suites() -> Dict[str, ReplaySuite]:
    """Load replay suites from evaluations/suites/*.json."""
    suites: Dict[str, ReplaySuite] = {}
    suites_dir = Path(__file__).resolve().parents[2] / "evaluations" / "suites"
    if suites_dir.exists():
        for path in suites_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            suite = ReplaySuite(
                id=data["id"],
                name=data.get("name", path.stem),
                description=data.get("description", ""),
                tasks=data.get("tasks", []),
                tags=data.get("tags", []),
            )
            suites[suite.id] = suite
    return suites


evaluator = Evaluator(registry, _load_suites())

# Bootstrap the initial runtime (v0)
if registry.get_current() is None:
    registry.create_runtime(
        version="v0",
        model_identifier="gpt-3.5-turbo",
        constitution_version=constitution.version,
        prompts={"system": "system_v1", "planner": "planner_v1"},
        created_by="system",
        description="Initial runtime v0",
    )

# --- App ---

app = FastAPI(
    title="Governed Evolving AI Runtime",
    description="Prototype of a governed, evolving AI runtime with versioned amendments",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Execute task (Operator) ---

@app.post("/execute")
async def execute_task(task_id: str, input_data: Dict[str, Any]):
    """Execute a user task with the current approved runtime (Operator role)."""
    current = registry.get_current()
    if not current:
        raise HTTPException(status_code=404, detail="No runtime configured")

    operator = Operator(registry=registry, current_runtime=current)
    result = operator.execute_task(task_id=task_id, input_data=input_data)

    telemetry = ExecutionTelemetry(
        run_id="",
        runtime_id=current.id,
        runtime_version=current.version,
        task_id=task_id,
        input=input_data,
        output=result.output,
        success=result.success,
        errors=result.errors,
        tools_used=result.tools_used,
        latency_ms=result.latency_ms,
        cost=result.cost,
    )
    run_id = telemetry_store.record_execution(telemetry)

    body = result.model_dump(mode="json")
    body["run_id"] = run_id
    return body


# --- Runtime endpoints ---

@app.get("/runtime/current")
async def get_current_runtime():
    """Get the current immutable runtime configuration."""
    runtime = registry.get_current()
    if not runtime:
        raise HTTPException(status_code=404, detail="No runtime configured")
    return runtime.model_dump(mode="json")


@app.get("/runtime/list")
async def list_runtimes():
    """List all runtime versions in chronological order."""
    return [r.model_dump(mode="json") for r in registry.list_runtimes()]


@app.get("/runtime/{version}")
async def get_runtime_version(version: str):
    """Get a specific runtime version."""
    runtime = registry.get_runtime(f"runtime-{version}")
    if not runtime:
        raise HTTPException(status_code=404, detail=f"Runtime version {version} not found")
    return runtime.model_dump(mode="json")


@app.post("/runtime/rollback/{target_version}")
async def rollback_runtime(target_version: str):
    """Rollback to a previous runtime version."""
    result = governor.rollback(target_version)
    if not result:
        raise HTTPException(status_code=404, detail=f"Cannot rollback to {target_version}")
    return {"status": "rolled_back", "runtime_id": result.id, "version": result.version}


# --- Amendment endpoints ---

@app.post("/amendment/propose")
async def propose_amendment(
    target: TargetType,
    description: str,
    rationale: str,
    proposed_diff: Dict[str, Any],
    proposer: str = "steward",
):
    """Steward proposes an amendment (PROPOSED state; cannot promote)."""
    if target not in (TargetType.PROMPT, TargetType.MEMORY):
        raise HTTPException(
            status_code=403,
            detail="v0 mutation scope is limited to prompt and memory-rule changes",
        )

    current = registry.get_current()
    parent_version = current.version if current else "v0"

    hypothesis = Hypothesis(
        id=f"hyp-{uuid.uuid4().hex[:8]}",
        description=description,
        target=target,
        expected_improvement="Improved performance on failing task class",
        rationale=rationale,
        confidence=0.6,
    )
    proposal = steward.propose_amendment(
        hypothesis=hypothesis,
        target_component=target.value,
        description=description,
        rationale=rationale,
        proposed_diff=proposed_diff,
    )

    amendment = Amendment(
        id=proposal.id,
        parent_version=proposal.parent_version,
        target=target,
        description=description,
        rationale=rationale,
        proposed_diff=proposed_diff,
        proposer=proposer,
    )
    amendments[amendment.id] = amendment
    return amendment.model_dump(mode="json")


@app.get("/amendments")
async def list_amendments(status: Optional[str] = None):
    """List all amendments, optionally filtered by status."""
    items = [a.model_dump(mode="json") for a in amendments.values()]
    if status:
        items = [i for i in items if i["status"] == status]
    return items


@app.get("/amendment/{amendment_id}")
async def get_amendment(amendment_id: str):
    """Inspect a single amendment (including evidence and evaluation)."""
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    return amendment.model_dump(mode="json")


# --- Evaluation endpoints ---

@app.post("/evaluation/run")
async def run_evaluation(suite_id: str, runtime_version: str):
    """Run a replay evaluation suite against a runtime version."""
    try:
        result = evaluator.run_suite_against_runtime(suite_id, runtime_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "status": "completed",
        "suite_id": result.suite_id,
        "runtime_version": result.runtime_version,
        "total_tasks": result.total_tasks,
        "passed_tasks": result.passed_tasks,
        "failed_tasks": result.failed_tasks,
        "correctness_avg": result.correctness_avg,
        "instruction_following_avg": result.instruction_following_avg,
        "safety_score": result.safety_score,
        "latency_avg_ms": result.latency_avg_ms,
        "cost_avg": result.cost_avg,
        "regressions": result.regressions,
        "new_failures": result.new_failures,
        "outcomes": [
            {
                "task_id": o.task_id,
                "correctness": o.correctness,
                "instruction_following": o.instruction_following,
                "safety_violation": o.safety_violation,
                "latency_ms": o.latency_ms,
            }
            for o in result.outcomes
        ],
    }


@app.post("/governance/evaluate/{amendment_id}")
async def evaluate_amendment(amendment_id: str, suite_id: str = "core"):
    """Run the evaluator against parent and candidate, attach evidence, move to REVIEW."""
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")

    try:
        parent_result = evaluator.run_suite_against_runtime(suite_id, amendment.parent_version)
        candidate_label = f"candidate-for-{amendment.parent_version}"
        candidate_result = evaluator.run_suite_against_runtime(suite_id, candidate_label)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    evaluation = Evaluation(
        id=f"eval-{uuid.uuid4().hex[:8]}",
        amendment_id=amendment_id,
        parent_runtime=parent_result.runtime_version,
        candidate_runtime=candidate_result.runtime_version,
        correctness=candidate_result.correctness_avg,
        instruction_following=candidate_result.instruction_following_avg,
        safety=candidate_result.safety_score,
        regressions=candidate_result.regressions,
        evidence=[
            Evidence(
                id=f"ev-{uuid.uuid4().hex[:8]}",
                type="replay",
                description=(
                    f"Replay of suite '{suite_id}': parent "
                    f"{parent_result.runtime_version} vs candidate "
                    f"({parent_result.passed_tasks}/{parent_result.total_tasks} vs "
                    f"{candidate_result.passed_tasks}/{candidate_result.total_tasks})"
                ),
                results={
                    "parent": {
                        "passed": parent_result.passed_tasks,
                        "failed": parent_result.failed_tasks,
                        "total": parent_result.total_tasks,
                        "correctness_avg": parent_result.correctness_avg,
                    },
                    "candidate": {
                        "passed": candidate_result.passed_tasks,
                        "failed": candidate_result.failed_tasks,
                        "total": candidate_result.total_tasks,
                        "correctness_avg": candidate_result.correctness_avg,
                    },
                    "outcomes": [
                        {
                            "task_id": o.task_id,
                            "expected": o.expected_output,
                            "actual": o.actual_output,
                            "correctness": o.correctness,
                        }
                        for o in candidate_result.outcomes
                    ],
                },
                runtime_version=candidate_result.runtime_version,
            )
        ],
        evaluator_id="evaluator:v0",
    )

    amendment.evaluation = evaluation
    amendment.status = AmendmentStatus.REVIEW

    known_failure_classes = {
        r.failure_class for r in telemetry_store.get_failure_records() if r.failure_class
    }
    exercised_failure_classes = {
        task.get("failure_class")
        for task in evaluator.suites.get(suite_id, ReplaySuite(id=suite_id, name=suite_id)).tasks
        if task.get("failure_class")
    }
    coverage = (
        len(known_failure_classes & exercised_failure_classes) / len(known_failure_classes)
        if known_failure_classes else 1.0
    )
    uncovered_failure_classes = sorted(known_failure_classes - exercised_failure_classes)

    return {
        "amendment_id": amendment_id,
        "status": amendment.status.value,
        "correctness": evaluation.correctness,
        "instruction_following": evaluation.instruction_following,
        "safety": evaluation.safety,
        "regressions": evaluation.regressions,
        "evidence": [e.model_dump(mode="json") for e in evaluation.evidence],
        "suite_coverage": {
            "known_failure_classes": len(known_failure_classes),
            "exercised_failure_classes": len(known_failure_classes & exercised_failure_classes),
            "fraction": coverage,
            "uncovered_failure_classes": uncovered_failure_classes,
        },
    }


# --- Governance endpoints ---

@app.post("/governance/approve/{amendment_id}")
async def approve_amendment(
    amendment_id: str,
    reviewer: str = "human",
    evidence_ids: Optional[List[str]] = None,
):
    """Approve and promote an amendment (human approval is mandatory)."""
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    if amendment.evaluation is None:
        raise HTTPException(status_code=400, detail="Amendment has no evaluation; run evaluation first")
    requested_evidence_ids = set(evidence_ids or [])
    available_evidence_ids = {e.id for e in amendment.evaluation.evidence}
    if not requested_evidence_ids or not requested_evidence_ids.issubset(available_evidence_ids):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Approval must reference evaluation evidence IDs",
                "available_evidence_ids": sorted(available_evidence_ids),
            },
        )
    if reviewer == amendment.proposer:
        raise HTTPException(status_code=403, detail="Reviewer cannot be the amendment proposer")

    amendment.status = AmendmentStatus.APPROVED
    amendment.reviewer = reviewer

    promotion = governor.approve_amendment(amendment)
    if not promotion.success:
        return {
            "status": "rejected",
            "amendment_id": amendment_id,
            "reason": promotion.reason,
        }

    return {
        "status": "promoted",
        "amendment_id": amendment_id,
        "new_runtime_id": promotion.new_runtime_id,
        "audit_log": promotion.audit_log,
    }


@app.post("/governance/reject/{amendment_id}")
async def reject_amendment(amendment_id: str, reason: str = "Rejected by reviewer"):
    """Reject an amendment."""
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    governor.reject_amendment(amendment, reason)
    return {"status": "rejected", "amendment_id": amendment_id, "reason": reason}


@app.get("/governance/audit")
async def governance_audit():
    """Full audit trail of promotions and rollbacks."""
    return {"entries": governor.audit_trail()}


# --- Memory endpoints ---

@app.post("/memory/lesson")
async def create_lesson(
    claim: str,
    scope: str,
    created_by: str,
    triggering_task_id: Optional[str] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
):
    """Create a new lesson (starts as candidate; never auto-trusted)."""
    lesson = memory_store.create_lesson(
        claim=claim,
        scope=scope,
        created_by=created_by,
        triggering_task_id=triggering_task_id,
        evidence=evidence,
    )
    return {
        "id": lesson.id,
        "claim": lesson.claim,
        "scope": lesson.scope,
        "status": lesson.status.value,
        "confidence": lesson.confidence,
        "created_at": lesson.created_at.isoformat(),
    }


@app.post("/memory/lesson/{lesson_id}/validate")
async def validate_lesson(lesson_id: str, validator_id: str):
    """Move a lesson from candidate to validated."""
    success = memory_store.validate_lesson(lesson_id, validator_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Lesson {lesson_id} not found or not in candidate state",
        )
    return {
        "status": "validated",
        "lesson_id": lesson_id,
        "times_validated": memory_store.get_lesson(lesson_id).times_validated,
    }


@app.post("/memory/lesson/{lesson_id}/activate")
async def activate_lesson(lesson_id: str):
    """Activate a validated lesson."""
    success = memory_store.activate_lesson(lesson_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not in validated state")
    return {"status": "active", "lesson_id": lesson_id}


@app.post("/memory/lesson/{lesson_id}/quarantine")
async def quarantine_lesson(lesson_id: str, reason: str, quarantiner_id: str):
    """Quarantine a lesson."""
    success = memory_store.quarantine_lesson(lesson_id, reason, quarantiner_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
    return {"status": "quarantined", "lesson_id": lesson_id, "reason": reason}


@app.get("/memory/lessons")
async def list_lessons(status: Optional[str] = None, scope: Optional[str] = None):
    """List lessons filtered by status and/or scope."""
    if status:
        lessons = memory_store.get_lessons_by_status(LessonStatus(status))
    else:
        lessons = list(memory_store._lessons.values())
    if scope:
        lessons = memory_store.get_lessons_by_scope(scope)
    return [
        {
            "id": l.id,
            "claim": l.claim,
            "scope": l.scope,
            "status": l.status.value,
            "confidence": l.confidence,
            "created_by": l.created_by,
            "created_at": l.created_at.isoformat(),
        }
        for l in lessons
    ]


# --- Telemetry endpoints ---

@app.post("/telemetry/record")
async def record_telemetry(
    runtime_version: str,
    task_id: str,
    input_data: Dict[str, Any],
    output: Any,
    success: bool,
    tools_used: List[str] = None,
    latency_ms: float = 0.0,
    cost: float = 0.0,
    errors: List[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
    failure_class: Optional[str] = None,
):
    """Record operator execution telemetry (including failures and user feedback)."""
    telemetry = ExecutionTelemetry(
        run_id="",
        runtime_id=f"runtime-{runtime_version}",
        runtime_version=runtime_version,
        task_id=task_id,
        input=input_data,
        output=output,
        success=success,
        errors=errors or [],
        failure_class=failure_class or TelemetryStore.classify_failure(errors or [], output),
        tools_used=tools_used or [],
        latency_ms=latency_ms,
        cost=cost,
        user_feedback=user_feedback,
    )
    run_id = telemetry_store.record_execution(telemetry)
    return {"status": "recorded", "run_id": run_id}


@app.get("/telemetry/failures")
async def get_failure_telemetry(runtime_version: Optional[str] = None):
    """Get failure telemetry records."""
    records = telemetry_store.get_failure_records(runtime_version)
    return [
        {
            "run_id": r.run_id,
            "runtime_version": r.runtime_version,
            "task_id": r.task_id,
            "success": r.success,
            "errors": r.errors,
            "failure_class": r.failure_class,
            "tools_used": r.tools_used,
            "latency_ms": r.latency_ms,
            "cost": r.cost,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


@app.get("/telemetry/runs")
async def get_all_telemetry(runtime_version: Optional[str] = None):
    """Get all telemetry records, optionally filtered by runtime version."""
    if runtime_version:
        records = telemetry_store.get_records_by_runtime(runtime_version)
    else:
        records = list(telemetry_store._records.values())
    return [
        {
            "run_id": r.run_id,
            "runtime_version": r.runtime_version,
            "task_id": r.task_id,
            "success": r.success,
            "errors": r.errors,
            "latency_ms": r.latency_ms,
            "cost": r.cost,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


# --- Steward loop ---

@app.post("/steward/analyze")
async def steward_analyze():
    """Run the steward analysis loop: telemetry → patterns → proposals (never promotion)."""
    failures = telemetry_store.get_failure_records()
    if not failures:
        return {"status": "no_failures", "failures": 0, "proposals": []}

    proposals = steward.recommend_amendments([dict(r) for r in failures])

    created = []
    for p in proposals:
        amendment = Amendment(
            id=p.id,
            parent_version=p.parent_version,
            target=p.target,
            description=p.description,
            rationale=p.rationale,
            proposed_diff=p.proposed_diff,
            proposer="steward",
        )
        amendments[amendment.id] = amendment
        regression_case = p.proposed_diff.get("regression_case")
        if regression_case:
            suite = evaluator.suites.setdefault(
                "core", ReplaySuite(id="core", name="Core regression suite")
            )
            if not any(t.get("id") == regression_case.get("id") for t in suite.tasks):
                suite.add_task(regression_case)
        created.append(amendment.id)

    return {"status": "analyzed", "failures": len(failures), "proposals": created}


# --- Dashboard ---

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Minimal Governor dashboard."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# --- Health ---

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Governed Evolving AI Runtime v0.1",
        "current_runtime": registry.get_current().version if registry.get_current() else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
