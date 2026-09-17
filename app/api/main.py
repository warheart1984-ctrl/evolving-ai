"""FastAPI application for the governed evolving AI runtime.

Wires together all components with:
- SQLite persistence for audit-critical data
- API key auth on sensitive endpoints
- Suite coverage metric on evaluation
- Evidence-linked approval
- Amendment diff surfacing
"""
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.auth import require_governance_key
from app.evaluation._init import Evaluator, FailureClassRegistry, ReplaySuite
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
from app.storage.store import StateStore
from app.telemetry._init import ExecutionTelemetry, TelemetryStore
from constitution.constitution import Constitution

# --- Persistent storage ---

DATA_DIR = os.environ.get("EVOLVING_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
DB_PATH = os.environ.get("EVOLVING_DB", os.path.join(DATA_DIR, "evolving.db"))
store = StateStore(DB_PATH)

# --- Constitution (hash-pinned at boot) ---

CONSTITUTION_PATH = Path(__file__).resolve().parents[2] / "constitution" / "constitution.yaml"
PIN_PATH = Path(str(CONSTITUTION_PATH) + ".sha256")

constitution = Constitution.from_file(CONSTITUTION_PATH, pin_path=str(PIN_PATH))

# --- Components with persistence ---

failure_class_registry = FailureClassRegistry()
registry = RuntimeRegistry(persistence=store)
governor = Governor(registry, constitution, persistence=store, failure_class_registry=failure_class_registry)
steward = Steward(registry, constitution)
memory_store = MemoryStore(persistence=store)
telemetry_store = TelemetryStore(persistence=store)

# In-memory amendment registry with write-through persistence
_amendments_raw = store.load_all("amendment")
amendments: Dict[str, Amendment] = {
    k: Amendment(**v) for k, v in _amendments_raw.items()
}


def _persist_amendment(amendment: Amendment):
    """Write-through persist an amendment."""
    store.save("amendment", amendment.id, amendment.model_dump(mode="json"))


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


suites = _load_suites()
evaluator = Evaluator(registry, suites, failure_class_registry=failure_class_registry)

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

# P7: CORS is explicit and fail-closed. No credentials + no wildcard by default.
_cors_origins = [
    o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]

app = FastAPI(
    title="Governed Evolving AI Runtime",
    description="Prototype of a governed, evolving AI runtime with versioned amendments",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# --- Execute task (Operator) ---

@app.post("/execute")
async def execute_task(task_id: str, input_data: Dict[str, Any]):
    """Execute a user task with the current approved runtime (Operator role)."""
    current = registry.get_current()
    if not current:
        raise HTTPException(status_code=404, detail="No runtime configured")

    operator = Operator(registry=registry, current_runtime=current)
    operator.set_constitution_hash(constitution.content_hash)
    result = operator.execute_task(task_id=task_id, input_data=input_data)

    telemetry = operator.build_telemetry(result, input_data)
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
async def rollback_runtime(
    target_version: str,
    initiated_by: str = "api-user",
    _key: str = Depends(require_governance_key),
):
    """Rollback to a previous runtime version. Requires governance API key."""
    result = governor.rollback(target_version, initiated_by=initiated_by)
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
    _key: str = Depends(require_governance_key),
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
        regression_cases=proposal.regression_cases,
    )
    amendments[amendment.id] = amendment
    _persist_amendment(amendment)
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
    """Inspect a single amendment (including evidence, evaluation, regression cases)."""
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    return amendment.model_dump(mode="json")


# --- Evaluation endpoints ---

@app.post("/evaluation/run")
async def run_evaluation(suite_id: str, runtime_version: str, _key: str = Depends(require_governance_key)):
    """Run a replay evaluation suite against a runtime version."""
    try:
        result = evaluator.run_suite_against_runtime(suite_id, runtime_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Compute coverage
    known = sorted(failure_class_registry.known_classes())
    exercised = {o.failure_class for o in result.outcomes if o.failure_class}
    coverage = failure_class_registry.coverage(exercised)

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
        "regression_cases_total": result.regression_cases_total,
        "regression_cases_passed": result.regression_cases_passed,
        "regression_cases_failed": result.regression_cases_failed,
        "coverage": {
            "known_classes": coverage.known_classes,
            "exercised_classes": coverage.exercised_classes,
            "fraction": coverage.fraction,
        },
        "outcomes": [
            {
                "task_id": o.task_id,
                "correctness": o.correctness,
                "instruction_following": o.instruction_following,
                "safety_violation": o.safety_violation,
                "latency_ms": o.latency_ms,
                "failure_class": o.failure_class,
                "is_regression_case": o.is_regression_case,
            }
            for o in result.outcomes
        ],
    }


@app.post("/governance/evaluate/{amendment_id}")
async def evaluate_amendment(amendment_id: str, suite_id: str = "core", _key: str = Depends(require_governance_key)):
    """Run the evaluator against parent and candidate with auto-derived regression cases.

    The evaluation augments the base suite with the amendment's regression cases
    and any rejection-derived cases, then reports suite coverage.
    """
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")

    # Build extra tasks from amendment's regression cases
    extra_tasks = []
    for rc in (amendment.regression_cases or []):
        task_dict = {
            "id": rc.task_id if hasattr(rc, "task_id") else rc.get("id", "unknown"),
            "type": rc.task_type if hasattr(rc, "task_type") else rc.get("task_type", "general"),
            "input": rc.input_data if hasattr(rc, "input_data") else rc.get("input_data", {}),
            "expected_output": rc.expected_output if hasattr(rc, "expected_output") else rc.get("expected_output"),
            "failure_class": rc.failure_class if hasattr(rc, "failure_class") else rc.get("failure_class"),
            "regression_case": True,
            "source": {"kind": rc.source if hasattr(rc, "source") else rc.get("source", "auto-derived"),
                        "reference": rc.source_reference if hasattr(rc, "source_reference") else rc.get("source_reference", "")},
        }
        extra_tasks.append(task_dict)

    # Add rejection-derived regression cases for the same failure classes
    failure_classes = {rc.failure_class for rc in (amendment.regression_cases or [])
                       if hasattr(rc, "failure_class") and rc.failure_class}
    for fc in failure_classes:
        for rejection_case in failure_class_registry.rejection_cases_for_class(fc):
            if rejection_case.get("id") not in {t.get("id") for t in extra_tasks}:
                extra_tasks.append(rejection_case)

    try:
        candidate = governor.materialize_candidate(amendment)
        candidate_label = candidate.version

        # Constitution-declared required suites, with the requested suite first
        # so aggregate metrics keep reflecting the primary suite.
        required = list(
            (getattr(constitution, "evaluation_rules", {}) or {}).get(
                "required_suites", ["core"]
            ) or ["core"]
        )
        run_suites = [suite_id] + [s for s in required if s != suite_id]

        parent_results = {}
        candidate_results = {}
        for sid in run_suites:
            # Regression cases describe core-task failures; only attach them to
            # the core suite so safety suites measure pure refusal behavior.
            extras = extra_tasks if sid == "core" else []
            parent_results[sid] = evaluator.run_suite_against_runtime(
                sid, amendment.parent_version, extra_tasks=extras
            )
            candidate_results[sid] = evaluator.run_suite_against_runtime(
                sid, candidate_label, extra_tasks=extras
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    parent_result = parent_results[suite_id]
    candidate_result = candidate_results[suite_id]

    # Compute coverage
    known = sorted(failure_class_registry.known_classes())
    exercised = {o.failure_class for o in candidate_result.outcomes if o.failure_class}
    coverage = failure_class_registry.coverage(exercised)

    evidence = []
    for sid in run_suites:
        pr = parent_results[sid]
        cr = candidate_results[sid]
        evidence.append(
            Evidence(
                id=f"ev-{uuid.uuid4().hex[:8]}",
                type="replay",
                description=(
                    f"Replay of suite '{sid}': parent "
                    f"{pr.runtime_version} vs candidate "
                    f"({pr.passed_tasks}/{pr.total_tasks} vs "
                    f"{cr.passed_tasks}/{cr.total_tasks}). "
                    f"Coverage: {coverage.total_exercised}/{coverage.total_known} known failure classes"
                ),
                results={
                    "suite_id": sid,
                    "parent": {
                        "passed": pr.passed_tasks,
                        "failed": pr.failed_tasks,
                        "total": pr.total_tasks,
                        "correctness_avg": pr.correctness_avg,
                    },
                    "candidate": {
                        "passed": cr.passed_tasks,
                        "failed": cr.failed_tasks,
                        "total": cr.total_tasks,
                        "correctness_avg": cr.correctness_avg,
                    },
                    "coverage": {
                        "known_classes": coverage.known_classes,
                        "exercised_classes": coverage.exercised_classes,
                        "fraction": coverage.fraction,
                    },
                    "regression_cases": {
                        "total": cr.regression_cases_total,
                        "passed": cr.regression_cases_passed,
                        "failed": cr.regression_cases_failed,
                    },
                    "outcomes": [
                        {
                            "task_id": o.task_id,
                            "expected": o.expected_output,
                            "actual": o.actual_output,
                            "correctness": o.correctness,
                            "failure_class": o.failure_class,
                            "is_regression_case": o.is_regression_case,
                        }
                        for o in cr.outcomes
                    ],
                },
                runtime_version=cr.runtime_version,
            )
        )

    evaluation = Evaluation(
        id=f"eval-{uuid.uuid4().hex[:8]}",
        amendment_id=amendment_id,
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
        coverage_known_classes=coverage.total_known,
        coverage_exercised_classes=coverage.total_exercised,
        coverage_fraction=coverage.fraction,
        evidence=evidence,
        candidate_manifest_hash=candidate.manifest_hash,
        evaluator_id="evaluator:v0",
    )

    amendment.evaluation = evaluation
    amendment.status = AmendmentStatus.REVIEW
    _persist_amendment(amendment)

    return {
        "amendment_id": amendment_id,
        "status": amendment.status.value,
        "correctness": evaluation.correctness,
        "instruction_following": evaluation.instruction_following,
        "safety": evaluation.safety,
        "regressions": evaluation.regressions,
        "required_suites": run_suites,
        "coverage": {
            "known_classes": coverage.known_classes,
            "exercised_classes": coverage.exercised_classes,
            "fraction": coverage.fraction,
        },
        "regression_cases": {
            "total": candidate_result.regression_cases_total,
            "passed": candidate_result.regression_cases_passed,
            "failed": candidate_result.regression_cases_failed,
        },
        "evidence": [e.model_dump(mode="json") for e in evaluation.evidence],
    }


# --- Governance endpoints ---

@app.get("/governance/diff/{amendment_id}")
async def amendment_diff(amendment_id: str):
    """Show exactly what will change between current and candidate runtime.

    This surfaces the diff as a first-class review artifact before approval.
    """
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    return governor.amendment_diff(amendment)


@app.post("/governance/approve/{amendment_id}")
async def approve_amendment(
    amendment_id: str,
    reviewer: str = "human",
    evidence_id: List[str] = [],
    _key: str = Depends(require_governance_key),
):
    """Approve and promote an amendment.

    Requires:
    - Governance API key (X-API-Key header)
    - Reviewer identity (not equal to proposer)
    - Evidence IDs referencing specific evaluation evidence
    """
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    if amendment.evaluation is None:
        raise HTTPException(status_code=400, detail="Amendment has no evaluation; run evaluation first")

    amendment.status = AmendmentStatus.APPROVED
    amendment.reviewer = reviewer
    _persist_amendment(amendment)

    promotion = governor.approve_amendment(amendment, evidence_ids=evidence_id, reviewer=reviewer)
    if not promotion.success:
        return {
            "status": "rejected",
            "amendment_id": amendment_id,
            "reason": promotion.reason,
        }

    _persist_amendment(amendment)

    return {
        "status": "promoted",
        "amendment_id": amendment_id,
        "new_runtime_id": promotion.new_runtime_id,
        "audit_log": promotion.audit_log,
    }


@app.post("/governance/reject/{amendment_id}")
async def reject_amendment(
    amendment_id: str,
    reason: str = "Rejected by reviewer",
    _key: str = Depends(require_governance_key),
):
    """Reject an amendment.

    Every REJECTED amendment's failure cases become permanent regression
    cases in the adversarial suite, so past rejections can't silently
    regress in a later amendment.

    Requires governance API key.
    """
    amendment = amendments.get(amendment_id)
    if not amendment:
        raise HTTPException(status_code=404, detail=f"Amendment {amendment_id} not found")
    result = governor.reject_amendment(amendment, reason)
    _persist_amendment(amendment)
    return {"status": "rejected", "amendment_id": amendment_id, "reason": reason}


@app.get("/governance/audit")
async def governance_audit():
    """Full audit trail of promotions, rejections, and rollbacks."""
    return {"entries": governor.audit_trail()}


@app.get("/governance/failure-classes")
async def list_failure_classes():
    """List all known failure classes and their rejection cases."""
    classes = sorted(failure_class_registry.known_classes())
    return {
        "classes": classes,
        "total": len(classes),
        "rejection_cases": {
            fc: failure_class_registry.rejection_cases_for_class(fc)
            for fc in classes
        },
    }


# --- Memory endpoints ---

@app.post("/memory/lesson")
async def create_lesson(
    claim: str,
    scope: str,
    created_by: str,
    triggering_task_id: Optional[str] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    _key: str = Depends(require_governance_key),
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
async def validate_lesson(lesson_id: str, validator_id: str, _key: str = Depends(require_governance_key)):
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
async def activate_lesson(lesson_id: str, _key: str = Depends(require_governance_key)):
    """Activate a validated lesson."""
    success = memory_store.activate_lesson(lesson_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not in validated state")
    return {"status": "active", "lesson_id": lesson_id}


@app.post("/memory/lesson/{lesson_id}/quarantine")
async def quarantine_lesson(
    lesson_id: str,
    reason: str,
    quarantiner_id: str,
    _key: str = Depends(require_governance_key),
):
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

_MAX_TELEMETRY_BYTES = 64 * 1024  # 64 KB payload cap


@app.post("/telemetry/record")
async def record_telemetry(
    payload: dict,
    request: Request,
    _key: str = Depends(require_governance_key),
):
    """Record execution telemetry from a governed subsystem.

    Requires the governance API key. Submitting a payload is trust-scored:
    - payloads carrying an operator/ evaluator/ steward source identity with
      a runtime_manifest_hash that resolves to a real, matching runtime are
      treated as ``trusted`` (eligible for steward analysis).
    - anything unmatched is persisted to the untrusted partition (audit only;
      never aggregated into failure classes or steward proposals).
    - rate-limited per client IP and capped at 64 KB / 50 fields.
    """
    from app.telemetry._init import (
        _external_telemetry_limiter,
        MAX_TELEMETRY_PAYLOAD_BYTES,
        MAX_TELEMETRY_FIELDS,
        manifest_hash as _manifest_hash,
    )

    # --- payload limits ---
    raw_size = len(json.dumps(payload, sort_keys=True))
    if raw_size > MAX_TELEMETRY_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Telemetry payload exceeds 64 KB limit")
    if len(payload) > MAX_TELEMETRY_FIELDS:
        raise HTTPException(status_code=413, detail="Telemetry payload has too many fields")

    # --- rate limit ---
    client_ip = request.client.host if request.client else "unknown"
    if not _external_telemetry_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Telemetry rate limit exceeded")

    source = payload.get("source", "external")
    source_identity = payload.get("source_identity") or {}
    runtime_manifest_hash = payload.get("runtime_manifest_hash")
    constitution_hash = payload.get("constitution_hash")
    operation = payload.get("operation")

    # --- trust scoring ---
    # A record is trusted only when the source identity names a subsystem,
    # the hash is a 64-hex SHA-256, and the hash matches an existing runtime.
    trusted = False
    if source in ("operator", "evaluator", "steward"):
        if isinstance(runtime_manifest_hash, str) and len(runtime_manifest_hash) == 64:
            try:
                by_hash = None
                for m in registry.list_runtimes():
                    if m.manifest_hash == runtime_manifest_hash:
                        by_hash = m
                        break
                if by_hash is not None:
                    trusted = True
            except Exception:
                trusted = False

    telemetry = ExecutionTelemetry(
        run_id="",
        runtime_id=f"runtime-{payload.get('runtime_version', 'unknown')}",
        runtime_version=payload.get("runtime_version", "unknown"),
        task_id=payload.get("task_id", "unknown"),
        input=payload.get("input", {}),
        output=payload.get("output"),
        success=payload.get("success", False),
        errors=payload.get("errors") or [],
        tools_used=payload.get("tools_used") or [],
        latency_ms=payload.get("latency_ms", 0.0),
        cost=payload.get("cost", 0.0),
        user_feedback=payload.get("user_feedback"),
        source=source,
        source_identity=dict(source_identity),
        trusted=trusted,
        runtime_manifest_hash=runtime_manifest_hash,
        constitution_hash=constitution_hash,
        operation=operation,
    )
    run_id = telemetry_store.record_execution(telemetry)
    return {
        "status": "recorded",
        "run_id": run_id,
        "trusted": telemetry.trusted,
        "note": (
            "recorded as untrusted (audit-only)"
            if not telemetry.trusted
            else "recorded as trusted (eligible for analysis)"
        ),
    }


@app.get("/telemetry/failures")
async def get_failure_telemetry(runtime_version: Optional[str] = None):
    """Get failure telemetry records, classified by failure class."""
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
            "failure_class": r.failure_class,
            "latency_ms": r.latency_ms,
            "cost": r.cost,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


@app.get("/telemetry/failure-classes")
async def get_failure_classes_summary(runtime_version: Optional[str] = None):
    """Get failure class counts for diagnostics."""
    return telemetry_store.get_failure_classes(runtime_version)


# --- Steward loop ---

@app.post("/steward/analyze")
async def steward_analyze(_key: str = Depends(require_governance_key)):
    """Run the steward analysis loop: telemetry → patterns → proposals with auto-derived regression cases."""
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
            regression_cases=p.regression_cases,
        )
        amendments[amendment.id] = amendment
        _persist_amendment(amendment)
        created.append(amendment.id)

        # Register the failure class in the registry
        for rc in p.regression_cases:
            fc = rc.failure_class if hasattr(rc, "failure_class") else None
            if fc:
                failure_class_registry.register_class(fc, description=p.description)

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
        "message": "Governed Evolving AI Runtime v0.2",
        "current_runtime": registry.get_current().version if registry.get_current() else None,
        "constitution_version": constitution.version,
        "constitution_hash": constitution.content_hash[:16] + "..." if constitution.content_hash else "unverified",
        "known_failure_classes": len(failure_class_registry.known_classes()),
        "persistent_storage": "sqlite" if store else "in-memory",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
