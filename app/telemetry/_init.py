"""Telemetry: records operator execution telemetry with failure classification."""
import hashlib
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def classify_failure(task_id: str, input_data: Dict[str, Any], errors: List[str], output: Any) -> str:
    """Classify a failure into a stable failure class tag.

    Classes are deterministic and based on task type + error signature.
    Examples: "math:incorrect-answer", "summarize:empty-output", "api:timeout"
    """
    task_kind = "unknown"
    if isinstance(input_data, dict):
        if "expression" in input_data:
            task_kind = "math"
        elif "text" in input_data:
            task_kind = "summarize"
        elif "url" in input_data:
            task_kind = "api"
        elif "query" in input_data:
            task_kind = "query"
    if task_kind == "unknown" and task_id:
        task_kind = task_id.split("_")[0].split("-")[0]

    err_msg = (errors[0] if errors else "empty-output").lower()
    err_tag = "incorrect-answer"
    if "timeout" in err_msg:
        err_tag = "timeout"
    elif "empty" in err_msg or "no output" in err_msg:
        err_tag = "empty-output"
    elif "permission" in err_msg or "denied" in err_msg:
        err_tag = "permission-denied"
    elif "not found" in err_msg or "missing" in err_msg:
        err_tag = "not-found"
    elif "overflow" in err_msg:
        err_tag = "overflow"
    elif "incorrect" in err_msg or "wrong" in err_msg:
        err_tag = "incorrect-answer"

    return f"{task_kind}:{err_tag}"


class ExecutionTelemetry(BaseModel):
    """Telemetry record for a single operator execution."""
    run_id: str
    runtime_id: str
    runtime_version: str

    task_id: str
    input: Dict[str, Any]
    output: Any

    success: bool
    errors: List[str] = Field(default_factory=list)

    # Classification of the failure (computed at record time for failures)
    failure_class: Optional[str] = None

    tools_used: List[str] = Field(default_factory=list)

    latency_ms: float = 0.0
    cost: float = 0.0

    evaluation_results: Optional[Dict[str, Any]] = None

    user_feedback: Optional[Dict[str, Any]] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TelemetryStore:
    """Records and queries operator execution telemetry with persistence."""

    def __init__(self, persistence=None):
        self._records: Dict[str, ExecutionTelemetry] = {}
        self._next_id = 1
        self._persist = persistence
        if persistence:
            self._load_from_persistence()

    def _load_from_persistence(self):
        """Load existing telemetry records from persistent store."""
        rows = self._persist.load_all("telemetry")
        for run_id, data in rows.items():
            try:
                self._records[run_id] = ExecutionTelemetry(**data)
                num = int(run_id.split("-")[-1]) + 1
                if num > self._next_id:
                    self._next_id = num
            except Exception:
                pass

    def record_execution(self, telemetry: ExecutionTelemetry) -> str:
        """Record an execution and return the run ID."""
        run_id = f"run-{self._next_id:06d}"
        self._next_id += 1
        telemetry.run_id = run_id

        # Auto-classify failures
        if not telemetry.success and not telemetry.failure_class:
            telemetry.failure_class = classify_failure(
                telemetry.task_id, telemetry.input, telemetry.errors, telemetry.output
            )

        self._records[run_id] = telemetry
        if self._persist:
            self._persist.save("telemetry", run_id, telemetry.model_dump(mode="json"))
        return run_id

    def get_record(self, run_id: str) -> Optional[ExecutionTelemetry]:
        """Get a telemetry record by ID."""
        return self._records.get(run_id)

    def get_records_by_runtime(self, runtime_version: str) -> List[ExecutionTelemetry]:
        """Get all records for a specific runtime version."""
        return [r for r in self._records.values() if r.runtime_version == runtime_version]

    def get_records_by_task(self, task_id: str) -> List[ExecutionTelemetry]:
        """Get all records for a specific task."""
        return [r for r in self._records.values() if r.task_id == task_id]

    def get_failure_records(self, runtime_version: str = None) -> List[ExecutionTelemetry]:
        """Get failure records, optionally filtered by runtime."""
        if runtime_version:
            return [r for r in self._records.values() if not r.success and r.runtime_version == runtime_version]
        return [r for r in self._records.values() if not r.success]

    def get_success_records(self, runtime_version: str = None) -> List[ExecutionTelemetry]:
        """Get success records, optionally filtered by runtime."""
        if runtime_version:
            return [r for r in self._records.values() if r.success and r.runtime_version == runtime_version]
        return [r for r in self._records.values() if r.success]

    def get_failure_classes(self, runtime_version: str = None) -> Dict[str, int]:
        """Get counts of each failure class, optionally filtered by runtime."""
        counts: Dict[str, int] = {}
        for r in self.get_failure_records(runtime_version):
            fc = r.failure_class or "unclassified"
            counts[fc] = counts.get(fc, 0) + 1
        return counts
