"""Telemetry: records operator execution telemetry with failure classification.

Telemetry records have an explicit trust model:
- source / source_identity / trusted identify provenance
- Only trusted telemetry is used for evaluations and steward amendments
- Untrusted telemetry is stored under a separate kind for audit only
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Rate limiting (simple per-IP token bucket)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """In-memory per-key token bucket. Allows `rate` tokens per `window` seconds."""

    def __init__(self, rate: int = 30, window: float = 60.0):
        self._rate = rate
        self._window = window
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets[key]
            # Prune expired timestamps
            self._buckets[key] = [t for t in bucket if now - t < self._window]
            if len(self._buckets[key]) >= self._rate:
                return False
            self._buckets[key].append(now)
            return True

# Module-level limiter for external telemetry submissions
_external_telemetry_limiter = _RateLimiter(rate=30, window=60.0)


# Maximum payload size for external telemetry submissions (bytes)
MAX_TELEMETRY_PAYLOAD_BYTES = 64 * 1024  # 64 KB
MAX_TELEMETRY_FIELDS = 50


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


# ---------------------------------------------------------------------------
# Canonical hashing helpers
# ---------------------------------------------------------------------------

def _canonical_json(data: Any) -> str:
    """Deterministic JSON serialization (sorted keys, no whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_hash(manifest_dict: Dict[str, Any]) -> str:
    """SHA-256 hex of a RuntimeManifest's canonical JSON (excluding id and manifest_hash)."""
    filtered = {k: v for k, v in manifest_dict.items() if k not in ("id", "manifest_hash")}
    return hashlib.sha256(_canonical_json(filtered).encode()).hexdigest()


def constitution_hash(yaml_content: str) -> str:
    """SHA-256 hex of the raw constitution YAML content."""
    return hashlib.sha256(yaml_content.encode()).hexdigest()


class ExecutionTelemetry(BaseModel):
    """Telemetry record for a single operator execution.

    Trust fields:
    - source: which subsystem produced this record ("operator", "evaluator", "steward", "external")
    - source_identity: owner + hash-signed content provided by the subsystem
    - trusted: True only when source + source_identity are populated by an internal subsystem
    - runtime_manifest_hash: hex SHA-256 of the runtime manifest (canonical JSON, excluding id/hash)
    - constitution_hash: hex SHA-256 of the constitution YAML (pin)
    - operation: structured description of the operation performed
    """
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

    # --- Trust / provenance fields (P3) ---
    source: Literal["operator", "evaluator", "steward", "external"] = "operator"
    source_identity: Optional[Dict[str, Any]] = None
    trusted: bool = False

    runtime_manifest_hash: Optional[str] = None
    constitution_hash: Optional[str] = None

    operation: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Telemetry store
# ---------------------------------------------------------------------------

class TelemetryStore:
    """Records and queries operator execution telemetry with persistence.

    Trust model:
    - trusted=True records go into "telemetry" (used by steward/evaluator)
    - trusted=False records go into "telemetry_untrusted" (audit-only, never aggregated)
    """

    def __init__(self, persistence=None):
        self._records: Dict[str, ExecutionTelemetry] = {}
        self._next_id = 1
        self._persist = persistence
        if persistence:
            self._load_from_persistence()

    def _load_from_persistence(self):
        """Load existing telemetry records from persistent store (trusted only).

        Corrupt records raise (via StateStore.integrity checks) so trusted
        telemetry can never be silently dropped at startup.
        """
        rows = self._persist.load_all("telemetry")
        for run_id, data in rows.items():
            self._records[run_id] = ExecutionTelemetry(**data)
            num = int(run_id.split("-")[-1]) + 1
            if num > self._next_id:
                self._next_id = num

    def record_execution(self, telemetry: ExecutionTelemetry) -> str:
        """Record an execution and return the run ID.

        Untrusted records are persisted to a separate kind and excluded
        from in-memory aggregation (they never appear in failure-class
        counts or steward recommendations).
        """
        run_id = f"run-{self._next_id:06d}"
        self._next_id += 1
        telemetry.run_id = run_id

        # Auto-classify failures
        if not telemetry.success and not telemetry.failure_class:
            telemetry.failure_class = classify_failure(
                telemetry.task_id, telemetry.input, telemetry.errors, telemetry.output
            )

        if telemetry.trusted:
            self._records[run_id] = telemetry
            if self._persist:
                self._persist.save("telemetry", run_id, telemetry.model_dump(mode="json"))
        else:
            # Untrusted: audit-only, stored under separate kind
            if self._persist:
                self._persist.save("telemetry_untrusted", run_id, telemetry.model_dump(mode="json"))
        return run_id

    def get_record(self, run_id: str) -> Optional[ExecutionTelemetry]:
        """Get a trusted telemetry record by ID."""
        return self._records.get(run_id)

    def get_records_by_runtime(self, runtime_version: str) -> List[ExecutionTelemetry]:
        """Get all trusted records for a specific runtime version."""
        return [r for r in self._records.values() if r.runtime_version == runtime_version]

    def get_records_by_task(self, task_id: str) -> List[ExecutionTelemetry]:
        """Get all trusted records for a specific task."""
        return [r for r in self._records.values() if r.task_id == task_id]

    def get_failure_records(self, runtime_version: str = None) -> List[ExecutionTelemetry]:
        """Get failure records, optionally filtered by runtime. Trusted only."""
        if runtime_version:
            return [r for r in self._records.values() if not r.success and r.runtime_version == runtime_version]
        return [r for r in self._records.values() if not r.success]

    def get_success_records(self, runtime_version: str = None) -> List[ExecutionTelemetry]:
        """Get success records, optionally filtered by runtime. Trusted only."""
        if runtime_version:
            return [r for r in self._records.values() if r.success and r.runtime_version == runtime_version]
        return [r for r in self._records.values() if r.success]

    def get_failure_classes(self, runtime_version: str = None) -> Dict[str, int]:
        """Get counts of each failure class, optionally filtered by runtime. Trusted only."""
        counts: Dict[str, int] = {}
        for r in self.get_failure_records(runtime_version):
            fc = r.failure_class or "unclassified"
            counts[fc] = counts.get(fc, 0) + 1
        return counts

    def get_untrusted_records(self) -> List[ExecutionTelemetry]:
        """Load all untrusted records from persistence (audit view)."""
        if not self._persist:
            return []
        rows = self._persist.load_all("telemetry_untrusted")
        return [ExecutionTelemetry(**data) for data in rows.values()]
