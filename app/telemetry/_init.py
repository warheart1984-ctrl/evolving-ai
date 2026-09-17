from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field


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
    
    tools_used: List[str] = Field(default_factory=list)
    
    latency_ms: float = 0.0
    cost: float = 0.0
    
    evaluation_results: Optional[Dict[str, Any]] = None
    
    user_feedback: Optional[Dict[str, Any]] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TelemetryStore:
    """Records and queries operator execution telemetry."""
    
    def __init__(self):
        self._records: Dict[str, ExecutionTelemetry] = {}
        self._next_id = 1
    
    def record_execution(self, telemetry: ExecutionTelemetry) -> str:
        """Record an execution and return the run ID."""
        run_id = f"run-{self._next_id:06d}"
        self._next_id += 1
        telemetry.run_id = run_id
        self._records[run_id] = telemetry
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
            return self.get_records_by_runtime(runtime_version)
        return [r for r in self._records.values() if not r.success]
    
    def get_success_records(self, runtime_version: str = None) -> List[ExecutionTelemetry]:
        """Get success records, optionally filtered by runtime."""
        if runtime_version:
            return self.get_records_by_runtime(runtime_version)
        return [r for r in self._records.values() if r.success]
