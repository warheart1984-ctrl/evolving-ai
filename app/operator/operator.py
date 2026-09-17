from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from app.evaluation.arithmetic import safe_arithmetic
from app.governance.governor import RuntimeManifest
from app.governance.registry import RuntimeRegistry
from app.telemetry._init import ExecutionTelemetry

class TaskResult(BaseModel):
    """Result of executing a task."""
    task_id: str
    input: Dict[str, Any]
    output: Any
    success: bool
    errors: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost: float = 0.0
    runtime_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Operator:
    """Executes normal user tasks. Cannot modify its own runtime."""
    
    def __init__(
        self,
        registry: 'RuntimeRegistry',
        current_runtime: RuntimeManifest,
        tools: Dict[str, Any] = None,
        memory: Dict[str, Any] = None,
    ):
        self.registry = registry
        self.current_runtime = current_runtime
        self.tools = tools or {}
        self.memory = memory or {}
        # Constitution hash pin, if known, so operator telemetry carries the
        # exact constitution the runtime executed under.
        self.constitution_hash = None

    def set_constitution_hash(self, constitution_hash: str):
        """Bind the constitution hash so telemetry records are hash-signed."""
        self.constitution_hash = constitution_hash

    def _compute_manifest_hash(self) -> str:
        """Compute the runtime manifest hash using the registry's canonical algorithm."""
        return RuntimeRegistry._with_hash(self.current_runtime).manifest_hash

    def build_telemetry(self, result: TaskResult, input_data: Dict[str, Any] = None) -> ExecutionTelemetry:
        """Build a trusted telemetry record from a task result.

        Source identity is a FrozenDict binding the operator role, the exact
        runtime manifest hash, and the constitution hash (if pinned), so the
        record is attributable and tamper-evident.
        """
        from app.governance.models import FrozenDict

        rm_hash = self._compute_manifest_hash()
        source_identity = FrozenDict({
            "owner": "operator",
            "runtime_manifest_hash": rm_hash,
            "constitution_hash": self.constitution_hash or "",
        })
        return ExecutionTelemetry(
            run_id="",
            runtime_id=self.current_runtime.id,
            runtime_version=self.current_runtime.version,
            task_id=result.task_id,
            input=input_data if input_data is not None else result.input,
            output=result.output,
            success=result.success,
            errors=result.errors,
            tools_used=result.tools_used,
            latency_ms=result.latency_ms,
            cost=result.cost,
            source="operator",
            source_identity=dict(source_identity),
            trusted=True,
            runtime_manifest_hash=rm_hash,
            constitution_hash=self.constitution_hash,
            operation={"verb": "execute_task", "task_id": result.task_id},
        )
    
    def execute_task(
        self,
        task_id: str,
        input_data: Dict[str, Any],
        available_tools: List[str] = None,
    ) -> TaskResult:
        """Execute a task using the current approved runtime configuration."""
        # Record that we're using the current runtime
        runtime_version = self.current_runtime.version
        
        # Get tools available for this task
        task_tools = available_tools or self._get_applicable_tools(input_data)
        
        # Execute the task (simplified - in production would call actual model/tools)
        output = self._run_with_configuration(input_data, task_tools)
        
        # Check for errors
        errors = self._validate_output(output, input_data)
        
        result = TaskResult(
            task_id=task_id,
            input=input_data,
            output=output,
            success=len(errors) == 0,
            errors=errors,
            tools_used=task_tools,
            latency_ms=0.0,  # Would be measured in production
            cost=0.0,  # Would be measured in production
            runtime_version=runtime_version,
        )
        
        # Telemetry would be recorded here
        # record_telemetry(execution_record)
        
        return result
    
    def _get_applicable_tools(self, input_data: Dict[str, Any]) -> List[str]:
        """Get tools applicable to the task based on current runtime config."""
        configured = dict(self.current_runtime.tools or {})
        if not self.tools:
            return []
        return [name for name in self.tools if not configured or name in configured]
    
    def _run_with_configuration(self, input_data: Dict, tools: List[str]) -> Any:
        """Run the task with the current runtime's configuration."""
        # In a real implementation, this would:
        # 1. Load the appropriate model based on runtime config
        # 2. Load the appropriate prompts based on runtime config
        # 3. Execute with the configured tools
        # 4. Return the output
        expression = input_data.get("expression")
        if expression is not None:
            value = self._safe_arithmetic(str(expression))
            if "step-by-step" in self.current_runtime.prompts.get("system", "").lower():
                return f"{expression} = {value}"
            return str(value)
        return f"Processed with {self.current_runtime.prompts.get('system', 'default prompt')}: {input_data}"

    @staticmethod
    def _safe_arithmetic(expression: str):
        try:
            return safe_arithmetic(expression)
        except ArithmeticError as e:
            raise ValueError(str(e))
    
    def _validate_output(self, output: Any, input_data: Dict) -> List[str]:
        """Validate the task output."""
        errors = []
        if output is None:
            errors.append("No output produced")
        return errors

    def inspect_runtime(self) -> Dict[str, Any]:
        return {
            "version": self.current_runtime.version,
            "model": self.current_runtime.model_identifier,
            "constitution": self.current_runtime.constitution_version,
            "prompts": dict(self.current_runtime.prompts),
            "tools": dict(self.current_runtime.tools),
            "memory": dict(self.current_runtime.memory),
            "evaluation": dict(self.current_runtime.evaluation),
        }

    def cannot_modify_runtime(self) -> Dict[str, str]:
        return {"status": "protected", "message": "Operator cannot modify runtime in place."}
