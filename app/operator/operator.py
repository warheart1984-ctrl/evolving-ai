from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from app.governance.governor import RuntimeManifest

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
        return f"Task output using runtime {self.current_runtime.version}"
    
    def _validate_output(self, output: Any, input_data: Dict) -> List[str]:
        """Validate the task output."""
        errors = []
        if output is None:
            errors.append("No output produced")
        return errors
    
    def inspect_runtime(self) -> Dict[str, Any]:
        """Inspect the current runtime configuration (read-only)."""
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
        """Confirm the operator cannot modify runtime in place."""
        return {
            "status": "protected",
            "message": "Operator cannot modify runtime in place. "
                       "Changes must go through the amendment pipeline: "
                       "Proposer → Evaluator → Governor → Promotion.",
        }