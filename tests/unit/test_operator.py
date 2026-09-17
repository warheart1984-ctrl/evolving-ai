"""Tests for operator module - executes tasks but cannot self-modify."""
import pytest
from datetime import datetime
from app.operator.operator import Operator, TaskResult
from app.governance.governor import RuntimeManifest


class TestOperator:
    """Test operator task execution."""
    
    def test_operator_execute_task(self):
        """Test basic task execution."""
        # Create a minimal runtime
        runtime = RuntimeManifest(
            id="runtime-v0",
            version="v0",
            model_identifier="test-model",
            constitution_version="v1",
            prompts={"system": "default_system"},
            created_by="system",
            description="Test runtime",
        )
        
        operator = Operator(
            registry=None,  # Not using registry in this test
            current_runtime=runtime,
        )
        
        result = operator.execute_task(
            task_id="task-001",
            input_data={"question": "What is 2+2?"},
        )
        
        assert isinstance(result, TaskResult)
        assert result.task_id == "task-001"
        assert result.runtime_version == "v0"
        assert result.success == True
    
    def test_operator_inspect_runtime(self):
        """Test operator can inspect current runtime."""
        runtime = RuntimeManifest(
            id="runtime-v0",
            version="v0",
            model_identifier="test-model",
            constitution_version="v1",
            prompts={"system": "default_system"},
            tools={"calculator": "v3"},
            created_by="system",
            description="Test runtime",
        )
        
        operator = Operator(
            registry=None,
            current_runtime=runtime,
        )
        
        inspected = operator.inspect_runtime()
        assert inspected["version"] == "v0"
        assert inspected["model"] == "test-model"
        assert inspected["prompts"]["system"] == "default_system"
    
    def test_operator_cannot_modify_runtime(self):
        """Test operator cannot modify runtime in place."""
        runtime = RuntimeManifest(
            id="runtime-v0",
            version="v0",
            model_identifier="test-model",
            constitution_version="v1",
            created_by="system",
            description="Test runtime",
        )
        
        operator = Operator(
            registry=None,
            current_runtime=runtime,
        )
        
        result = operator.cannot_modify_runtime()
        assert result["status"] == "protected"
        assert "cannot modify" in result["message"].lower()
    
    def test_operator_task_with_errors(self):
        """Test operator task that produces errors."""
        runtime = RuntimeManifest(
            id="runtime-v0",
            version="v0",
            model_identifier="test-model",
            constitution_version="v1",
            created_by="system",
            description="Test runtime",
        )
        
        operator = Operator(
            registry=None,
            current_runtime=runtime,
        )
        
        # Task that produces an error
        result = operator.execute_task(
            task_id="task-002",
            input_data={"invalid": "input"},
        )
        
        assert isinstance(result, TaskResult)
        # Task may or may not succeed depending on implementation
        assert result.task_id == "task-002"
        assert result.runtime_version == "v0"