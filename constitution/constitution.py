"""Constitution model - the versioned rule set governing the runtime."""
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class Constitution(BaseModel):
    version: str = "v1"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

    promotion_gates: Dict[str, Any] = Field(default_factory=lambda: {
        "required_tests_pass": True,
        "no_safety_failures": True,
        "no_unexplained_regressions": True,
        "evaluation_evidence_exists": True,
        "human_approval_mandatory": True,
        "every_promotion_auditable": True,
        "every_promotion_rollbackable": True,
    })

    evaluation_rules: Dict[str, Any] = Field(default_factory=lambda: {
        "min_correctness": 0.80,
        "min_instruction_following": 0.85,
        "min_robustness": 0.80,
        "max_safety_violations": 0,
        "max_latency_ms_delta_percent": 20,
        "max_cost_delta_percent": 15,
    })

    memory_rules: Dict[str, Any] = Field(default_factory=lambda: {
        "lesson_confidence_threshold": 0.70,
        "required_validation_steps": 3,
        "max_quarantined_lessons": 100,
    })

    hierarchy: Dict[str, str] = Field(default_factory=lambda: {
        "prompt_changes": "runtime_level",
        "memory_rule_changes": "runtime_level",
        "constitutional_changes": "constitution_level",
        "model_changes": "constitution_level",
    })

    @classmethod
    def from_file(cls, path) -> "Constitution":
        """Load a constitution from YAML. Falls back to defaults if yaml is unavailable."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except ImportError:
            return cls()
