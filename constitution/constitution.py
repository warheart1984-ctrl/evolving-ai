"""Constitution model - the versioned rule set governing the runtime.

Constitution as root of trust:
- Hash-pinned at boot: if constitution.yaml is tampered with outside the
  amendment pipeline, startup refuses (ConstitutionIntegrityError).
- In-memory immutable: Constitution is a frozen Pydantic model so no
  code path can mutate it at runtime.
- Explicitly non-amendable in v0: constitutional changes require a
  separate quorum process, not the standard amendment pipeline.
"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConstitutionIntegrityError(Exception):
    """Raised when constitution.yaml has been tampered with at boot."""
    pass


def _sha256_of_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class Constitution(BaseModel):
    """Immutable constitution. Frozen to prevent in-memory mutation.

    The promotion gates here are the machine-enforced rules.
    """
    model_config = ConfigDict(frozen=True)

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

    # v0: constitutional changes require separate quorum, not the standard pipeline
    constitution_amendment_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "required_reviewers": 2,
        "cooling_off_period_minutes": 1440,  # 24 hours
        "quorum_required": True,
        "note": "Constitutional changes are NOT amendable in v0. "
                "This policy is preserved for future implementation.",
    })

    content_hash: str = ""  # SHA-256 of constitution.yaml at load time

    @classmethod
    def from_file(cls, path, pin_path: Optional[str] = None) -> "Constitution":
        """Load constitution from YAML, verifying hash pin if provided.

        Args:
            path: Path to constitution.yaml
            pin_path: Optional path to .sha256 pin file. If provided and
                the hash doesn't match, raises ConstitutionIntegrityError.
        """
        path = str(path)
        actual_hash = _sha256_of_file(path)

        # Verify pin if provided
        if pin_path:
            pin_path = str(pin_path)
            if not os.path.exists(pin_path):
                raise ConstitutionIntegrityError(
                    f"Constitution pin file not found: {pin_path}. "
                    f"Cannot verify integrity of {path}"
                )
            with open(pin_path, "r") as f:
                expected_hash = f.read().strip()
            if actual_hash != expected_hash:
                raise ConstitutionIntegrityError(
                    f"Constitution integrity violation! "
                    f"Expected hash {expected_hash[:16]}... but got {actual_hash[:16]}... "
                    f"File {path} has been modified outside the amendment pipeline."
                )

        # Load YAML
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            data = {}

        data["content_hash"] = actual_hash
        return cls(**data)


import os  # noqa: E402 (needed at module level for from_file)
