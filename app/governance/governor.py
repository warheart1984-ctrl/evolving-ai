"""Governor: enforces the constitution and manages promotions."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.governance.models import (
    Amendment,
    AmendmentStatus,
    ConstitutionGates,
    Decision,
    Evaluation,
    Evidence,
    GateResult,
    PromotionResult,
    RuntimeManifest,
    TargetType,
)
from app.governance.registry import RuntimeRegistry

__all__ = [
    "RuntimeRegistry",
    "Governor",
    "Amendment",
    "AmendmentStatus",
    "TargetType",
    "ConstitutionGates",
    "Evaluation",
    "Evidence",
    "GateResult",
    "PromotionResult",
    "RuntimeManifest",
    "Decision",
]


class Governor:
    """Enforces constitutional rules and manages promotions."""

    def __init__(self, registry: RuntimeRegistry, constitution):
        self.registry = registry
        self.constitution = constitution
        self.promotion_log: List[Dict[str, Any]] = []
        self.rollback_log: List[Dict[str, Any]] = []

    def evaluate_amendment(self, amendment: Amendment) -> Dict[str, Any]:
        """Run governance gates on an amendment's evaluation."""
        if amendment.evaluation is None:
            return {
                "success": False,
                "reason": "No evaluation results attached to amendment",
                "gates_passed": [],
                "gates_failed": ["evaluation_evidence_exists"],
            }

        gates = ConstitutionGates.check_all_gates(amendment.evaluation)
        gates_passed = [name for name, gate in gates.items() if gate.passed]
        gates_failed = [name for name, gate in gates.items() if not gate.passed]

        return {
            "success": len(gates_failed) == 0,
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
        }

    def approve_amendment(self, amendment: Amendment) -> PromotionResult:
        """Promote an amendment if all gates pass and human approval is recorded."""
        result = PromotionResult(success=False)

        if amendment.evaluation is None:
            result.reason = "Amendment must have evaluation results before approval"
            return result

        if not amendment.evaluation.evidence:
            result.reason = "Approval requires at least one linked evaluation evidence item"
            return result
        if amendment.reviewer and amendment.reviewer == amendment.proposer:
            result.reason = "Reviewer cannot be the amendment proposer"
            return result

        gate_result = self.evaluate_amendment(amendment)
        if not gate_result["success"]:
            result.reason = f"Governance gates failed: {gate_result['gates_failed']}"
            return result

        if amendment.status.value not in ("review", "approved"):
            result.reason = "Human approval required but amendment not in review/approved state"
            return result

        current = self.registry.get_current()
        if current:
            n = int(current.version.lstrip("v")) + 1
            while self.registry.has_version(f"v{n}"):
                n += 1
            new_version = f"v{n}"
        else:
            new_version = "v1"
        new_runtime_id = f"runtime-{new_version}"

        parent_config = current.model_dump() if current else {}
        new_config = self._apply_diff(parent_config, amendment.proposed_diff)

        self.registry.create_runtime(
            version=new_version,
            model_identifier=new_config.get("model_identifier", "default-model"),
            constitution_version=new_config.get("constitution_version", self.constitution.version),
            prompts=new_config.get("prompts", {}),
            tools=new_config.get("tools", {}),
            memory=new_config.get("memory", {}),
            evaluation=new_config.get("evaluation", {}),
            created_by=amendment.reviewer or "governor",
            description=f"Promoted amendment {amendment.id}: {amendment.description}",
            amendment_id=amendment.id,
        )

        amendment.status = AmendmentStatus.PROMOTED
        amendment.resulting_runtime = new_runtime_id
        amendment.decided_at = datetime.utcnow()
        amendment.decision = Decision.APPROVE

        result.success = True
        result.new_runtime_id = new_runtime_id
        audit_entry = {
            "action": "promote",
            "amendment_id": amendment.id,
            "from_runtime": current.id if current else None,
            "to_runtime": new_runtime_id,
            "timestamp": datetime.utcnow().isoformat(),
            "reviewer": amendment.reviewer,
            "gates_passed": gate_result["gates_passed"],
        }
        result.audit_log.append(audit_entry)
        self.promotion_log.append(audit_entry)

        return result

    def _apply_diff(self, parent: Dict, diff: Dict) -> Dict:
        """Apply a diff to a parent configuration, producing a new config."""
        result = dict(parent)
        for key, value in diff.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            elif key.startswith("prompt_") and key not in result:
                result.setdefault("prompts", {})[key[len("prompt_"):]] = value
            elif key.startswith("memory_") and key not in result:
                result.setdefault("memory", {})[key[len("memory_"):]] = value
            else:
                result[key] = value
        return result

    def reject_amendment(self, amendment: Amendment, reason: str) -> PromotionResult:
        """Reject an amendment."""
        amendment.status = AmendmentStatus.REJECTED
        amendment.decided_at = datetime.utcnow()
        amendment.decision = Decision.REJECT

        return PromotionResult(
            success=False,
            reason=reason,
            audit_log=[{
                "action": "reject",
                "amendment_id": amendment.id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }],
        )

    def rollback(self, target_version: str) -> Optional[RuntimeManifest]:
        """Rollback to a previous runtime version (auditable)."""
        previous = self.registry.get_current()
        runtime = self.registry.rollback_to(target_version)
        if runtime:
            self.rollback_log.append({
                "action": "rollback",
                "from_runtime": previous.id if previous else None,
                "to_runtime": runtime.id,
                "reviewer": "governor",
                "timestamp": datetime.utcnow().isoformat(),
            })
        return runtime

    def audit_trail(self) -> List[Dict[str, Any]]:
        """Full audit trail of promotions and rollbacks, chronological."""
        entries = list(self.promotion_log) + list(self.rollback_log)
        return sorted(entries, key=lambda e: e["timestamp"])
