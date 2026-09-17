"""Governor: enforces the constitution and manages promotions.

Enforces:
- Evidence-linked approval (reviewer must reference specific evaluation evidence IDs)
- Reviewer ≠ proposer (self-approval blocked)
- Concurrent amendment handling (sequential lock)
- Rollback audit entries (who, when, from/to)
- Rejection → adversarial suite growth (past rejections can't silently regress)
"""
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.evaluation._init import FailureClassRegistry
from app.governance.models import (
    Amendment,
    AmendmentStatus,
    ConstitutionGates,
    Decision,
    Evaluation,
    Evidence,
    GateResult,
    PromotionResult,
    RegressionCase,
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
    """Enforces constitutional rules and manages promotions.

    Approval requires:
    1. Evaluation with all gates passing
    2. Status in REVIEW or APPROVED
    3. Explicit reviewer identity (not equal to proposer)
    4. Evidence IDs referencing specific evaluation evidence
    5. No concurrent promotion in progress
    """

    def __init__(self, registry: RuntimeRegistry, constitution,
                 persistence=None,
                 failure_class_registry: FailureClassRegistry = None):
        self.registry = registry
        self.constitution = constitution
        self._persist = persistence
        self.promotion_log: List[Dict[str, Any]] = []
        self.rollback_log: List[Dict[str, Any]] = []
        self.failure_class_registry = failure_class_registry or FailureClassRegistry()
        self._promoting_amendment: Optional[str] = None

        if persistence:
            self._load_audit_from_persistence()

    def _load_audit_from_persistence(self):
        """Load audit entries from persistent store."""
        rows = self._persist.load_all("audit")
        for entry_id, data in rows.items():
            action = data.get("action")
            if action == "promote":
                self.promotion_log.append(data)
            elif action == "rollback":
                self.rollback_log.append(data)

    def _persist_audit_entry(self, entry: Dict[str, Any]):
        """Persist an audit entry."""
        if self._persist:
            key = f"{entry['action']}-{entry.get('amendment_id', entry.get('to_runtime', ''))}-{entry.get('timestamp', '')}"
            self._persist.save("audit", hashlib.sha256(key.encode()).hexdigest()[:12], entry)

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

    def amendment_diff(self, amendment: Amendment) -> Dict[str, Any]:
        """Compute the diff between the current runtime and the proposed amendment.

        This surfaces exactly what will change (prompt diff / memory diff)
        as a first-class review artifact before approval.
        """
        current = self.registry.get_current()
        parent_config = current.model_dump(mode="json") if current else {}
        candidate_config = self._apply_diff(dict(parent_config), amendment.proposed_diff)

        changes: Dict[str, List[Dict[str, Any]]] = {
            "prompts": [],
            "memory": [],
            "tools": [],
            "evaluation": [],
            "other": [],
        }

        all_keys = sorted(set(parent_config.keys()) | set(candidate_config.keys()))
        for key in all_keys:
            parent_val = parent_config.get(key)
            candidate_val = candidate_config.get(key)
            if parent_val == candidate_val:
                continue
            category = key if key in changes else "other"
            changes[category].append({
                "field": key,
                "from": parent_val,
                "to": candidate_val,
            })

        return {
            "parent_runtime": current.id if current else None,
            "parent_version": current.version if current else None,
            "changes": changes,
            "has_changes": any(v for v in changes.values()),
        }

    def validate_diff(self, amendment: Amendment) -> Optional[str]:
        """Reject out-of-scope mutations at the governance boundary."""
        if amendment.target not in (TargetType.PROMPT, TargetType.MEMORY):
            return f"Target '{amendment.target.value}' is outside v0 scope"
        forbidden = {"model_identifier", "tools", "constitution_version", "rollback_to", "id", "version"}
        illegal = sorted(forbidden & set(amendment.proposed_diff))
        if illegal:
            return f"Illegal v0 diff fields: {illegal}"
        return None

    def materialize_candidate(self, amendment: Amendment) -> RuntimeManifest:
        """Apply an amendment to a cloned manifest and register it in SANDBOX."""
        reason = self.validate_diff(amendment)
        if reason:
            raise ValueError(reason)
        parent = self.registry.get_runtime(f"runtime-{amendment.parent_version}")
        if not parent:
            raise ValueError(f"Parent runtime {amendment.parent_version} not found")
        config = self._apply_diff(parent.model_dump(mode="json"), amendment.proposed_diff)
        candidate = RuntimeManifest(
            id=f"runtime-{parent.version}-candidate-{amendment.id}",
            version=f"{parent.version}-candidate-{amendment.id}",
            parent_version=parent.version,
            model_identifier=parent.model_identifier,
            constitution_version=parent.constitution_version,
            prompts=config.get("prompts", {}),
            tools=parent.tools,
            memory=config.get("memory", {}),
            evaluation=config.get("evaluation", {}),
            created_by="governor:sandbox",
            description=f"SANDBOX candidate for {amendment.id}",
            amendment_id=amendment.id,
        )
        amendment.status = AmendmentStatus.SANDBOX
        return self.registry.register_candidate(candidate)

    def approve_amendment(
        self,
        amendment: Amendment,
        evidence_ids: Optional[List[str]] = None,
        reviewer: Optional[str] = None,
    ) -> PromotionResult:
        """Promote an amendment if all gates pass and human approval is recorded.

        Approval requires:
        1. Evaluation with all gates passing
        2. Status in REVIEW or APPROVED
        3. Explicit reviewer identity (not equal to proposer)
        4. Evidence IDs referencing specific evaluation evidence
        5. No concurrent promotion in progress
        """
        result = PromotionResult(success=False)

        illegal_diff = self.validate_diff(amendment)
        if illegal_diff:
            result.reason = illegal_diff
            return result

        # 1. Evaluation required
        if amendment.evaluation is None:
            result.reason = "Amendment must have evaluation results before approval"
            return result

        # 2. Gates must pass
        gate_result = self.evaluate_amendment(amendment)
        if not gate_result["success"]:
            result.reason = f"Governance gates failed: {gate_result['gates_failed']}"
            return result

        # 3. Status must be review or approved
        if amendment.status.value not in ("review", "approved"):
            result.reason = "Human approval required but amendment not in review/approved state"
            return result

        # 4. Reviewer identity required
        effective_reviewer = reviewer or amendment.reviewer
        if not effective_reviewer:
            result.reason = "Human approval required: no reviewer identity provided"
            return result

        # 5. Evidence IDs required and must be valid
        if not evidence_ids:
            result.reason = "Approval must reference evaluation evidence IDs"
            return result

        valid_evidence_ids = {e.id for e in amendment.evaluation.evidence}
        invalid_ids = set(evidence_ids) - valid_evidence_ids
        if invalid_ids:
            result.reason = f"Approval references unknown evidence: {sorted(invalid_ids)}"
            return result

        # 6. Reviewer must not be the proposer
        if effective_reviewer == amendment.proposer:
            result.reason = (
                f"Self-approval blocked: reviewer '{effective_reviewer}' "
                f"cannot approve own proposal"
            )
            return result

        # 7. Concurrent amendment lock
        if self._promoting_amendment and self._promoting_amendment != amendment.id:
            result.reason = (
                f"Concurrent promotion in progress (amendment {self._promoting_amendment}); "
                f"only one promotion may be in progress at a time"
            )
            return result

        # 8. v0 mutation scope: constitutional changes require separate quorum
        if amendment.target not in (TargetType.PROMPT, TargetType.MEMORY):
            result.reason = (
                f"Target '{amendment.target.value}' requires a separate constitutional "
                f"quorum process, not the standard amendment pipeline"
            )
            return result

        # --- Promotion ---
        self._promoting_amendment = amendment.id
        try:
            current = self.registry.get_current()
            if current:
                n = int(current.version.lstrip("v")) + 1
                while self.registry.has_version(f"v{n}"):
                    n += 1
                new_version = f"v{n}"
            else:
                new_version = "v1"
            new_runtime_id = f"runtime-{new_version}"

            parent_config = current.model_dump(mode="json") if current else {}
            new_config = self._apply_diff(dict(parent_config), amendment.proposed_diff)

            self.registry.create_runtime(
                version=new_version,
                model_identifier=new_config.get("model_identifier", "default-model"),
                constitution_version=new_config.get("constitution_version", self.constitution.version),
                prompts=new_config.get("prompts", {}),
                tools=new_config.get("tools", {}),
                memory=new_config.get("memory", {}),
                evaluation=new_config.get("evaluation", {}),
                created_by=effective_reviewer,
                description=f"Promoted amendment {amendment.id}: {amendment.description}",
                amendment_id=amendment.id,
            )

            amendment.status = AmendmentStatus.PROMOTED
            amendment.resulting_runtime = new_runtime_id
            amendment.decided_at = datetime.utcnow()
            amendment.decision = Decision.APPROVE
            amendment.reviewer = effective_reviewer

            result.success = True
            result.new_runtime_id = new_runtime_id
            audit_entry = {
                "action": "promote",
                "amendment_id": amendment.id,
                "from_runtime": current.id if current else None,
                "to_runtime": new_runtime_id,
                "from_version": current.version if current else None,
                "to_version": new_version,
                "timestamp": datetime.utcnow().isoformat(),
                "reviewer": effective_reviewer,
                "evidence_ids": evidence_ids,
                "gates_passed": gate_result["gates_passed"],
            }
            result.audit_log.append(audit_entry)
            self.promotion_log.append(audit_entry)
            self._persist_audit_entry(audit_entry)

        finally:
            self._promoting_amendment = None

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
        """Reject an amendment.

        Every REJECTED amendment's failure reason gets converted into a
        permanent regression case via the failure_class_registry, so the
        eval suite only ever grows and past rejections can't silently
        regress in a later amendment.
        """
        amendment.status = AmendmentStatus.REJECTED
        amendment.decided_at = datetime.utcnow()
        amendment.decision = Decision.REJECT

        # Convert rejection into permanent regression cases
        failure_class = self._derive_failure_class(amendment)
        if failure_class:
            self.failure_class_registry.register_rejection(
                amendment_id=amendment.id,
                failure_class=failure_class,
                reason=reason,
                regression_cases=[
                    rc.model_dump(mode="json") if hasattr(rc, "model_dump") else rc
                    for rc in (amendment.regression_cases or [])
                ],
            )

        audit_entry = {
            "action": "reject",
            "amendment_id": amendment.id,
            "reason": reason,
            "failure_class": failure_class,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return PromotionResult(
            success=False,
            reason=reason,
            audit_log=[audit_entry],
        )

    def _derive_failure_class(self, amendment: Amendment) -> str:
        """Derive a failure class from an amendment's regression cases or description."""
        if amendment.regression_cases:
            # Use the most common failure class from regression cases
            classes = [rc.failure_class for rc in amendment.regression_cases if rc.failure_class]
            if classes:
                return max(set(classes), key=classes.count)
        # Fallback: derive from amendment ID
        slug = amendment.description[:40].lower().replace(" ", "-").replace(":", "")
        return f"amendment:{slug}"

    def rollback(self, target_version: str, initiated_by: str = "governor") -> Optional[RuntimeManifest]:
        """Rollback to a previous runtime version with a full audit entry."""
        current = self.registry.get_current()
        from_id = current.id if current else None
        from_version = current.version if current else None

        runtime = self.registry.rollback_to(target_version)
        if runtime:
            audit_entry = {
                "action": "rollback",
                "to_runtime": runtime.id,
                "to_version": runtime.version,
                "from_runtime": from_id,
                "from_version": from_version,
                "initiated_by": initiated_by,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.rollback_log.append(audit_entry)
            self._persist_audit_entry(audit_entry)
        return runtime

    def audit_trail(self) -> List[Dict[str, Any]]:
        """Full audit trail of promotions, rejections, and rollbacks, chronological."""
        entries = list(self.promotion_log) + list(self.rollback_log)
        return sorted(entries, key=lambda e: e["timestamp"])
