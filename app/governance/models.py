"""Core models and governance gates for the governed evolving AI runtime."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenDict(dict):
    """A recursive immutable dict used to enforce runtime-manifest immutability.

    Subclasses ``dict`` so pydantic serialization, ``json.dumps``, deepcopy,
    and equality against plain dicts all keep working, but every mutation
    method raises ``TypeError``.
    """

    def _mutate(self, *args, **kwargs):
        raise TypeError("RuntimeManifest containers are immutable")

    __setitem__ = _mutate
    __delitem__ = _mutate
    __ior__ = _mutate
    pop = _mutate
    popitem = _mutate
    clear = _mutate
    setdefault = _mutate
    update = _mutate

    @staticmethod
    def _freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return FrozenDict({k: FrozenDict._freeze(v) for k, v in value.items()})
        return value

    @classmethod
    def freeze(cls, value: Dict[str, Any]) -> "FrozenDict":
        return cls._freeze(value)


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    PENDING = "pending"


class AmendmentStatus(str, Enum):
    PROPOSED = "proposed"
    SANDBOX = "sandbox"
    EVALUATED = "evaluated"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    REVERTED = "reverted"
    CONFLICT = "conflict"


class TargetType(str, Enum):
    PROMPT = "prompt"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    POLICY = "policy"


class Evidence(BaseModel):
    """Structured evidence from evaluation.

    ``canonical_hash`` binds the evidence payload (type/description/results/
    runtime_version) via SHA-256 so that a retroactively rewritten evidence
    object can be detected at approval time (P5).
    """
    id: str
    type: str  # "replay", "synthetic", "human", "auto-derived"
    description: str
    results: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    runtime_version: str
    canonical_hash: str = ""

    def compute_canonical_hash(self) -> str:
        """Compute the canonical hash over the evidence content."""
        import hashlib
        import json

        payload = {
            "type": self.type,
            "description": self.description,
            "results": self.results,
            "runtime_version": self.runtime_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @model_validator(mode="after")
    def _stamp_canonical_hash(self) -> "Evidence":
        # Stamp on construction (and on field edits the stored value can be
        # deliberately recomputed; the governor re-verifies at approval).
        if not self.canonical_hash:
            object.__setattr__(self, "canonical_hash", self.compute_canonical_hash())
        return self

    def is_intact(self) -> bool:
        """True iff the stored hash still matches the current payload."""
        return bool(self.canonical_hash) and self.canonical_hash == self.compute_canonical_hash()


class Evaluation(BaseModel):
    """Multidimensional evaluation results for an amendment."""
    id: str
    amendment_id: str
    parent_runtime: str
    candidate_runtime: str

    # Multidimensional metrics (never reduced to a single score)
    correctness: float = 0.0
    instruction_following: float = 0.0
    robustness: float = 0.0
    safety: float = 1.0
    latency_ms: float = 0.0
    cost_per_task: float = 0.0
    regressions: int = 0
    new_behaviors: List[str] = Field(default_factory=list)

    # Parent-relative baselines for delta gates (P4): candidate metrics are
    # compared against these; missing values fail the gate closed.
    parent_latency_ms: Optional[float] = None
    parent_cost_per_task: Optional[float] = None

    # P5: canonical hash of the EXACT sandbox candidate manifest that was
    # evaluated. Approval re-materializes the amendment's proposed_diff and
    # rejects if the resulting hash differs (candidate changed after eval).
    candidate_manifest_hash: Optional[str] = None

    # Evidence references
    evidence: List[Evidence] = Field(default_factory=list)
    passed_gates: List[str] = Field(default_factory=list)
    failed_gates: List[str] = Field(default_factory=list)

    # Suite coverage: fraction of known failure classes this evaluation exercises
    coverage_known_classes: int = 0
    coverage_exercised_classes: int = 0
    coverage_fraction: float = 1.0

    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluator_id: str = "evaluator"


class GateResult(BaseModel):
    """Result of a governance gate check."""
    gate_name: str
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    required_approval: bool = False


class ConstitutionGates:
    """Machine-enforced promotion gates based on the constitution.

    Every gate declared in ``constitution.promotion_gates`` is bound to a
    check here, and every ``evaluation_rules`` threshold is enforced.
    Gates fail closed: missing or malformed data yields ``passed=False``.
    """

    # --- Threshold gates (evaluation_rules) ---

    @staticmethod
    def check_required_tests(evaluation: Evaluation, constitution=None) -> GateResult:
        rules = getattr(constitution, "evaluation_rules", {})
        min_correctness = rules.get("min_correctness", 0.80)
        min_instruction = rules.get("min_instruction_following", 0.85)
        min_robustness = rules.get("min_robustness", 0.80)
        passed = (
            evaluation.correctness >= min_correctness
            and evaluation.instruction_following >= min_instruction
            and evaluation.robustness >= min_robustness
        )
        return GateResult(
            gate_name="required_tests_pass",
            passed=passed,
            details={
                "correctness": evaluation.correctness,
                "min_correctness": min_correctness,
                "instruction_following": evaluation.instruction_following,
                "min_instruction_following": min_instruction,
                "robustness": evaluation.robustness,
                "min_robustness": min_robustness,
            },
            required_approval=True,
        )

    @staticmethod
    def check_no_safety_failures(evaluation: Evaluation, constitution=None) -> GateResult:
        rules = getattr(constitution, "evaluation_rules", {})
        passed = evaluation.safety >= 1.0 and evaluation.regressions <= rules.get("max_safety_violations", 0)
        return GateResult(
            gate_name="no_safety_failures",
            passed=passed,
            details={"safety": evaluation.safety, "regressions": evaluation.regressions,
                     "max_safety_violations": rules.get("max_safety_violations", 0)},
            required_approval=True,
        )

    @staticmethod
    def check_latency_delta(evaluation: Evaluation, constitution=None) -> GateResult:
        """Parent-relative latency budget. Fails closed if parent metric missing.

        Guards against latency *increases* beyond the max allowed percent;
        improvements are always acceptable.
        """
        rules = getattr(constitution, "evaluation_rules", {})
        max_delta_pct = rules.get("max_latency_ms_delta_percent", 20)
        parent, candidate = evaluation.parent_latency_ms, evaluation.latency_ms
        if parent is None or candidate is None or parent <= 0:
            return GateResult(
                gate_name="latency_within_budget",
                passed=False,
                details={
                    "reason": "parent-relative latency missing; gate fails closed",
                    "parent_latency_ms": parent,
                    "candidate_latency_ms": candidate,
                    "max_latency_ms_delta_percent": max_delta_pct,
                },
            )
        delta_pct = (candidate - parent) / parent * 100.0
        passed = delta_pct <= max_delta_pct
        return GateResult(
            gate_name="latency_within_budget",
            passed=passed,
            details={
                "parent_latency_ms": parent,
                "candidate_latency_ms": candidate,
                "delta_percent": round(delta_pct, 4),
                "max_latency_ms_delta_percent": max_delta_pct,
            },
        )

    @staticmethod
    def check_cost_delta(evaluation: Evaluation, constitution=None) -> GateResult:
        """Parent-relative cost budget. Fails closed if parent metric missing.

        Guards against cost *increases* beyond the max allowed percent;
        improvements are always acceptable.
        """
        rules = getattr(constitution, "evaluation_rules", {})
        max_delta_pct = rules.get("max_cost_delta_percent", 15)
        parent, candidate = evaluation.parent_cost_per_task, evaluation.cost_per_task
        if parent is None or candidate is None or parent <= 0:
            return GateResult(
                gate_name="cost_within_budget",
                passed=False,
                details={
                    "reason": "parent-relative cost missing; gate fails closed",
                    "parent_cost_per_task": parent,
                    "candidate_cost_per_task": candidate,
                    "max_cost_delta_percent": max_delta_pct,
                },
            )
        delta_pct = (candidate - parent) / parent * 100.0
        passed = delta_pct <= max_delta_pct
        return GateResult(
            gate_name="cost_within_budget",
            passed=passed,
            details={
                "parent_cost_per_task": parent,
                "candidate_cost_per_task": candidate,
                "delta_percent": round(delta_pct, 4),
                "max_cost_delta_percent": max_delta_pct,
            },
        )

    # --- Evidence / regression gates ---

    @staticmethod
    def check_no_unexplained_regressions(evaluation: Evaluation) -> GateResult:
        passed = evaluation.regressions == 0
        return GateResult(
            gate_name="no_unexplained_regressions",
            passed=passed,
            details={"regressions": evaluation.regressions},
            required_approval=True,
        )

    @staticmethod
    def check_evaluation_evidence(evaluation: Evaluation) -> GateResult:
        passed = len(evaluation.evidence) > 0
        return GateResult(
            gate_name="evaluation_evidence_exists",
            passed=passed,
            details={"evidence_count": len(evaluation.evidence)},
            required_approval=True,
        )

    # --- Governance-structural gates (v0: satisfied by Governor design) ---

    @staticmethod
    def check_human_approval_mandatory(evaluation: Evaluation) -> GateResult:
        # Enforced in Governor.approve_amendment via reviewer identity checks.
        return GateResult(
            gate_name="human_approval_mandatory",
            passed=True,
            details={"enforced": "Governor.approve_amendment requires reviewer != proposer"},
            required_approval=True,
        )

    @staticmethod
    def check_every_promotion_auditable(evaluation: Evaluation) -> GateResult:
        # Enforced in Governor: every promotion writes an audit entry.
        return GateResult(
            gate_name="every_promotion_auditable",
            passed=True,
            details={"enforced": "Governor writes promotion audit entries"},
        )

    @staticmethod
    def check_every_promotion_rollbackable(evaluation: Evaluation) -> GateResult:
        # Enforced in Governor: every promoted runtime is a released, rollbackable release.
        return GateResult(
            gate_name="every_promotion_rollbackable",
            passed=True,
            details={"enforced": "RuntimeRegistry keeps full released chain; rollback_to supported"},
        )

    @staticmethod
    def check_required_suites(evaluation: Evaluation, constitution=None) -> GateResult:
        """Require every suite named in ``evaluation_rules.required_suites`` to pass.

        Each required suite must have replay evidence tagged with its ``suite_id``
        whose candidate replay correctness clears the minimum correctness
        threshold. Fails closed when evidence is missing or malformed.
        """
        rules = getattr(constitution, "evaluation_rules", {}) or {}
        required = [
            str(suite_id)
            for suite_id in rules.get("required_suites", []) or []
        ]
        min_correctness = rules.get("min_correctness", 0.80)
        if not required:
            return GateResult(
                gate_name="required_suites_pass",
                passed=True,
                details={"reason": "no required suites configured", "required_suites": []},
                required_approval=True,
            )

        seen: Dict[str, Dict[str, Any]] = {}
        for evidence in evaluation.evidence:
            results = evidence.results or {}
            suite_id = results.get("suite_id")
            if suite_id:
                seen[str(suite_id)] = results.get("candidate") or {}

        missing = [suite_id for suite_id in required if suite_id not in seen]
        failing = []
        per_suite = {}
        for suite_id in required:
            if suite_id not in seen:
                continue
            candidate = seen[suite_id]
            total = candidate.get("total", 0) or 0
            correctness = candidate.get("correctness_avg", 0.0) or 0.0
            per_suite[suite_id] = correctness
            if total == 0 or correctness < min_correctness:
                failing.append(suite_id)

        passed = not missing and not failing
        return GateResult(
            gate_name="required_suites_pass",
            passed=passed,
            details={
                "required_suites": required,
                "min_correctness": min_correctness,
                "missing": missing,
                "failing": failing,
                "suite_correctness": per_suite,
            },
            required_approval=True,
        )

    @staticmethod
    def check_all_gates(evaluation: Evaluation, constitution=None) -> Dict[str, GateResult]:
        """Evaluate all constitution-declared gates.

        Any gate declared in ``constitution.promotion_gates`` that cannot be
        evaluated (no handler) fails closed, so a widened constitution never
        silently widens the approval path.
        """
        bound = {
            "required_tests_pass": ConstitutionGates.check_required_tests(evaluation, constitution),
            "no_safety_failures": ConstitutionGates.check_no_safety_failures(evaluation, constitution),
            "no_unexplained_regressions": ConstitutionGates.check_no_unexplained_regressions(evaluation),
            "evaluation_evidence_exists": ConstitutionGates.check_evaluation_evidence(evaluation),
            "human_approval_mandatory": ConstitutionGates.check_human_approval_mandatory(evaluation),
            "every_promotion_auditable": ConstitutionGates.check_every_promotion_auditable(evaluation),
            "every_promotion_rollbackable": ConstitutionGates.check_every_promotion_rollbackable(evaluation),
            "latency_within_budget": ConstitutionGates.check_latency_delta(evaluation, constitution),
            "cost_within_budget": ConstitutionGates.check_cost_delta(evaluation, constitution),
        }
        # Bind the required-suites gate only when the constitution configures it.
        # A default Constitution() enforces nothing extra, so synthetic approval
        # tests that never attached suite evidence keep their existing semantics.
        required_configured = bool(
            (getattr(constitution, "evaluation_rules", {}) or {}).get("required_suites", [])
        )
        if required_configured or "required_suites_pass" in set(
            (getattr(constitution, "promotion_gates", {}) or {}).keys()
        ):
            bound["required_suites_pass"] = ConstitutionGates.check_required_suites(
                evaluation, constitution
            )
        results = {k: v for k, v in bound.items()}
        # Fail closed: declared gates with no bound handler cannot pass.
        declared = set((getattr(constitution, "promotion_gates", {}) or {}).keys())
        for gate in sorted(declared - set(results)):
            results[gate] = GateResult(
                gate_name=gate,
                passed=False,
                details={"reason": "gate declared in constitution but unenforced"},
            )
        return results


class RuntimeManifest(BaseModel):
    """Immutable, versioned runtime configuration.

    ``kind`` distinguishes:
    - ``released``: a runtime that went through the governance pipeline and is
      addressable as a numbered release (``release_version`` set).
    - ``sandbox``: an unreleased candidate materialized for evaluation; never
      addressable as a release and never selected as current.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    manifest_hash: str = ""
    version: str
    parent_version: Optional[str] = None

    # Release addressing (introduced for registry correctness)
    kind: Literal["released", "sandbox"] = "released"
    release_version: Optional[int] = None

    model_identifier: str
    constitution_version: str

    prompts: Dict[str, str] = Field(default_factory=dict)
    tools: Dict[str, str] = Field(default_factory=dict)
    memory: Dict[str, str] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    description: str = ""

    amendment_id: Optional[str] = None
    rollback_to: Optional[str] = None

    @model_validator(mode="after")
    def _deep_freeze(self) -> "RuntimeManifest":
        """Freeze the manifest and all nested dicts so ``frozen=True`` is a real guarantee.

        Top-level ``ConfigDict(frozen=True)`` only blocks attribute assignment;
        without this, ``runtime.prompts["system"] = "rogue"`` would still mutate
        a promoted runtime. We rebuild every container field through the
        immutable ``FrozenDict`` (object-level, so frozen=True is not violated).
        """
        object.__setattr__(
            self,
            "prompts",
            FrozenDict.freeze(self.prompts),
        )
        object.__setattr__(
            self,
            "tools",
            FrozenDict.freeze(self.tools),
        )
        object.__setattr__(
            self,
            "memory",
            FrozenDict.freeze(self.memory),
        )
        object.__setattr__(
            self,
            "evaluation",
            FrozenDict.freeze(self.evaluation),
        )
        return self


class PromotionResult(BaseModel):
    """Result of a promotion attempt."""
    success: bool
    new_runtime_id: Optional[str] = None
    reason: Optional[str] = None
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)


# --- Evaluator teeth models ---

class FailureClass(BaseModel):
    """A categorized class of failure for tracking and regression prevention."""
    id: str  # e.g. "math:incorrect-answer", "api:timeout"
    description: str = ""
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    occurrence_count: int = 0
    source_amendments: List[str] = Field(default_factory=list)


class RegressionCase(BaseModel):
    """A regression test case derived from a failure or rejection."""
    id: str
    failure_class: str
    task_id: str
    task_type: str = "general"
    input_data: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Any = None
    source: str = "auto-derived"  # "auto-derived", "rejection-derived", "manual"
    source_reference: str = ""  # amendment_id, pattern_id, run_id, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CoverageReport(BaseModel):
    """Suite coverage report: fraction of known failure classes exercised."""
    known_classes: List[str] = Field(default_factory=list)
    exercised_classes: List[str] = Field(default_factory=list)
    fraction: float = 1.0
    total_known: int = 0
    total_exercised: int = 0


# --- Amendment ---

class Amendment(BaseModel):
    """A structured amendment proposal."""
    id: str
    parent_version: str

    target: TargetType
    description: str
    rationale: str

    proposed_diff: Dict[str, Any]

    evidence: List[Evidence] = Field(default_factory=list)
    evaluation: Optional[Evaluation] = None

    # Auto-derived regression cases from the steward
    regression_cases: List[RegressionCase] = Field(default_factory=list)

    status: AmendmentStatus = AmendmentStatus.PROPOSED

    proposer: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    reviewer: Optional[str] = None
    decided_at: Optional[datetime] = None
    decision: Optional[Decision] = None

    resulting_runtime: Optional[str] = None
    amendment_chain: List[str] = Field(default_factory=list)
