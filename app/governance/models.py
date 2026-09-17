"""Core models and governance gates for the governed evolving AI runtime."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

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
    """Structured evidence from evaluation."""
    id: str
    type: str  # "replay", "synthetic", "human", "auto-derived"
    description: str
    results: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    runtime_version: str


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
    """Machine-enforced promotion gates based on the constitution."""

    @staticmethod
    def check_required_tests(evaluation: Evaluation, constitution=None) -> GateResult:
        rules = getattr(constitution, "evaluation_rules", {})
        passed = evaluation.correctness >= rules.get("min_correctness", 0.80) and evaluation.instruction_following >= rules.get("min_instruction_following", 0.85)
        return GateResult(
            gate_name="required_tests_pass",
            passed=passed,
            details={
                "correctness": evaluation.correctness,
                "instruction_following": evaluation.instruction_following,
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
            details={"safety": evaluation.safety, "regressions": evaluation.regressions},
            required_approval=True,
        )

    @staticmethod
    def check_no_unexplained_regressions(evaluation: Evaluation) -> GateResult:
        passed = evaluation.regressions == 0
        return GateResult(
            gate_name="no_unexplained_regressions",
            passed=passed,
            details={"regressions": evaluation.regressions},
        )

    @staticmethod
    def check_evaluation_evidence(evaluation: Evaluation) -> GateResult:
        passed = len(evaluation.evidence) > 0
        return GateResult(
            gate_name="evaluation_evidence_exists",
            passed=passed,
            details={"evidence_count": len(evaluation.evidence)},
        )

    @staticmethod
    def check_all_gates(evaluation: Evaluation, constitution=None) -> Dict[str, GateResult]:
        return {
            "required_tests_pass": ConstitutionGates.check_required_tests(evaluation, constitution),
            "no_safety_failures": ConstitutionGates.check_no_safety_failures(evaluation, constitution),
            "no_unexplained_regressions": ConstitutionGates.check_no_unexplained_regressions(evaluation),
            "evaluation_evidence_exists": ConstitutionGates.check_evaluation_evidence(evaluation),
        }


class RuntimeManifest(BaseModel):
    """Immutable, versioned runtime configuration."""
    model_config = ConfigDict(frozen=True)

    id: str
    manifest_hash: str = ""
    version: str
    parent_version: Optional[str] = None

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
