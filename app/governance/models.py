"""Core models and governance gates for the governed evolving AI runtime."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class TargetType(str, Enum):
    PROMPT = "prompt"
    TOOL = "tool"
    MEMORY = "memory"
    MODEL = "model"
    POLICY = "policy"


class Evidence(BaseModel):
    """Structured evidence from evaluation."""
    id: str
    type: str  # "replay", "synthetic", "human"
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
    def check_required_tests(evaluation: Evaluation) -> GateResult:
        passed = evaluation.correctness >= 0.80 and evaluation.instruction_following >= 0.85
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
    def check_no_safety_failures(evaluation: Evaluation) -> GateResult:
        passed = evaluation.safety >= 1.0 and evaluation.regressions == 0
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
    def check_all_gates(evaluation: Evaluation) -> Dict[str, GateResult]:
        return {
            "required_tests_pass": ConstitutionGates.check_required_tests(evaluation),
            "no_safety_failures": ConstitutionGates.check_no_safety_failures(evaluation),
            "no_unexplained_regressions": ConstitutionGates.check_no_unexplained_regressions(evaluation),
            "evaluation_evidence_exists": ConstitutionGates.check_evaluation_evidence(evaluation),
        }


class RuntimeManifest(BaseModel):
    """Immutable, versioned runtime configuration."""
    model_config = ConfigDict(frozen=True)

    id: str
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


class PromotionResult(BaseModel):
    """Result of a promotion attempt."""
    success: bool
    new_runtime_id: Optional[str] = None
    reason: Optional[str] = None
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)


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

    status: AmendmentStatus = AmendmentStatus.PROPOSED

    proposer: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    reviewer: Optional[str] = None
    decided_at: Optional[datetime] = None
    decision: Optional[Decision] = None

    resulting_runtime: Optional[str] = None
    amendment_chain: List[str] = Field(default_factory=list)
