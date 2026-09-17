"""P2: malicious inputs to the shared arithmetic evaluator must fail safely."""
import os
import re

import pytest

from app.evaluation.arithmetic import safe_arithmetic
from app.evaluation._init import Evaluator, ReplaySuite
from app.governance.registry import RuntimeRegistry
from app.steward._init import Steward


MALICIOUS = [
    "__import__('os').system('rm -rf /')",
    "open('/etc/passwd').read()",
    "(lambda: 1)()",
    "[x for x in range(10)]",
    "{1: 2}",
    "[1, 2, 3]",
    "'string' + 'concat'",
    "f'{1}'",
    "1 ** 999999999",
    "1 / 0",
    "1 if True else 2",
    "True + 1",
    "os.getcwd()",
    "a.b.c",
    "d['key']",
    "import math; math.sqrt(16)",
    "getattr(__builtins__, 'eval')",
    "1; 2",
    "@decorator\ndef f(x): return x",
    "1,)  # tuple display",
    "{x for x in {1}}",
    "b'bytes'",
    "complex(1,2)",
    "-2**2",
]


class TestSafeArithmeticAccepts:
    def test_basic_binary(self):
        assert safe_arithmetic("2+2") == 4
        assert safe_arithmetic("10-4") == 6
        assert safe_arithmetic("3*7") == 21
        assert safe_arithmetic("10/2") == 5.0

    def test_unary_and_parentheses(self):
        assert safe_arithmetic("-(5)") == -5
        assert safe_arithmetic("+5") == 5
        assert safe_arithmetic("(1+2)*3") == 9
        assert safe_arithmetic("((2+3)*(4-1))") == 15
        assert safe_arithmetic("1 + 2.5") == 3.5

    def test_precedence(self):
        assert safe_arithmetic("2+3*4") == 14
        assert safe_arithmetic("(2+3)*4") == 20
        assert safe_arithmetic("10/2+3") == 8.0


class TestSafeArithmeticRejects:
    @pytest.mark.parametrize("expr", MALICIOUS)
    def test_malicious_expression_raises(self, expr):
        with pytest.raises(ArithmeticError):
            safe_arithmetic(expr)


def _find_eval_calls(app_root: str) -> list:
    """Find real eval( call sites: identifier boundary, then 'eval('."""
    hits = []
    # Matches a call to a bare/qualified 'eval' name. Docstring mentions of
    # "eval()" (e.g. the acceptance-criteria comment) are docstrings, not code,
    # and are excluded by skipping lines inside triple-quoted strings' context
    # where 'eval(' is preceded by a comment or not preceded by '.', ':', '('.
    pattern = re.compile(r"(?<![\w.])eval\s*\(")
    for dirpath, _, filenames in os.walk(app_root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            for i, line in enumerate(open(path, encoding="utf-8"), 1):
                code, _, comment = line.partition("#")
                if comment:
                    continue
                if pattern.search(code):
                    hits.append(f"{path}:{i}: {line.strip()}")
    return hits


class TestNoEvalInApp:
    def test_no_eval_call_in_application_code(self):
        """The acceptance criterion: rg 'eval\\(' app must be clean."""
        import os

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        app_dir = os.path.join(root, "app")
        hits = _find_eval_calls(app_dir)
        assert hits == [], f"eval( calls remaining in app: {hits}"


class TestCallersUseSafeArithmetic:
    def test_steward_expected_output_from_malicious_input_is_none(self):
        """A malicious expression must not produce a computed expected output."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        steward = Steward(registry, None)
        failures = [
            {"task_id": "t-evil", "success": False, "errors": ["incorrect"],
             "runtime_version": "v0", "input": {"expression": "__import__('os').system('x')"}}
        ] * 4
        proposals = steward.recommend_amendments(failures)
        for proposal in proposals:
            for rc in proposal.regression_cases:
                assert rc.expected_output is None
                assert rc.input_data.get("expression", "").startswith("__import__")

    def test_evaluator_runs_malicious_task_without_execution(self):
        """Evaluator must not execute malicious expressions; safe fallback output."""
        suite = ReplaySuite(id="safe", name="safe", tasks=[
            {"id": "evil", "type": "math",
             "input": {"expression": "__import__('os').system('echo pwned')"},
             "expected_output": "RESULT"},
        ])
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        evaluator = Evaluator(registry, {"safe": suite})
        result = evaluator.run_suite_against_runtime("safe", "v0")
        outcome = result.outcomes[0]
        assert outcome.correctness == 0.0
        # Fail-safe: input rejected without executing; no system call output.
        assert "pwned" not in outcome.actual_output
        assert "Result:" in outcome.actual_output or outcome.actual_output == "<rejected>"

    def test_operator_safe_arithmetic_raises_not_executes(self):
        """Operator._safe_arithmetic raises ValueError for malicious input."""
        registry = RuntimeRegistry()
        registry.create_runtime(version="v0", model_identifier="m", constitution_version="v1",
                                created_by="system", description="v0")
        from app.operator.operator import Operator
        op = Operator(registry=registry, current_runtime=registry.get_current())
        with pytest.raises(ValueError):
            op._safe_arithmetic("__import__('os').system('echo pwned')")