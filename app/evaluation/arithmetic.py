"""Restricted arithmetic evaluator shared by Operator, Evaluator, and Steward.

Walks the AST and evaluates a minimal arithmetic subset, replacing unsafe
dynamic code execution in application code with a safe, auditable path.

Allowed tokens:
- integer and float constants (no booleans)
- binary +, -, *, /
- unary + and -
- parentheses (implicit in AST)

Rejected:
- names, function calls, attributes, subscripts, strings
- list/dict/set/display/comprehensions
- import statements, f-strings
- exponentiation (``**``), modulo (``%``), floor division (``//``)
- boolean constants (``True``/``False``)
- complex numbers
"""
from __future__ import annotations

import ast
import operator
from typing import Any


__all__ = ["safe_arithmetic"]

_BINOP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARYOP_MAP = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _visit(node: ast.AST) -> Any:
    """Evaluate an AST node inside the restricted arithmetic language."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ArithmeticError(f"unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op = _BINOP_MAP.get(type(node.op))
        if op is None:
            raise ArithmeticError(f"unsupported operator: {type(node.op).__name__}")
        left = _visit(node.left)
        right = _visit(node.right)
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOP_MAP.get(type(node.op))
        if op is None:
            raise ArithmeticError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_visit(node.operand))

    if isinstance(node, ast.Expression):
        return _visit(node.body)

    # Every other AST node type (Call, Attribute, Subscript, Name, Import,
    # comprehension, list/dict/set literal, f-string, etc.) is disallowed.
    raise ArithmeticError(
        f"forbidden AST node: {type(node).__name__}"
    )


def safe_arithmetic(expression: str) -> Any:
    """Parse and evaluate an arithmetic expression safely.

    Returns a numeric ``int`` or ``float``.  Raises ``ArithmeticError``
    on any expression outside the allowed subset.  The caller decides
    how to format or convert the return value.
    """
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as e:
        raise ArithmeticError(f"invalid arithmetic syntax: {e.msg}") from e
    return _visit(tree)
