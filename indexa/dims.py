"""Compile-time dimension evaluation (spec sections 8-10)."""

from __future__ import annotations

from . import ir


class DimEvalError(Exception):
    """Raised for a division that is not exact or a division by zero."""


def eval_dim(expr: ir.DimExpr, values: dict[int, int | None]) -> int | None:
    """Evaluate a dimension expression.

    ``values`` maps dimension ids to known values (or ``None`` when the
    dimension is only known at runtime).  Returns ``None`` when the result
    depends on an unknown dimension.
    """
    if isinstance(expr, ir.DimLit):
        return expr.value
    if isinstance(expr, ir.DimRef):
        return values.get(expr.dim_id)
    assert isinstance(expr, ir.DimBinary)
    left = eval_dim(expr.left, values)
    right = eval_dim(expr.right, values)
    if expr.op == "/" and right == 0:
        raise DimEvalError("division by zero in dimension expression")
    if left is None or right is None:
        return None
    if expr.op == "+":
        return left + right
    if expr.op == "-":
        return left - right
    if expr.op == "*":
        return left * right
    if expr.op == "/":
        if left % right != 0:
            raise DimEvalError(f"{left} is not divisible by {right}; dimension division must be exact")
        return left // right
    raise AssertionError(expr.op)


def compare(op: str, left: int, right: int) -> bool:
    return {
        "==": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }[op]


def dim_refs(expr: ir.DimExpr) -> set[int]:
    if isinstance(expr, ir.DimLit):
        return set()
    if isinstance(expr, ir.DimRef):
        return {expr.dim_id}
    assert isinstance(expr, ir.DimBinary)
    return dim_refs(expr.left) | dim_refs(expr.right)
