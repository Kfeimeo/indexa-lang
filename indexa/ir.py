"""Explicit indexed core IR (spec section 28).

All names are resolved to stable integer ids.  Expression nodes carry their
scalar type and a source span.  The IR keeps reductions explicit; the backend
recognizes contraction patterns on this representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .diagnostics import Span
from .types import ScalarType, TensorType

# --- dimensions ----------------------------------------------------------------


@dataclass(frozen=True)
class DimExpr:
    pass


@dataclass(frozen=True)
class DimLit(DimExpr):
    value: int


@dataclass(frozen=True)
class DimRef(DimExpr):
    dim_id: int


@dataclass(frozen=True)
class DimBinary(DimExpr):
    op: str  # + - * /
    left: DimExpr
    right: DimExpr


@dataclass
class DimDecl:
    id: int
    name: str
    #: ``None`` for an uninitialized (module-parameter) dimension.
    expr: DimExpr | None
    span: Span
    #: Value known at compile time (from the declaration or a supplied dim), if any.
    value: int | None = None

    @property
    def is_free(self) -> bool:
        """True for a dimension that must be supplied before specialization."""
        return self.expr is None


@dataclass
class SpaceDecl:
    id: int
    name: str
    size: DimExpr
    span: Span
    value: int | None = None


@dataclass
class Constraint:
    op: str
    left: DimExpr
    right: DimExpr
    span: Span
    #: Source text of the constraint for runtime error messages.
    text: str


# --- expressions -----------------------------------------------------------------


@dataclass
class Expr:
    span: Span
    type: ScalarType


@dataclass
class Literal(Expr):
    #: Source text for Int/Real literals; ``True``/``False`` for Bool.
    value: int | float | bool
    text: str


@dataclass
class ScalarRef(Expr):
    """Reference to a scalar-typed module value (input, param or const)."""

    value_id: int


@dataclass
class LocalRef(Expr):
    """Reference to a ``let``-bound local."""

    local_id: int


@dataclass
class DimValue(Expr):
    """A dimension used as an ``Int`` scalar (for `real(N)` and similar)."""

    dim_id: int


@dataclass
class TensorRead(Expr):
    tensor_id: int
    index_ids: tuple[int, ...]


@dataclass
class Unary(Expr):
    op: str
    operand: Expr


@dataclass
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class Call(Expr):
    function: str
    args: list[Expr]


@dataclass
class If(Expr):
    cond: Expr
    then: Expr
    otherwise: Expr


@dataclass
class Let(Expr):
    local_id: int
    value: Expr
    body: Expr


@dataclass
class Reduce(Expr):
    op: str
    #: (index_id, space_id) pairs in binder order.
    bound: tuple[tuple[int, int], ...]
    body: Expr


# --- values -------------------------------------------------------------------------


class ValueKind:
    INPUT = "input"
    PARAM = "param"
    CONST = "const"
    DEF = "def"


@dataclass
class Value:
    id: int
    name: str
    kind: str
    span: Span


@dataclass
class Input(Value):
    type: ScalarType | TensorType


@dataclass
class Param(Value):
    type: ScalarType | TensorType


@dataclass
class Const(Value):
    type: ScalarType
    body: Expr


@dataclass
class TensorDef(Value):
    #: (index_id, space_id) pairs in declared (axis) order.
    free_indices: tuple[tuple[int, int], ...]
    element_type: ScalarType
    body: Expr

    @property
    def type(self) -> TensorType:
        return TensorType(self.element_type, tuple(s for _, s in self.free_indices))


@dataclass
class IndexVar:
    id: int
    name: str
    space_id: int
    span: Span


@dataclass
class Local:
    id: int
    name: str
    type: ScalarType
    span: Span


@dataclass
class Module:
    name: str
    span: Span
    dims: dict[int, DimDecl] = field(default_factory=dict)
    spaces: dict[int, SpaceDecl] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    values: dict[int, Value] = field(default_factory=dict)
    indices: dict[int, IndexVar] = field(default_factory=dict)
    locals: dict[int, Local] = field(default_factory=dict)
    #: Output value ids in declaration order.
    outputs: list[int] = field(default_factory=list)
    #: Dimension ids in a valid evaluation order.
    dim_order: list[int] = field(default_factory=list)
    #: Value ids (consts and defs) in a valid evaluation order (set by dag validation).
    value_order: list[int] = field(default_factory=list)

    # -- convenience accessors --------------------------------------------
    def dim_named(self, name: str) -> DimDecl:
        return next(d for d in self.dims.values() if d.name == name)

    def space_named(self, name: str) -> SpaceDecl:
        return next(s for s in self.spaces.values() if s.name == name)

    def value_named(self, name: str) -> Value:
        return next(v for v in self.values.values() if v.name == name)

    def inputs(self) -> list[Input]:
        return [v for v in self.values.values() if isinstance(v, Input)]

    def params(self) -> list[Param]:
        return [v for v in self.values.values() if isinstance(v, Param)]

    def consts(self) -> list[Const]:
        return [v for v in self.values.values() if isinstance(v, Const)]

    def defs(self) -> list[TensorDef]:
        return [v for v in self.values.values() if isinstance(v, TensorDef)]

    def value_type(self, value_id: int) -> ScalarType | TensorType:
        return self.values[value_id].type  # type: ignore[union-attr]


def dim_expr_str(expr: DimExpr, module: Module) -> str:
    if isinstance(expr, DimLit):
        return str(expr.value)
    if isinstance(expr, DimRef):
        return module.dims[expr.dim_id].name
    assert isinstance(expr, DimBinary)
    return f"({dim_expr_str(expr.left, module)} {expr.op} {dim_expr_str(expr.right, module)})"


def walk(expr: Expr):
    """Pre-order traversal of an expression tree."""
    yield expr
    if isinstance(expr, Unary):
        yield from walk(expr.operand)
    elif isinstance(expr, Binary):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, Call):
        for a in expr.args:
            yield from walk(a)
    elif isinstance(expr, If):
        yield from walk(expr.cond)
        yield from walk(expr.then)
        yield from walk(expr.otherwise)
    elif isinstance(expr, Let):
        yield from walk(expr.value)
        yield from walk(expr.body)
    elif isinstance(expr, Reduce):
        yield from walk(expr.body)


def value_dependencies(expr: Expr) -> set[int]:
    """Ids of module values read by an expression."""
    deps: set[int] = set()
    for node in walk(expr):
        if isinstance(node, TensorRead):
            deps.add(node.tensor_id)
        elif isinstance(node, ScalarRef):
            deps.add(node.value_id)
    return deps
