"""Surface abstract syntax tree produced by the parser (spec sections 7-21).

The AST still uses source names everywhere.  The checker resolves it into the
core IR of :mod:`indexa.ir`, where names are replaced by stable ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .diagnostics import Span


# --- dimension expressions ---------------------------------------------------


@dataclass
class DimNode:
    span: Span


@dataclass
class DimLit(DimNode):
    value: int


@dataclass
class DimName(DimNode):
    name: str


@dataclass
class DimBinary(DimNode):
    op: str  # + - * /
    left: DimNode
    right: DimNode


# --- types --------------------------------------------------------------------


@dataclass
class ScalarTypeNode:
    span: Span
    name: str  # Bool Int Real Complex


@dataclass
class TensorTypeNode:
    span: Span
    element: ScalarTypeNode
    axes: list["Name"]


TypeNode = ScalarTypeNode | TensorTypeNode


@dataclass
class Name:
    span: Span
    name: str


# --- scalar expressions -------------------------------------------------------


@dataclass
class Expr:
    span: Span


@dataclass
class IntLit(Expr):
    text: str

    @property
    def value(self) -> int:
        return int(self.text)


@dataclass
class RealLit(Expr):
    text: str

    @property
    def value(self) -> float:
        return float(self.text)


@dataclass
class BoolLit(Expr):
    value: bool


@dataclass
class Ident(Expr):
    name: str


@dataclass
class TensorRead(Expr):
    tensor: Name
    indices: list[Name]


@dataclass
class Call(Expr):
    function: Name
    args: list[Expr]


@dataclass
class Unary(Expr):
    op: str  # + - !
    operand: Expr


@dataclass
class Binary(Expr):
    op: str  # + - * / ** == != < <= > >= && ||
    left: Expr
    right: Expr


@dataclass
class If(Expr):
    cond: Expr
    then: Expr
    otherwise: Expr


@dataclass
class Let(Expr):
    name: Name
    value: Expr
    body: Expr


@dataclass
class IndexBinder:
    span: Span
    name: Name
    space: Name


@dataclass
class Reduce(Expr):
    op: str  # sum prod max min all any
    binders: list[IndexBinder]
    body: Expr


# --- module items -------------------------------------------------------------


@dataclass
class Item:
    span: Span


@dataclass
class DimDecl(Item):
    name: Name
    value: DimNode | None


@dataclass
class SpaceDecl(Item):
    name: Name
    size: DimNode


@dataclass
class ConstraintDecl(Item):
    op: str
    left: DimNode
    right: DimNode


@dataclass
class InputDecl(Item):
    name: Name
    type: TypeNode


@dataclass
class ParamDecl(Item):
    name: Name
    type: TypeNode


@dataclass
class ConstDecl(Item):
    name: Name
    type: ScalarTypeNode | None
    value: Expr


@dataclass
class DefDecl(Item):
    name: Name
    binders: list[IndexBinder]
    element_type: ScalarTypeNode | None
    body: Expr


@dataclass
class OutputDecl(Item):
    names: list[Name]


@dataclass
class Module:
    span: Span
    name: Name
    items: list[Item] = field(default_factory=list)
