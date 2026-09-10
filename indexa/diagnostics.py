"""Source spans, diagnostics and the compile-error exception.

Diagnostics are rendered in a rustc-like layout (see spec section 29)::

    error[E0204]: index-space mismatch
      --> model.a:18:23
       |
    18 |     def y[o : Output] = x[o];
       |                           ^ expected index from `Input`, found `Output`
       |
       = note: `Input` and `Output` both have size 768, but index spaces are nominal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Span:
    """A half-open source region.  Lines and columns are 1-based."""

    file: str
    line: int
    col: int
    end_line: int
    end_col: int
    #: Absolute character offsets into the source text (half-open).
    start: int = 0
    end: int = 0

    def to(self, other: "Span") -> "Span":
        """The smallest span covering ``self`` and ``other``."""
        return Span(self.file, self.line, self.col, other.end_line, other.end_col, self.start, other.end)

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class Code(str, Enum):
    """Diagnostic codes.  The numbering groups errors by compiler stage."""

    # Lexing / parsing
    SYNTAX = "E0001"
    # Name resolution
    UNKNOWN_NAME = "E0101"
    DUPLICATE_DECLARATION = "E0102"
    ILLEGAL_SHADOWING = "E0103"
    # Indices and shapes
    UNBOUND_INDEX = "E0201"
    WRONG_RANK = "E0202"
    NOT_A_TENSOR = "E0203"
    INDEX_SPACE_MISMATCH = "E0204"
    # Scalar typing
    TYPE_MISMATCH = "E0301"
    INVALID_REDUCTION_BODY = "E0302"
    INVALID_CONST = "E0303"
    # Dimensions
    BAD_DIMENSION = "E0401"
    UNSATISFIED_CONSTRAINT = "E0402"
    # Dependencies and outputs
    DEPENDENCY_CYCLE = "E0501"
    MISSING_OUTPUT = "E0502"
    # Backend
    UNSUPPORTED = "E0601"
    # Warnings
    STYLE = "W0001"
    UNUSED = "W0002"

    @property
    def title(self) -> str:
        return _TITLES[self]

    @property
    def is_warning(self) -> bool:
        return self.value.startswith("W")


_TITLES = {
    Code.SYNTAX: "syntax error",
    Code.UNKNOWN_NAME: "unknown name",
    Code.DUPLICATE_DECLARATION: "duplicate declaration",
    Code.ILLEGAL_SHADOWING: "illegal lexical shadowing",
    Code.UNBOUND_INDEX: "undefined or unbound index",
    Code.WRONG_RANK: "wrong tensor rank",
    Code.NOT_A_TENSOR: "value is not a tensor",
    Code.INDEX_SPACE_MISMATCH: "index-space mismatch",
    Code.TYPE_MISMATCH: "scalar type mismatch",
    Code.INVALID_REDUCTION_BODY: "invalid reduction body",
    Code.INVALID_CONST: "invalid constant expression",
    Code.BAD_DIMENSION: "non-positive or non-integral dimension",
    Code.UNSATISFIED_CONSTRAINT: "unsatisfied dimension constraint",
    Code.DEPENDENCY_CYCLE: "dependency cycle",
    Code.MISSING_OUTPUT: "missing output",
    Code.UNSUPPORTED: "unsupported construct in the current backend",
    Code.STYLE: "style warning",
    Code.UNUSED: "unused declaration",
}


@dataclass
class Diagnostic:
    code: Code
    message: str
    span: Span | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return not self.code.is_warning

    def render(self, source: str | None = None) -> str:
        kind = "warning" if self.code.is_warning else "error"
        lines = [f"{kind}[{self.code.value}]: {self.code.title}"]
        if self.span is None:
            lines[0] += f": {self.message}"
        else:
            lines.append(f"  --> {self.span}")
            src_line = _source_line(source, self.span.line)
            gutter = " " * len(str(self.span.line))
            lines.append(f"{gutter} |")
            if src_line is not None:
                lines.append(f"{self.span.line} | {src_line}")
                if self.span.end_line == self.span.line:
                    width = max(1, self.span.end_col - self.span.col)
                else:
                    width = max(1, len(src_line) - self.span.col + 1)
                lines.append(f"{gutter} | {' ' * (self.span.col - 1)}{'^' * width} {self.message}")
            else:
                lines.append(f"{gutter} | {self.message}")
            lines.append(f"{gutter} |")
            for note in self.notes:
                lines.append(f"{gutter} = note: {note}")
        if self.span is None:
            for note in self.notes:
                lines.append(f"  = note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _source_line(source: str | None, line: int) -> str | None:
    if source is None:
        return None
    lines = source.split("\n")
    if 1 <= line <= len(lines):
        return lines[line - 1].replace("\t", "    ")
    return None


class CompileError(Exception):
    """Raised when a compilation stage produced one or more errors."""

    def __init__(self, diagnostics: list[Diagnostic], source: str | None = None):
        self.diagnostics = list(diagnostics)
        self.source = source
        super().__init__(self.render())

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def codes(self) -> list[Code]:
        return [d.code for d in self.diagnostics]

    def has(self, code: Code) -> bool:
        return any(d.code is code for d in self.diagnostics)

    def render(self) -> str:
        return "\n\n".join(d.render(self.source) for d in self.diagnostics)


class DiagnosticBag:
    """Collects diagnostics for one stage; ``check()`` raises if any are errors."""

    def __init__(self, source: str | None = None):
        self.source = source
        self.items: list[Diagnostic] = []

    def error(self, code: Code, message: str, span: Span | None, *notes: str) -> None:
        self.items.append(Diagnostic(code, message, span, list(notes)))

    def warn(self, code: Code, message: str, span: Span | None, *notes: str) -> None:
        self.items.append(Diagnostic(code, message, span, list(notes)))

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.items if d.is_error]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.items if not d.is_error]

    def check(self) -> None:
        if self.errors:
            raise CompileError(self.items, self.source)
