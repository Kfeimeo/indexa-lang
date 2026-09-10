"""The compiler pipeline (spec section 28)::

    source -> lexer -> parser AST -> checker (names, dims, types -> core IR)
           -> DAG validation -> PyTorch code generation
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from . import dag, ir
from .backend import generate_torch
from .checker import check
from .diagnostics import Diagnostic, DiagnosticBag
from .parser import parse


@dataclass
class CompileResult:
    module: ir.Module
    python: str
    warnings: list[Diagnostic] = field(default_factory=list)

    def load(self, name: str | None = None) -> types.ModuleType:
        """Execute the generated Python source and return it as a module object."""
        return load_generated(self.python, name or f"indexa_generated_{self.module.name}")


def analyze(source: str, filename: str = "<input>", dims: dict[str, int] | None = None) -> tuple[ir.Module, list[Diagnostic]]:
    """Parse, resolve and type-check; returns the core IR and any warnings."""
    tree = parse(source, filename)
    module, diags = check(tree, source, dims)
    bag = DiagnosticBag(source)
    dag.validate(module, bag)
    bag.check()
    return module, diags.warnings


def compile_source(source: str, filename: str = "<input>", dims: dict[str, int] | None = None) -> CompileResult:
    """Compile A source text to a PyTorch Python module."""
    module, warnings = analyze(source, filename, dims)
    python = generate_torch(module, source)
    return CompileResult(module, python, warnings)


def compile_file(path: str | Path, dims: dict[str, int] | None = None) -> CompileResult:
    path = Path(path)
    return compile_source(path.read_text(encoding="utf-8"), str(path), dims)


def load_generated(python: str, name: str = "indexa_generated") -> types.ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = f"<{name}>"
    exec(compile(python, f"<{name}>", "exec"), module.__dict__)
    sys.modules[name] = module
    return module
