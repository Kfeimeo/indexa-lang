"""indexa: compiler for the A indexed tensor-algebra language (spec v0.1)."""

from .compiler import CompileResult, analyze, compile_file, compile_source, load_generated
from .diagnostics import Code, CompileError, Diagnostic

__all__ = [
    "CompileResult",
    "Code",
    "CompileError",
    "Diagnostic",
    "analyze",
    "compile_file",
    "compile_source",
    "load_generated",
]

__version__ = "0.1.0"
