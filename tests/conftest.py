import pytest

from indexa import compile_source
from indexa.diagnostics import CompileError


@pytest.fixture
def compile_ok():
    """Compile source text; fail the test with rendered diagnostics on error."""

    def run(source: str, dims: dict[str, int] | None = None, filename: str = "test.a"):
        try:
            return compile_source(source, filename, dims)
        except CompileError as e:  # pragma: no cover - only on failure
            pytest.fail("unexpected compile error:\n" + e.render())

    return run


@pytest.fixture
def compile_err():
    """Compile source text expecting failure; returns the CompileError."""

    def run(source: str, dims: dict[str, int] | None = None, filename: str = "test.a") -> CompileError:
        with pytest.raises(CompileError) as info:
            compile_source(source, filename, dims)
        return info.value

    return run
