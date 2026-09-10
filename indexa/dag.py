"""Dependency-graph utilities: cycle detection and topological ordering.

Used for dimension declarations, constant/tensor definitions (spec sections
23.9 and 25) and for pruning the generated program to what the outputs need.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import TypeVar

from . import ir
from .diagnostics import Code, DiagnosticBag

T = TypeVar("T", bound=Hashable)


def find_cycle(nodes: Iterable[T], edges: Callable[[T], Iterable[T]]) -> list[T] | None:
    """Return a cycle as a list ``[a, b, ..., a]`` if one exists."""
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[T, int] = {}
    stack: list[T] = []

    def visit(n: T) -> list[T] | None:
        color[n] = GREY
        stack.append(n)
        for m in edges(n):
            c = color.get(m, WHITE)
            if c == GREY:
                start = stack.index(m)
                return stack[start:] + [m]
            if c == WHITE:
                found = visit(m)
                if found:
                    return found
        stack.pop()
        color[n] = BLACK
        return None

    for n in nodes:
        if color.get(n, WHITE) == WHITE:
            found = visit(n)
            if found:
                return found
    return None


def toposort(nodes: Iterable[T], edges: Callable[[T], Iterable[T]]) -> list[T]:
    """Dependencies-first order.  Assumes the graph is acyclic; keeps the
    input order where no dependency forces otherwise."""
    order: list[T] = []
    seen: set[T] = set()

    def visit(n: T) -> None:
        if n in seen:
            return
        seen.add(n)
        for m in edges(n):
            visit(m)
        order.append(n)

    for n in nodes:
        visit(n)
    return order


def validate(module: ir.Module, diags: DiagnosticBag) -> None:
    """Check that value dependencies form a DAG and record an evaluation order."""
    deps: dict[int, set[int]] = {}
    for v in module.values.values():
        if isinstance(v, (ir.Const, ir.TensorDef)):
            deps[v.id] = {d for d in ir.value_dependencies(v.body) if d in module.values}
        else:
            deps[v.id] = set()

    cycle = find_cycle(list(deps), lambda n: sorted(deps[n]))
    if cycle is not None:
        names = [module.values[n].name for n in cycle]
        first = module.values[cycle[0]]
        diags.error(
            Code.DEPENDENCY_CYCLE,
            f"`{first.name}` depends on itself through {' -> '.join(f'`{n}`' for n in names)}",
            first.span,
            "all value dependencies must form a directed acyclic graph",
        )
        return
    module.value_order = toposort(sorted(deps), lambda n: sorted(deps[n]))


def required_values(module: ir.Module) -> list[int]:
    """Ids of the values needed to compute the outputs, in evaluation order."""
    deps: dict[int, set[int]] = {}
    for v in module.values.values():
        if isinstance(v, (ir.Const, ir.TensorDef)):
            deps[v.id] = ir.value_dependencies(v.body)
        else:
            deps[v.id] = set()
    needed = toposort(list(module.outputs), lambda n: sorted(deps[n]))
    return [n for n in module.value_order if n in set(needed)]
