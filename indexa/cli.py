"""Command-line interface: ``indexa compile|check|ir``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ir
from .compiler import analyze, compile_file
from .diagnostics import CompileError


def _parse_dims(items: list[str] | None) -> dict[str, int]:
    dims: dict[str, int] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"error: --dim expects NAME=VALUE, got {item!r}")
        try:
            dims[name.strip()] = int(value)
        except ValueError:
            raise SystemExit(f"error: dimension `{name.strip()}` must be an integer, got {value!r}") from None
    return dims


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="A source file (.a)")
    p.add_argument("--dim", action="append", metavar="NAME=VALUE", help="fix an uninitialized dimension at compile time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indexa", description="Compiler for the A indexed tensor-algebra language.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="compile a module to a PyTorch Python file")
    _add_common(c)
    c.add_argument("-o", "--output", help="output .py path (default: stdout)")

    k = sub.add_parser("check", help="parse and type-check a module")
    _add_common(k)

    d = sub.add_parser("ir", help="print the resolved core IR")
    _add_common(d)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dims = _parse_dims(args.dim)
    path = Path(args.source)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    try:
        if args.command == "compile":
            result = compile_file(path, dims)
            for w in result.warnings:
                print(w.render(path.read_text(encoding="utf-8")), file=sys.stderr)
            if args.output:
                Path(args.output).write_text(result.python, encoding="utf-8")
                print(f"wrote {args.output}", file=sys.stderr)
            else:
                sys.stdout.write(result.python)
        else:
            source = path.read_text(encoding="utf-8")
            module, warnings = analyze(source, str(path), dims)
            for w in warnings:
                print(w.render(source), file=sys.stderr)
            if args.command == "check":
                print(f"ok: module `{module.name}` type-checks "
                      f"({len(module.defs())} definitions, {len(module.outputs)} outputs)")
            else:
                print(dump_ir(module))
    except CompileError as e:
        print(e.render(), file=sys.stderr)
        n = len(e.errors)
        print(f"\nerror: could not compile `{path}` due to {n} previous error{'s' if n != 1 else ''}", file=sys.stderr)
        return 1
    return 0


def dump_ir(m: ir.Module) -> str:
    """Human-readable rendering of the core IR."""
    lines = [f"module {m.name}"]
    for dim_id in m.dim_order:
        d = m.dims[dim_id]
        rhs = "" if d.expr is None else f" = {ir.dim_expr_str(d.expr, m)}"
        known = f"  // = {d.value}" if d.value is not None else ""
        lines.append(f"  dim #{d.id} {d.name}{rhs};{known}")
    for s in m.spaces.values():
        known = f"  // = {s.value}" if s.value is not None else ""
        lines.append(f"  space #{s.id} {s.name} = Fin({ir.dim_expr_str(s.size, m)});{known}")
    for c in m.constraints:
        lines.append(f"  constraint {ir.dim_expr_str(c.left, m)} {c.op} {ir.dim_expr_str(c.right, m)};")
    for v in m.values.values():
        if isinstance(v, (ir.Input, ir.Param)):
            lines.append(f"  {v.kind} #{v.id} {v.name} : {_type_str(m, v.type)};")
    for value_id in m.value_order:
        v = m.values[value_id]
        if isinstance(v, ir.Const):
            lines.append(f"  const #{v.id} {v.name} : {v.type} = {expr_str(m, v.body)};")
        elif isinstance(v, ir.TensorDef):
            binders = ", ".join(f"{m.indices[i].name}#{i} : {m.spaces[s].name}" for i, s in v.free_indices)
            lines.append(f"  def #{v.id} {v.name}[{binders}] : {v.element_type} =")
            lines.append(f"      {expr_str(m, v.body)};")
    lines.append("  output " + ", ".join(m.values[o].name for o in m.outputs) + ";")
    return "\n".join(lines)


def _type_str(m: ir.Module, t) -> str:
    if hasattr(t, "axes"):
        return f"Tensor<{t.element}>[{', '.join(m.spaces[s].name for s in t.axes)}]"
    return str(t)


def expr_str(m: ir.Module, e: ir.Expr) -> str:
    if isinstance(e, ir.Literal):
        return e.text
    if isinstance(e, ir.ScalarRef):
        return f"{m.values[e.value_id].name}#{e.value_id}"
    if isinstance(e, ir.LocalRef):
        return f"{m.locals[e.local_id].name}#{e.local_id}"
    if isinstance(e, ir.DimValue):
        return f"dim({m.dims[e.dim_id].name})"
    if isinstance(e, ir.TensorRead):
        return f"{m.values[e.tensor_id].name}#{e.tensor_id}[{', '.join(f'{m.indices[i].name}#{i}' for i in e.index_ids)}]"
    if isinstance(e, ir.Unary):
        return f"{e.op}{expr_str(m, e.operand)}"
    if isinstance(e, ir.Binary):
        return f"({expr_str(m, e.left)} {e.op} {expr_str(m, e.right)})"
    if isinstance(e, ir.Call):
        return f"{e.function}({', '.join(expr_str(m, a) for a in e.args)})"
    if isinstance(e, ir.If):
        return f"(if {expr_str(m, e.cond)} then {expr_str(m, e.then)} else {expr_str(m, e.otherwise)})"
    if isinstance(e, ir.Let):
        return f"(let {m.locals[e.local_id].name}#{e.local_id} = {expr_str(m, e.value)} in {expr_str(m, e.body)})"
    if isinstance(e, ir.Reduce):
        binders = ", ".join(f"{m.indices[i].name}#{i} : {m.spaces[s].name}" for i, s in e.bound)
        return f"{e.op}[{binders}]({expr_str(m, e.body)})"
    raise AssertionError(type(e))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
