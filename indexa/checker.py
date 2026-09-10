"""Name resolution, dimension checking and type checking.

Turns the parser AST into the explicit indexed core IR (:mod:`indexa.ir`),
implementing the static rules of spec sections 8-23.  All errors of one stage
are collected before the stage aborts, so a module with several independent
mistakes reports all of them at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import ast, dag, dims, ir
from .diagnostics import Code, DiagnosticBag, Span
from .stdlib import FUNCTIONS, signature_text
from .types import ScalarType as T
from .types import TensorType

_ARITH = ("+", "-", "*", "/")
_RELATIONAL = ("<", "<=", ">", ">=")
_EQUALITY = ("==", "!=")
_LOGICAL = ("&&", "||")


@dataclass
class Scope:
    """Lexical environment for one expression: index variables and locals."""

    indices: dict[str, ir.IndexVar] = field(default_factory=dict)
    locals: dict[str, ir.Local] = field(default_factory=dict)
    in_const: bool = False

    def child(self) -> "Scope":
        return Scope(dict(self.indices), dict(self.locals), self.in_const)


class Checker:
    def __init__(self, module: ast.Module, source: str | None, given_dims: dict[str, int] | None = None):
        self.ast = module
        self.diags = DiagnosticBag(source)
        self.given_dims = dict(given_dims or {})
        self.module = ir.Module(module.name.name, module.span)
        self._next_id = 0
        # name tables
        self.dims_by_name: dict[str, ir.DimDecl] = {}
        self.spaces_by_name: dict[str, ir.SpaceDecl] = {}
        self.values_by_name: dict[str, ir.Value] = {}
        self._const_asts: dict[int, ast.ConstDecl] = {}
        self._def_asts: dict[int, ast.DefDecl] = {}

    # ------------------------------------------------------------------
    def fresh(self) -> int:
        self._next_id += 1
        return self._next_id

    def run(self) -> ir.Module:
        self._declare_compile_time_names()
        self._declare_value_names()
        self.diags.check()

        self._resolve_dims()
        self._resolve_spaces()
        self._resolve_constraints()
        self._resolve_interface_types()
        self.diags.check()

        order = self._dependency_order()
        self.diags.check()

        for value_id in order:
            value = self.module.values[value_id]
            if isinstance(value, ir.Const):
                self._check_const(value, self._const_asts[value_id])
            elif isinstance(value, ir.TensorDef):
                self._check_def(value, self._def_asts[value_id])
        self._resolve_outputs()
        self.diags.check()
        return self.module

    # ------------------------------------------------------------------
    # Stage 1: declarations and duplicate detection
    # ------------------------------------------------------------------
    def _declare_compile_time_names(self) -> None:
        for item in self.ast.items:
            if isinstance(item, ast.DimDecl):
                if self._duplicate_compile_time(item.name):
                    continue
                decl = ir.DimDecl(self.fresh(), item.name.name, None, item.name.span)
                self.module.dims[decl.id] = decl
                self.dims_by_name[decl.name] = decl
            elif isinstance(item, ast.SpaceDecl):
                if self._duplicate_compile_time(item.name):
                    continue
                decl = ir.SpaceDecl(self.fresh(), item.name.name, ir.DimLit(1), item.name.span)
                self.module.spaces[decl.id] = decl
                self.spaces_by_name[decl.name] = decl

    def _duplicate_compile_time(self, name: ast.Name) -> bool:
        prev = self.dims_by_name.get(name.name) or self.spaces_by_name.get(name.name)
        if prev is not None:
            kind = "dimension" if isinstance(prev, ir.DimDecl) else "space"
            self.diags.error(
                Code.DUPLICATE_DECLARATION,
                f"`{name.name}` is already declared as a {kind}",
                name.span,
                f"previous declaration at {prev.span}",
                "dimensions and index spaces share one compile-time namespace",
            )
            return True
        return False

    def _declare_value_names(self) -> None:
        for item in self.ast.items:
            if isinstance(item, ast.InputDecl):
                self._declare_value(ir.Input(self.fresh(), item.name.name, ir.ValueKind.INPUT, item.name.span, T.ERROR), item.name)
            elif isinstance(item, ast.ParamDecl):
                self._declare_value(ir.Param(self.fresh(), item.name.name, ir.ValueKind.PARAM, item.name.span, T.ERROR), item.name)
            elif isinstance(item, ast.ConstDecl):
                const = ir.Const(self.fresh(), item.name.name, ir.ValueKind.CONST, item.name.span, T.ERROR, None)  # type: ignore[arg-type]
                if self._declare_value(const, item.name):
                    self._const_asts[const.id] = item
            elif isinstance(item, ast.DefDecl):
                tdef = ir.TensorDef(self.fresh(), item.name.name, ir.ValueKind.DEF, item.name.span, (), T.ERROR, None)  # type: ignore[arg-type]
                if self._declare_value(tdef, item.name):
                    self._def_asts[tdef.id] = item

    def _declare_value(self, value: ir.Value, name: ast.Name) -> bool:
        prev = self.values_by_name.get(name.name)
        if prev is not None:
            self.diags.error(
                Code.DUPLICATE_DECLARATION,
                f"`{name.name}` is already declared as a {prev.kind}",
                name.span,
                f"previous declaration at {prev.span}",
                "inputs, parameters, constants and tensor definitions share one value namespace",
            )
            return False
        if name.name in FUNCTIONS:
            self.diags.error(
                Code.DUPLICATE_DECLARATION,
                f"`{name.name}` is a standard-library function and cannot be redeclared",
                name.span,
            )
            return False
        self.module.values[value.id] = value
        self.values_by_name[name.name] = value
        return True

    # ------------------------------------------------------------------
    # Stage 2: dimensions, spaces, constraints, interface types
    # ------------------------------------------------------------------
    def _dim_expr(self, node: ast.DimNode) -> ir.DimExpr:
        if isinstance(node, ast.DimLit):
            return ir.DimLit(node.value)
        if isinstance(node, ast.DimName):
            decl = self.dims_by_name.get(node.name)
            if decl is None:
                if node.name in self.spaces_by_name:
                    self.diags.error(Code.UNKNOWN_NAME, f"`{node.name}` is an index space, not a dimension", node.span,
                                     "use the dimension the space was declared with")
                else:
                    self.diags.error(Code.UNKNOWN_NAME, f"unknown dimension `{node.name}`", node.span)
                return ir.DimLit(1)
            return ir.DimRef(decl.id)
        assert isinstance(node, ast.DimBinary)
        return ir.DimBinary(node.op, self._dim_expr(node.left), self._dim_expr(node.right))

    def _resolve_dims(self) -> None:
        for item in self.ast.items:
            if isinstance(item, ast.DimDecl) and item.name.name in self.dims_by_name:
                decl = self.dims_by_name[item.name.name]
                if item.value is not None:
                    decl.expr = self._dim_expr(item.value)
        # Supplied values for free dimensions.
        for name, value in self.given_dims.items():
            decl = self.dims_by_name.get(name)
            if decl is None:
                self.diags.error(Code.UNKNOWN_NAME, f"supplied value for unknown dimension `{name}`", None)
                continue
            if not decl.is_free:
                self.diags.error(Code.BAD_DIMENSION, f"dimension `{name}` is defined in the module and cannot be supplied", decl.span)
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                self.diags.error(Code.BAD_DIMENSION, f"dimension `{name}` must be a positive integer, got {value!r}", decl.span)
                continue
            decl.value = value
        if self.diags.errors:
            return

        # Cycle detection and evaluation order.
        def edges(dim_id: int) -> list[int]:
            expr = self.module.dims[dim_id].expr
            return sorted(dims.dim_refs(expr)) if expr is not None else []

        cycle = dag.find_cycle(sorted(self.module.dims), edges)
        if cycle is not None:
            names = " -> ".join(f"`{self.module.dims[d].name}`" for d in cycle)
            self.diags.error(Code.DEPENDENCY_CYCLE, f"dimension definitions form a cycle: {names}",
                             self.module.dims[cycle[0]].span)
            return
        self.module.dim_order = dag.toposort(sorted(self.module.dims), edges)

        values: dict[int, int | None] = {d.id: d.value for d in self.module.dims.values()}
        for dim_id in self.module.dim_order:
            decl = self.module.dims[dim_id]
            if decl.expr is None:
                continue
            try:
                value = dims.eval_dim(decl.expr, values)
            except dims.DimEvalError as e:
                self.diags.error(Code.BAD_DIMENSION, f"in `dim {decl.name}`: {e}", decl.span)
                continue
            if value is not None and value <= 0:
                self.diags.error(Code.BAD_DIMENSION, f"dimension `{decl.name}` evaluates to {value}; dimensions must be positive", decl.span)
                continue
            decl.value = value
            values[dim_id] = value

    def _known_dims(self) -> dict[int, int | None]:
        return {d.id: d.value for d in self.module.dims.values()}

    def _resolve_spaces(self) -> None:
        for item in self.ast.items:
            if isinstance(item, ast.SpaceDecl) and item.name.name in self.spaces_by_name:
                decl = self.spaces_by_name[item.name.name]
                decl.size = self._dim_expr(item.size)
                try:
                    decl.value = dims.eval_dim(decl.size, self._known_dims())
                except dims.DimEvalError as e:
                    self.diags.error(Code.BAD_DIMENSION, f"in `space {decl.name}`: {e}", item.size.span)
                    continue
                if decl.value is not None and decl.value <= 0:
                    self.diags.error(Code.BAD_DIMENSION,
                                     f"space `{decl.name}` has size {decl.value}; index spaces must be non-empty", item.size.span)

    def _resolve_constraints(self) -> None:
        for item in self.ast.items:
            if not isinstance(item, ast.ConstraintDecl):
                continue
            left = self._dim_expr(item.left)
            right = self._dim_expr(item.right)
            text = _source_text(self.diags.source, item.span)
            constraint = ir.Constraint(item.op, left, right, item.span, text)
            self.module.constraints.append(constraint)
            try:
                lv = dims.eval_dim(left, self._known_dims())
                rv = dims.eval_dim(right, self._known_dims())
            except dims.DimEvalError as e:
                self.diags.error(Code.BAD_DIMENSION, f"in constraint: {e}", item.span)
                continue
            if lv is not None and rv is not None and not dims.compare(item.op, lv, rv):
                self.diags.error(Code.UNSATISFIED_CONSTRAINT,
                                 f"constraint does not hold: {lv} {item.op} {rv} is false", item.span)

    def _space_ref(self, name: ast.Name) -> int | None:
        decl = self.spaces_by_name.get(name.name)
        if decl is not None:
            return decl.id
        if name.name in self.dims_by_name:
            self.diags.error(Code.UNKNOWN_NAME, f"`{name.name}` is a dimension, not an index space", name.span,
                             f"declare a space with `space S = Fin({name.name});` and use `S` here")
        else:
            self.diags.error(Code.UNKNOWN_NAME, f"unknown index space `{name.name}`", name.span)
        return None

    def _type(self, node: ast.TypeNode) -> T | TensorType:
        if isinstance(node, ast.ScalarTypeNode):
            return T.parse(node.name)
        axes = []
        for axis in node.axes:
            space_id = self._space_ref(axis)
            axes.append(space_id if space_id is not None else -1)
        return TensorType(T.parse(node.element.name), tuple(axes))

    def _resolve_interface_types(self) -> None:
        for item in self.ast.items:
            if isinstance(item, (ast.InputDecl, ast.ParamDecl)):
                value = self.values_by_name.get(item.name.name)
                if isinstance(value, (ir.Input, ir.Param)):
                    value.type = self._type(item.type)

    # ------------------------------------------------------------------
    # Stage 3: value dependency order (syntactic)
    # ------------------------------------------------------------------
    def _dependency_order(self) -> list[int]:
        deps: dict[int, set[int]] = {}
        for value_id, const in self._const_asts.items():
            deps[value_id] = self._referenced_values(const.value)
        for value_id, tdef in self._def_asts.items():
            deps[value_id] = self._referenced_values(tdef.body)
        cycle = dag.find_cycle(sorted(deps), lambda n: sorted(deps[n]))
        if cycle is not None:
            names = [self.module.values[n].name for n in cycle]
            first = self.module.values[cycle[0]]
            self.diags.error(
                Code.DEPENDENCY_CYCLE,
                f"`{first.name}` depends on itself through {' -> '.join(f'`{n}`' for n in names)}",
                first.span,
                "all value dependencies must form a directed acyclic graph",
            )
            return []
        return dag.toposort(sorted(deps), lambda n: sorted(deps[n]))

    def _referenced_values(self, expr: ast.Expr) -> set[int]:
        found: set[int] = set()

        def visit(e: ast.Expr) -> None:
            if isinstance(e, ast.TensorRead):
                v = self.values_by_name.get(e.tensor.name)
                if v is not None and isinstance(v, (ir.Const, ir.TensorDef)):
                    found.add(v.id)
            elif isinstance(e, ast.Ident):
                v = self.values_by_name.get(e.name)
                if v is not None and isinstance(v, (ir.Const, ir.TensorDef)):
                    found.add(v.id)
            elif isinstance(e, ast.Unary):
                visit(e.operand)
            elif isinstance(e, ast.Binary):
                visit(e.left)
                visit(e.right)
            elif isinstance(e, ast.Call):
                for a in e.args:
                    visit(a)
            elif isinstance(e, ast.If):
                visit(e.cond)
                visit(e.then)
                visit(e.otherwise)
            elif isinstance(e, ast.Let):
                visit(e.value)
                visit(e.body)
            elif isinstance(e, ast.Reduce):
                visit(e.body)

        visit(expr)
        return found

    # ------------------------------------------------------------------
    # Stage 4: constants, definitions and outputs
    # ------------------------------------------------------------------
    def _check_const(self, const: ir.Const, node: ast.ConstDecl) -> None:
        scope = Scope(in_const=True)
        body = self._expr(node.value, scope)
        const.body = body
        if node.type is not None:
            annotated = T.parse(node.type.name)
            if not body.type.is_error and body.type is not annotated:
                self.diags.error(Code.TYPE_MISMATCH,
                                 f"constant `{const.name}` is annotated `{annotated}` but its value has type `{body.type}`",
                                 node.value.span)
            const.type = annotated
        else:
            const.type = body.type

    def _check_def(self, tdef: ir.TensorDef, node: ast.DefDecl) -> None:
        scope = Scope()
        free: list[tuple[int, int]] = []
        for binder in node.binders:
            var = self._bind_index(binder, scope)
            if var is not None:
                free.append((var.id, var.space_id))
        tdef.free_indices = tuple(free)
        body = self._expr(node.body, scope)
        tdef.body = body
        if node.element_type is not None:
            annotated = T.parse(node.element_type.name)
            if not body.type.is_error and body.type is not annotated:
                self.diags.error(Code.TYPE_MISMATCH,
                                 f"`{tdef.name}` is annotated with element type `{annotated}` but its body has type `{body.type}`",
                                 node.body.span)
            tdef.element_type = annotated
        else:
            tdef.element_type = body.type

    def _bind_index(self, binder: ast.IndexBinder, scope: Scope) -> ir.IndexVar | None:
        """Add an index binder to ``scope``; returns None if the space is unknown."""
        name = binder.name.name
        if name in scope.indices:
            self.diags.error(Code.ILLEGAL_SHADOWING, f"index `{name}` is already bound in this scope", binder.name.span,
                             f"previous binding at {scope.indices[name].span}",
                             "v0.1 rejects lexical shadowing")
        elif name in scope.locals:
            self.diags.error(Code.ILLEGAL_SHADOWING, f"index `{name}` shadows the local scalar `{name}`", binder.name.span,
                             f"local bound at {scope.locals[name].span}")
        elif name in self.values_by_name:
            self.diags.warn(Code.STYLE, f"index `{name}` has the same spelling as the {self.values_by_name[name].kind} `{name}`",
                            binder.name.span, "the syntactic positions distinguish them, but a different name is clearer")
        space_id = self._space_ref(binder.space)
        if space_id is None:
            return None
        var = ir.IndexVar(self.fresh(), name, space_id, binder.name.span)
        self.module.indices[var.id] = var
        scope.indices[name] = var
        return var

    def _resolve_outputs(self) -> None:
        seen: dict[str, Span] = {}
        for item in self.ast.items:
            if not isinstance(item, ast.OutputDecl):
                continue
            for name in item.names:
                value = self.values_by_name.get(name.name)
                if value is None:
                    self.diags.error(Code.MISSING_OUTPUT, f"output `{name.name}` is not a declared value", name.span,
                                     "an output must name an input, parameter, constant or defined tensor")
                    continue
                if name.name in seen:
                    self.diags.error(Code.DUPLICATE_DECLARATION, f"`{name.name}` is already declared as an output", name.span,
                                     f"previous output declaration at {seen[name.name]}")
                    continue
                seen[name.name] = name.span
                self.module.outputs.append(value.id)
        if not self.module.outputs and not self.diags.errors:
            self.diags.error(Code.MISSING_OUTPUT, f"module `{self.module.name}` declares no outputs", self.ast.name.span,
                             "add `output name;` for every tensor the module produces")

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def _expr(self, node: ast.Expr, scope: Scope) -> ir.Expr:
        if isinstance(node, ast.IntLit):
            return ir.Literal(node.span, T.INT, node.value, node.text)
        if isinstance(node, ast.RealLit):
            return ir.Literal(node.span, T.REAL, node.value, node.text)
        if isinstance(node, ast.BoolLit):
            return ir.Literal(node.span, T.BOOL, node.value, "true" if node.value else "false")
        if isinstance(node, ast.Ident):
            return self._ident(node, scope)
        if isinstance(node, ast.TensorRead):
            return self._tensor_read(node, scope)
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        if isinstance(node, ast.Unary):
            return self._unary(node, scope)
        if isinstance(node, ast.Binary):
            return self._binary(node, scope)
        if isinstance(node, ast.If):
            return self._if(node, scope)
        if isinstance(node, ast.Let):
            return self._let(node, scope)
        if isinstance(node, ast.Reduce):
            return self._reduce(node, scope)
        raise AssertionError(f"unhandled AST node {type(node).__name__}")

    def _ident(self, node: ast.Ident, scope: Scope) -> ir.Expr:
        name = node.name
        if name in scope.locals:
            local = scope.locals[name]
            return ir.LocalRef(node.span, local.type, local.id)
        value = self.values_by_name.get(name)
        if value is not None:
            if isinstance(value, ir.TensorDef) or isinstance(value.type, TensorType):  # type: ignore[union-attr]
                rank = value.type.rank  # type: ignore[union-attr]
                self.diags.error(Code.WRONG_RANK, f"tensor `{name}` of rank {rank} is used without an index list", node.span,
                                 f"read it as `{name}[{', '.join('...' for _ in range(rank))}]`" if rank else f"read the rank-zero tensor as `{name}[]`")
                return ir.ScalarRef(node.span, T.ERROR, value.id)
            if scope.in_const and not isinstance(value, ir.Const):
                self.diags.error(Code.INVALID_CONST, f"a constant expression cannot refer to the {value.kind} `{name}`", node.span,
                                 "constants may use literals, other constants, dimensions and pure scalar functions")
                return ir.ScalarRef(node.span, T.ERROR, value.id)
            return ir.ScalarRef(node.span, value.type, value.id)  # type: ignore[union-attr]
        dim = self.dims_by_name.get(name)
        if dim is not None:
            return ir.DimValue(node.span, T.INT, dim.id)
        if name in scope.indices:
            self.diags.error(Code.TYPE_MISMATCH, f"index variable `{name}` cannot be used as a scalar value in v0.1", node.span,
                             "index arithmetic is a Post-MVP feature")
            return ir.Literal(node.span, T.ERROR, 0, "0")
        if name in FUNCTIONS:
            self.diags.error(Code.TYPE_MISMATCH, f"standard-library function `{name}` must be called with arguments", node.span)
            return ir.Literal(node.span, T.ERROR, 0, "0")
        if name in self.spaces_by_name:
            self.diags.error(Code.UNKNOWN_NAME, f"`{name}` is an index space, not a value", node.span)
        else:
            self.diags.error(Code.UNKNOWN_NAME, f"unknown name `{name}`", node.span)
        return ir.Literal(node.span, T.ERROR, 0, "0")

    def _tensor_read(self, node: ast.TensorRead, scope: Scope) -> ir.Expr:
        name = node.tensor.name
        value = self.values_by_name.get(name)
        if value is None:
            if name in scope.locals:
                self.diags.error(Code.NOT_A_TENSOR, f"`{name}` is a local scalar, not a tensor", node.tensor.span)
            elif name in scope.indices:
                self.diags.error(Code.NOT_A_TENSOR, f"`{name}` is an index variable, not a tensor", node.tensor.span)
            else:
                self.diags.error(Code.UNKNOWN_NAME, f"unknown tensor `{name}`", node.tensor.span)
            return ir.Literal(node.span, T.ERROR, 0, "0")
        if scope.in_const:
            self.diags.error(Code.INVALID_CONST, f"a constant expression cannot read the tensor `{name}`", node.span,
                             "constants may use literals, other constants, dimensions and pure scalar functions")
            return ir.Literal(node.span, T.ERROR, 0, "0")
        ttype = value.type  # type: ignore[union-attr]
        if not isinstance(ttype, TensorType):
            self.diags.error(Code.NOT_A_TENSOR, f"`{name}` is a scalar {value.kind} of type `{ttype}`, not a tensor", node.tensor.span,
                             f"refer to it as `{name}` without brackets")
            return ir.Literal(node.span, T.ERROR, 0, "0")
        ok = True
        if len(node.indices) != ttype.rank:
            self.diags.error(Code.WRONG_RANK,
                             f"`{name}` has rank {ttype.rank} but is read with {len(node.indices)} "
                             f"{'index' if len(node.indices) == 1 else 'indices'}", node.span,
                             f"`{name}` : {self._type_str(ttype)}")
            ok = False
        index_ids: list[int] = []
        for position, index_name in enumerate(node.indices):
            var = scope.indices.get(index_name.name)
            if var is None:
                if index_name.name in scope.locals:
                    self.diags.error(Code.UNBOUND_INDEX, f"`{index_name.name}` is a local scalar, not an index", index_name.span,
                                     "v0.1 index expressions must be index variables")
                elif index_name.name in self.values_by_name:
                    self.diags.error(Code.UNBOUND_INDEX, f"`{index_name.name}` is a {self.values_by_name[index_name.name].kind}, not an index", index_name.span)
                else:
                    self.diags.error(Code.UNBOUND_INDEX, f"index `{index_name.name}` is not bound here", index_name.span,
                                     "bind it on the left side of the definition or with a reduction such as `sum[i : S](...)`")
                ok = False
                continue
            index_ids.append(var.id)
            if position < ttype.rank and var.space_id != ttype.axes[position]:
                expected = self.module.spaces[ttype.axes[position]]
                found = self.module.spaces[var.space_id]
                notes = []
                if expected.value is not None and expected.value == found.value:
                    notes.append(f"`{expected.name}` and `{found.name}` both have size {expected.value}, but index spaces are nominal")
                else:
                    notes.append("index spaces are nominal: equal sizes never make two spaces interchangeable")
                self.diags.error(Code.INDEX_SPACE_MISMATCH,
                                 f"expected index from `{expected.name}`, found `{found.name}`", index_name.span, *notes)
                ok = False
        if not ok:
            return ir.TensorRead(node.span, T.ERROR, value.id, tuple(index_ids))
        return ir.TensorRead(node.span, ttype.element, value.id, tuple(index_ids))

    def _type_str(self, t: T | TensorType) -> str:
        if isinstance(t, TensorType):
            axes = ", ".join(self.module.spaces[s].name if s in self.module.spaces else "?" for s in t.axes)
            return f"Tensor<{t.element}>[{axes}]"
        return str(t)

    def _call(self, node: ast.Call, scope: Scope) -> ir.Expr:
        name = node.function.name
        args = [self._expr(a, scope) for a in node.args]
        signatures = FUNCTIONS.get(name)
        if signatures is None:
            if name in self.values_by_name:
                self.diags.error(Code.TYPE_MISMATCH, f"`{name}` is a {self.values_by_name[name].kind}, not a function", node.function.span)
            else:
                self.diags.error(Code.UNKNOWN_NAME, f"unknown function `{name}`", node.function.span,
                                 "v0.1 supports only the scalar standard library: " + ", ".join(sorted(FUNCTIONS)))
            return ir.Call(node.span, T.ERROR, name, args)
        arg_types = tuple(a.type for a in args)
        if any(t.is_error for t in arg_types):
            return ir.Call(node.span, T.ERROR, name, args)
        for params, result in signatures:
            if params == arg_types:
                return ir.Call(node.span, result, name, args)
        got = ", ".join(str(t) for t in arg_types)
        self.diags.error(Code.TYPE_MISMATCH, f"no signature of `{name}` accepts ({got})", node.span,
                         f"available: {signature_text(name)}")
        return ir.Call(node.span, T.ERROR, name, args)

    def _unary(self, node: ast.Unary, scope: Scope) -> ir.Expr:
        operand = self._expr(node.operand, scope)
        t = operand.type
        if t.is_error:
            return ir.Unary(node.span, T.ERROR, node.op, operand)
        if node.op == "!":
            if t is not T.BOOL:
                self.diags.error(Code.TYPE_MISMATCH, f"`!` expects a `Bool` operand, found `{t}`", node.span)
                return ir.Unary(node.span, T.ERROR, node.op, operand)
            return ir.Unary(node.span, T.BOOL, node.op, operand)
        if not t.is_numeric:
            self.diags.error(Code.TYPE_MISMATCH, f"unary `{node.op}` expects a numeric operand, found `{t}`", node.span)
            return ir.Unary(node.span, T.ERROR, node.op, operand)
        return ir.Unary(node.span, t, node.op, operand)

    def _binary(self, node: ast.Binary, scope: Scope) -> ir.Expr:
        left = self._expr(node.left, scope)
        right = self._expr(node.right, scope)
        lt, rt, op = left.type, right.type, node.op
        if lt.is_error or rt.is_error:
            return ir.Binary(node.span, T.ERROR, op, left, right)

        def fail(message: str, *notes: str) -> ir.Expr:
            self.diags.error(Code.TYPE_MISMATCH, message, node.span, *notes)
            return ir.Binary(node.span, T.ERROR, op, left, right)

        if op in _ARITH or op == "**":
            if not lt.is_numeric or not rt.is_numeric:
                bad = lt if not lt.is_numeric else rt
                return fail(f"operator `{op}` expects numeric operands, found `{bad}`")
            if lt is not rt:
                return fail(f"operator `{op}` expects operands of the same type, found `{lt}` and `{rt}`",
                            "v0.1 has no implicit numeric conversion; use `real(...)` to convert an `Int`")
            return ir.Binary(node.span, lt, op, left, right)
        if op in _RELATIONAL:
            if not lt.is_ordered or not rt.is_ordered:
                bad = lt if not lt.is_ordered else rt
                return fail(f"operator `{op}` expects `Int` or `Real` operands, found `{bad}`")
            if lt is not rt:
                return fail(f"operator `{op}` expects operands of the same type, found `{lt}` and `{rt}`",
                            "use `real(...)` to convert an `Int`")
            return ir.Binary(node.span, T.BOOL, op, left, right)
        if op in _EQUALITY:
            if lt is not rt:
                return fail(f"operator `{op}` expects operands of the same type, found `{lt}` and `{rt}`")
            return ir.Binary(node.span, T.BOOL, op, left, right)
        if op in _LOGICAL:
            if lt is not T.BOOL or rt is not T.BOOL:
                bad = lt if lt is not T.BOOL else rt
                return fail(f"operator `{op}` expects `Bool` operands, found `{bad}`")
            return ir.Binary(node.span, T.BOOL, op, left, right)
        raise AssertionError(op)

    def _if(self, node: ast.If, scope: Scope) -> ir.Expr:
        cond = self._expr(node.cond, scope)
        then = self._expr(node.then, scope)
        otherwise = self._expr(node.otherwise, scope)
        result = then.type
        if not cond.type.is_error and cond.type is not T.BOOL:
            self.diags.error(Code.TYPE_MISMATCH, f"`if` condition must be `Bool`, found `{cond.type}`", node.cond.span)
            result = T.ERROR
        if not then.type.is_error and not otherwise.type.is_error and then.type is not otherwise.type:
            self.diags.error(Code.TYPE_MISMATCH,
                             f"`if` branches must have the same type, found `{then.type}` and `{otherwise.type}`", node.span)
            result = T.ERROR
        if otherwise.type.is_error:
            result = T.ERROR
        return ir.If(node.span, result, cond, then, otherwise)

    def _let(self, node: ast.Let, scope: Scope) -> ir.Expr:
        name = node.name.name
        value = self._expr(node.value, scope)
        clash = None
        if name in scope.locals:
            clash = f"local `{name}` bound at {scope.locals[name].span}"
        elif name in scope.indices:
            clash = f"index `{name}` bound at {scope.indices[name].span}"
        elif name in self.values_by_name:
            clash = f"the {self.values_by_name[name].kind} `{name}` declared at {self.values_by_name[name].span}"
        elif name in self.dims_by_name:
            clash = f"the dimension `{name}`"
        elif name in FUNCTIONS:
            clash = f"the standard-library function `{name}`"
        if clash is not None:
            self.diags.error(Code.ILLEGAL_SHADOWING, f"`let {name}` shadows {clash}", node.name.span,
                             "v0.1 rejects lexical shadowing")
        local = ir.Local(self.fresh(), name, value.type, node.name.span)
        self.module.locals[local.id] = local
        inner = scope.child()
        inner.locals[name] = local
        body = self._expr(node.body, inner)
        return ir.Let(node.span, body.type, local.id, value, body)

    def _reduce(self, node: ast.Reduce, scope: Scope) -> ir.Expr:
        if scope.in_const:
            self.diags.error(Code.INVALID_CONST, "a constant expression cannot contain a reduction", node.span)
            return ir.Literal(node.span, T.ERROR, 0, "0")
        inner = scope.child()
        bound: list[tuple[int, int]] = []
        for binder in node.binders:
            var = self._bind_index(binder, inner)
            if var is not None:
                bound.append((var.id, var.space_id))
        body = self._expr(node.body, inner)
        t = body.type
        if t.is_error:
            return ir.Reduce(node.span, T.ERROR, node.op, tuple(bound), body)
        if node.op in ("sum", "prod"):
            if not t.is_numeric:
                self.diags.error(Code.INVALID_REDUCTION_BODY, f"`{node.op}` needs a numeric body, found `{t}`", node.body.span)
                t = T.ERROR
        elif node.op in ("max", "min"):
            if not t.is_ordered:
                self.diags.error(Code.INVALID_REDUCTION_BODY, f"`{node.op}` needs an `Int` or `Real` body, found `{t}`", node.body.span)
                t = T.ERROR
        else:  # all / any
            if t is not T.BOOL:
                self.diags.error(Code.INVALID_REDUCTION_BODY, f"`{node.op}` needs a `Bool` body, found `{t}`", node.body.span)
                t = T.ERROR
        return ir.Reduce(node.span, t, node.op, tuple(bound), body)


def _source_text(source: str | None, span: Span) -> str:
    if source is None:
        return "<constraint>"
    text = source[span.start : span.end].strip()
    if text.startswith("constraint"):
        text = text[len("constraint"):].strip()
    return text.rstrip(";").strip()


def check(module: ast.Module, source: str | None = None, dims: dict[str, int] | None = None) -> tuple[ir.Module, DiagnosticBag]:
    """Resolve and type-check a parsed module.  Raises :class:`CompileError`."""
    checker = Checker(module, source, dims)
    result = checker.run()
    return result, checker.diags
