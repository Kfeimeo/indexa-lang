"""PyTorch code generation (spec section 26).

Lowering strategy
-----------------
Every sub-expression of a definition is lowered to a tensor whose axes are a
subset of the definition's *canonical index order*: the left-hand-side indices
in declared order followed by reduction binders in order of appearance.  Because
every intermediate keeps its axes in that order, aligning two operands only
ever needs ``None`` insertions (``bias[None, :]``) and never a transpose; the
only permutes are on tensor reads whose axis order differs from the canonical
one.

* A ``sum`` over a product of tensor reads is recognized as a contraction and
  lowered to ``torch.einsum``.
* Other reductions lower to ``torch.sum`` / ``torch.amax`` / ``torch.amin`` /
  ``torch.prod`` / ``torch.all`` / ``torch.any`` over the bound axes.
* An omitted left-side index is replicated with ``expand``.
* ``let`` locals become temporaries; ``if`` becomes ``torch.where``.

The generated module exposes ``forward(...)`` plus ``resolve_dims(...)`` which
infers unresolved module dimensions from the runtime shapes and asserts every
interface shape.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass

from .. import dag, ir
from ..diagnostics import Code, CompileError, Diagnostic
from ..types import ScalarType as T
from ..types import TensorType

# Python operator precedence used to parenthesize generated expressions.
ATOM, POW, UNARY, MUL, ADD, BAND, BOR, CMP = 100, 90, 80, 70, 60, 50, 40, 30

_BINARY_PREC = {"*": MUL, "/": MUL, "+": ADD, "-": ADD, "&&": BAND, "||": BOR,
                "==": CMP, "!=": CMP, "<": CMP, "<=": CMP, ">": CMP, ">=": CMP}
_PY_BINARY = {"&&": "&", "||": "|"}

_TORCH_FUNCTIONS = {
    "abs": "torch.abs", "sign": "torch.sign", "exp": "torch.exp", "log": "torch.log",
    "sqrt": "torch.sqrt", "sin": "torch.sin", "cos": "torch.cos", "tanh": "torch.tanh",
    "relu": "torch.relu", "sigmoid": "torch.sigmoid", "floor": "torch.floor", "ceil": "torch.ceil",
    "scalar_min": "torch.minimum", "scalar_max": "torch.maximum",
}

_EINSUM_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

_RUNTIME_HELPERS = '''
class ShapeError(ValueError):
    """Raised when runtime shapes or dimensions violate the module interface."""


def _exact_div(numerator, denominator, what):
    if denominator == 0:
        raise ShapeError(f"division by zero while computing dimension `{what}`")
    if numerator % denominator != 0:
        raise ShapeError(
            f"dimension `{what}`: {numerator} is not divisible by {denominator} (dimension division must be exact)"
        )
    return numerator // denominator


def _positive(name, value):
    if value <= 0:
        raise ShapeError(f"dimension `{name}` evaluates to {value}; dimensions must be positive")
    return value


def _free_dim(name, given, sources):
    """Resolve an uninitialized module dimension from a keyword argument or from tensor shapes."""
    value = None
    if given is not None:
        value = int(given)
        if value <= 0:
            raise ShapeError(f"dimension `{name}` must be positive, got {given!r}")
    for tensor_name, tensor, axis in sources:
        if tensor.dim() <= axis:
            raise ShapeError(f"`{tensor_name}` has rank {tensor.dim()} but axis {axis} is needed to infer dimension `{name}`")
        size = tensor.shape[axis]
        if value is None:
            value = size
        elif size != value:
            raise ShapeError(
                f"dimension `{name}` is {value} but `{tensor_name}` has size {size} on axis {axis}"
            )
    if value is None:
        raise ShapeError(f"dimension `{name}` cannot be inferred from any input; pass it as a keyword argument")
    return value


def _check_constraint(holds, text, left, op, right):
    if not holds:
        raise ShapeError(f"constraint `{text}` is violated: {left} {op} {right} is false")


def _check_shape(name, kind, tensor, expected, type_text):
    if tuple(tensor.shape) != tuple(expected):
        raise ShapeError(
            f"{kind} `{name}` has shape {tuple(tensor.shape)} but `{type_text}` requires {tuple(expected)}"
        )
'''


@dataclass(frozen=True)
class Lowered:
    """A lowered sub-expression."""

    code: str
    prec: int
    #: Index ids of the tensor axes, in canonical order.  Empty for scalars.
    indices: tuple[int, ...]
    #: True when ``code`` is a plain Python number/bool rather than a tensor.
    scalar: bool = False

    def atom(self) -> str:
        return f"({self.code})" if self.prec < ATOM else self.code


class _Unsupported(Exception):
    pass


class TorchGenerator:
    def __init__(self, module: ir.Module, source: str | None = None):
        self.module = module
        self.source = source
        self.names = self._assign_names()
        #: Const ids whose generated value is a plain Python scalar rather than a tensor.
        self.const_scalar: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    def _assign_names(self) -> dict[str, str]:
        """Map every A value/local/dim/space name to a Python identifier.

        Internal helper names live in ``reserved``; an A name that collides
        with one of them (or with a Python keyword) gets a ``_`` suffix.
        """
        m = self.module
        reserved = {"torch", "dims", "given", "ShapeError", "forward", "resolve_dims"} | set(keyword.kwlist)
        reserved |= {f"size_{s.name}" for s in m.spaces.values()}
        reserved |= {"_exact_div", "_positive", "_free_dim", "_check_constraint", "_check_shape"}
        names: dict[str, str] = {}
        a_names = [v.name for v in m.values.values()] + [loc.name for loc in m.locals.values()]
        for name in a_names:
            py = name
            while py in reserved:
                py += "_"
            names[name] = py
        return names

    def py(self, name: str) -> str:
        return self.names[name]

    def size_name(self, space_id: int) -> str:
        return f"size_{self.module.spaces[space_id].name}"

    def type_text(self, t: T | TensorType) -> str:
        if isinstance(t, TensorType):
            return f"Tensor<{t.element}>[{', '.join(self.module.spaces[s].name for s in t.axes)}]"
        return str(t)

    # ------------------------------------------------------------------
    # Module layout
    # ------------------------------------------------------------------
    def generate(self) -> str:
        m = self.module
        inputs = m.inputs()
        params = m.params()
        interface = inputs + params
        free_dims = [d for d in m.dims.values() if d.is_free and d.value is None]
        arg_list = ", ".join(self.py(v.name) for v in interface)
        kw_list = "".join(f", {d.name}=None" for d in free_dims)
        signature = arg_list + (", *" + kw_list if free_dims else "")
        forward_call_args = arg_list + "".join(f", {d.name}={d.name}" for d in free_dims)

        out: list[str] = []
        w = out.append
        w(f'"""PyTorch program generated by indexa from the A module `{m.name}`.')
        w("")
        w("Do not edit by hand; regenerate from the `.a` source instead.")
        w('"""')
        w("")
        w("import torch")
        w("")
        w(f"MODULE_NAME = {m.name!r}")
        w(f"INPUTS = {_tuple(repr(v.name) for v in inputs)}")
        w(f"PARAMS = {_tuple(repr(v.name) for v in params)}")
        w(f"OUTPUTS = {_tuple(repr(m.values[o].name) for o in m.outputs)}")
        w("#: Dimension values fixed at compile time; ``None`` marks a dimension resolved at runtime.")
        w("DIMS = {" + ", ".join(f"{m.dims[d].name!r}: {m.dims[d].value!r}" for d in m.dim_order) + "}")
        w(f"FREE_DIMS = {_tuple(repr(d.name) for d in free_dims)}")
        w("")
        w(_RUNTIME_HELPERS.rstrip("\n"))
        w("")
        w("")
        w(f"def resolve_dims({signature}):")
        w('    """Resolve every module dimension, check constraints and return them as a dict."""')
        w("    dims = {}")
        for line in self._dim_resolution(interface):
            w("    " + line)
        w("    return dims")
        w("")
        w("")
        w(f"def forward({signature}):")
        w(f'    """Evaluate module `{m.name}`; returns {self._return_doc()}."""')
        for v in interface:
            w(f"    {self.py(v.name)} = torch.as_tensor({self.py(v.name)})")
        w(f"    dims = resolve_dims({forward_call_args})")
        for space_id in sorted(m.spaces, key=lambda s: m.spaces[s].name):
            w(f"    {self.size_name(space_id)} = {self._dim_code(m.spaces[space_id].size, m.spaces[space_id].name)}")
        for v in interface:
            for line in self._shape_check(v):
                w("    " + line)
        w("")
        for value_id in dag.required_values(m):
            value = m.values[value_id]
            if isinstance(value, ir.Const):
                for line in self._const(value):
                    w("    " + line)
            elif isinstance(value, ir.TensorDef):
                for line in self._definition(value):
                    w("    " + line)
        results = ", ".join(self._output_code(o) for o in m.outputs)
        w(f"    return {results}")
        w("")
        return "\n".join(out)

    def _output_code(self, value_id: int) -> str:
        name = self.py(self.module.values[value_id].name)
        if self.const_scalar.get(value_id):
            return f"torch.tensor({name})"  # constants are returned as rank-zero tensors
        return name

    def _return_doc(self) -> str:
        names = [self.module.values[o].name for o in self.module.outputs]
        if len(names) == 1:
            return f"`{names[0]}`"
        return "the tuple (" + ", ".join(f"`{n}`" for n in names) + ")"

    def _dim_code(self, expr: ir.DimExpr, what: str) -> str:
        """Python code evaluating a dimension expression against ``dims``."""
        if isinstance(expr, ir.DimLit):
            return str(expr.value)
        if isinstance(expr, ir.DimRef):
            decl = self.module.dims[expr.dim_id]
            return f'dims[{decl.name!r}]'
        assert isinstance(expr, ir.DimBinary)
        left = self._dim_code(expr.left, what)
        right = self._dim_code(expr.right, what)
        if expr.op == "/":
            return f"_exact_div({left}, {right}, {what!r})"
        wrap = lambda e, code: f"({code})" if isinstance(e, ir.DimBinary) and e.op in "+-" and expr.op == "*" else code
        return f"{wrap(expr.left, left)} {expr.op} {wrap(expr.right, right)}"

    def _dim_resolution(self, interface: list[ir.Value]) -> list[str]:
        m = self.module
        lines: list[str] = []
        for dim_id in m.dim_order:
            decl = m.dims[dim_id]
            if decl.value is not None:
                lines.append(f"dims[{decl.name!r}] = {decl.value}")
            elif decl.is_free:
                sources = []
                for v in interface:
                    t = v.type  # type: ignore[union-attr]
                    if isinstance(t, TensorType):
                        for axis, space_id in enumerate(t.axes):
                            size = m.spaces[space_id].size
                            if isinstance(size, ir.DimRef) and size.dim_id == dim_id:
                                sources.append(f"({v.name!r}, {self.py(v.name)}, {axis})")
                src = _tuple(sources)
                lines.append(f"dims[{decl.name!r}] = _free_dim({decl.name!r}, {decl.name}, {src})")
            else:
                assert decl.expr is not None
                lines.append(f"dims[{decl.name!r}] = _positive({decl.name!r}, {self._dim_code(decl.expr, decl.name)})")
        for c in m.constraints:
            left = self._dim_code(c.left, "constraint")
            right = self._dim_code(c.right, "constraint")
            lines.append(f"_check_constraint({left} {c.op} {right}, {c.text!r}, {left}, {c.op!r}, {right})")
        return lines

    def _shape_check(self, v: ir.Value) -> list[str]:
        t = v.type  # type: ignore[union-attr]
        name = self.py(v.name)
        if isinstance(t, TensorType):
            shape = _tuple(self.size_name(s) for s in t.axes)
            return [f"_check_shape({v.name!r}, {v.kind!r}, {name}, {shape}, {self.type_text(t)!r})"]
        return [f"_check_shape({v.name!r}, {v.kind!r}, {name}, (), {self.type_text(t)!r})"]

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------
    def _const(self, const: ir.Const) -> list[str]:
        ctx = _DefContext(self, {}, ())
        lowered = ctx.lower(const.body)
        self.const_scalar[const.id] = lowered.scalar
        return ctx.statements + [f"{self.py(const.name)} = {lowered.code}"]

    def _definition(self, tdef: ir.TensorDef) -> list[str]:
        canonical: dict[int, int] = {}
        for index_id, _ in tdef.free_indices:
            canonical[index_id] = len(canonical)
        for node in ir.walk(tdef.body):
            if isinstance(node, ir.Reduce):
                for index_id, _ in node.bound:
                    if index_id not in canonical:
                        canonical[index_id] = len(canonical)
        lhs = tuple(i for i, _ in tdef.free_indices)
        ctx = _DefContext(self, canonical, lhs)
        try:
            lowered = ctx.lower(tdef.body)
            code = ctx.expand_to(lowered, lhs)
        except _Unsupported as e:
            raise CompileError([Diagnostic(Code.UNSUPPORTED, str(e), tdef.span)], self.source) from None
        return ctx.statements + [f"{self.py(tdef.name)} = {code}"]


class _DefContext:
    """Per-definition lowering state."""

    def __init__(self, gen: TorchGenerator, canonical: dict[int, int], lhs: tuple[int, ...]):
        self.gen = gen
        self.module = gen.module
        self.canonical = canonical
        self.lhs = lhs
        self.statements: list[str] = []
        self.const_scalar = gen.const_scalar
        self._letters: dict[int, str] = {}
        self._locals: dict[int, Lowered] = {}

    # -- index bookkeeping -------------------------------------------------
    def merge(self, *index_lists: tuple[int, ...]) -> tuple[int, ...]:
        ids = {i for lst in index_lists for i in lst}
        return tuple(sorted(ids, key=self.canonical.__getitem__))

    def size_of(self, index_id: int) -> str:
        return self.gen.size_name(self.module.indices[index_id].space_id)

    def letter(self, index_id: int) -> str:
        if index_id not in self._letters:
            if len(self._letters) >= len(_EINSUM_LETTERS):
                raise _Unsupported("too many distinct indices for torch.einsum")
            self._letters[index_id] = _EINSUM_LETTERS[len(self._letters)]
        return self._letters[index_id]

    # -- alignment ------------------------------------------------------------
    def as_tensor(self, l: Lowered) -> Lowered:
        if l.scalar:
            return Lowered(f"torch.tensor({l.code})", ATOM, ())
        return l

    def align(self, l: Lowered, target: tuple[int, ...]) -> Lowered:
        """Insert ``None`` axes so ``l`` broadcasts against ``target``.

        Python scalars and rank-zero tensors broadcast without indexing.
        """
        if l.scalar or not l.indices or l.indices == target:
            return l
        assert set(l.indices) <= set(target), (l.indices, target)
        if l.indices == target[len(target) - len(l.indices):]:
            return l  # trailing axes already line up under PyTorch's right-aligned broadcasting
        subscript = ", ".join(":" if i in l.indices else "None" for i in target)
        return Lowered(f"{l.atom()}[{subscript}]", ATOM, target)

    def expand_to(self, l: Lowered, target: tuple[int, ...]) -> str:
        """Materialize ``l`` with exactly the axes ``target`` (replicating missing ones)."""
        t = self.as_tensor(l)
        if t.indices == target:
            return t.code
        if not target:
            return t.code
        if t.indices:
            aligned = self.align(t, target)
            code = aligned.code
        else:
            code = f"{t.atom()}[{', '.join('None' for _ in target)}]"
        sizes = ", ".join(self.size_of(i) for i in target)
        return f"{code}.expand({sizes})"

    # -- expressions ------------------------------------------------------------
    def lower(self, expr: ir.Expr) -> Lowered:
        method = getattr(self, "_lower_" + type(expr).__name__)
        return method(expr)

    def _lower_Literal(self, e: ir.Literal) -> Lowered:
        if e.type is T.BOOL:
            return Lowered("True" if e.value else "False", ATOM, (), scalar=True)
        return Lowered(e.text, ATOM, (), scalar=True)

    def _lower_ScalarRef(self, e: ir.ScalarRef) -> Lowered:
        value = self.module.values[e.value_id]
        scalar = self.const_scalar.get(e.value_id, False)
        return Lowered(self.gen.py(value.name), ATOM, (), scalar=scalar)

    def _lower_LocalRef(self, e: ir.LocalRef) -> Lowered:
        return self._locals[e.local_id]

    def _lower_DimValue(self, e: ir.DimValue) -> Lowered:
        return Lowered(f"dims[{self.module.dims[e.dim_id].name!r}]", ATOM, (), scalar=True)

    def _lower_TensorRead(self, e: ir.TensorRead) -> Lowered:
        name = self.gen.py(self.module.values[e.tensor_id].name)
        axes = e.index_ids
        unique = self.merge(axes)
        if axes == unique:
            return Lowered(name, ATOM, axes)
        if len(set(axes)) == len(axes):
            perm = ", ".join(str(axes.index(i)) for i in unique)
            return Lowered(f"{name}.permute({perm})", ATOM, unique)
        spec = "".join(self.letter(i) for i in axes) + "->" + "".join(self.letter(i) for i in unique)
        return Lowered(f'torch.einsum("{spec}", {name})', ATOM, unique)

    def _lower_Unary(self, e: ir.Unary) -> Lowered:
        operand = self.lower(e.operand)
        if e.op == "!":
            if operand.scalar:
                return Lowered(f"(not {operand.code})", ATOM, (), scalar=True)
            return Lowered(f"~{_paren(operand, UNARY)}", UNARY, operand.indices)
        if e.op == "+":
            return operand
        return Lowered(f"-{_paren(operand, UNARY)}", UNARY, operand.indices, operand.scalar)

    def _lower_Binary(self, e: ir.Binary) -> Lowered:
        left = self.lower(e.left)
        right = self.lower(e.right)
        target = self.merge(left.indices, right.indices)
        left = self.align(left, target)
        right = self.align(right, target)
        scalar = left.scalar and right.scalar
        if e.op == "**":
            return Lowered(f"{_paren(left, POW + 1)} ** {_paren(right, POW)}", POW, target, scalar)
        if e.op == "/" and e.type is T.INT:
            if scalar:
                return Lowered(f"{_paren(left, MUL)} // {_paren(right, MUL + 1)}", MUL, target, True)
            return Lowered(f'torch.div({left.code}, {right.code}, rounding_mode="floor")', ATOM, target)
        if e.op in ("&&", "||") and not scalar:
            # Bool tensors combine with & and |; a Python bool operand must become a tensor.
            left, right = self.as_tensor(left), self.as_tensor(right)
        prec = _BINARY_PREC[e.op]
        pyop = _PY_BINARY.get(e.op, e.op)
        return Lowered(f"{_paren(left, prec)} {pyop} {_paren(right, prec + 1)}", prec, target, scalar)

    def _lower_Call(self, e: ir.Call) -> Lowered:
        args = [self.lower(a) for a in e.args]
        target = self.merge(*(a.indices for a in args))
        args = [self.align(a, target) for a in args]
        if e.function == "real":
            (arg,) = args
            if arg.scalar:
                return Lowered(f"float({arg.code})", ATOM, (), scalar=True)
            if e.args[0].type is T.REAL:
                return arg
            return Lowered(f"{arg.atom()}.to(torch.get_default_dtype())", ATOM, target)
        tensor_args = [self.as_tensor(a) for a in args]
        fn = _TORCH_FUNCTIONS[e.function]
        return Lowered(f"{fn}({', '.join(a.code for a in tensor_args)})", ATOM, target)

    def _lower_If(self, e: ir.If) -> Lowered:
        cond = self.lower(e.cond)
        then = self.lower(e.then)
        otherwise = self.lower(e.otherwise)
        if cond.scalar and then.scalar and otherwise.scalar:
            return Lowered(f"({then.code} if {cond.code} else {otherwise.code})", ATOM, (), scalar=True)
        target = self.merge(cond.indices, then.indices, otherwise.indices)
        parts = [self.as_tensor(self.align(x, target)) for x in (cond, then, otherwise)]
        return Lowered(f"torch.where({', '.join(p.code for p in parts)})", ATOM, target)

    def _lower_Let(self, e: ir.Let) -> Lowered:
        value = self.lower(e.value)
        local = self.module.locals[e.local_id]
        name = self.gen.py(local.name)
        self.statements.append(f"{name} = {value.code}")
        self._locals[e.local_id] = Lowered(name, ATOM, value.indices, value.scalar)
        return self.lower(e.body)

    def _lower_Reduce(self, e: ir.Reduce) -> Lowered:
        bound = tuple(i for i, _ in e.bound)
        contraction = self._try_einsum(e, bound)
        if contraction is not None:
            return contraction
        body = self.lower(e.body)
        target = self.merge(body.indices, bound)
        code = self.expand_to(body, target)
        positions = [target.index(i) for i in bound]
        result = tuple(i for i in target if i not in bound)
        dim_arg = str(positions[0]) if len(positions) == 1 else "(" + ", ".join(map(str, positions)) + ")"
        if e.op in ("sum", "max", "min", "all", "any"):
            fn = {"sum": "torch.sum", "max": "torch.amax", "min": "torch.amin", "all": "torch.all", "any": "torch.any"}[e.op]
            return Lowered(f"{fn}({code}, dim={dim_arg})", ATOM, result)
        assert e.op == "prod"
        for p in sorted(positions, reverse=True):  # torch.prod reduces one axis at a time
            code = f"torch.prod({code}, dim={p})"
        return Lowered(code, ATOM, result)

    def _try_einsum(self, e: ir.Reduce, bound: tuple[int, ...]) -> Lowered | None:
        """Recognize ``sum[...](A[..] * B[..] * ...)`` as a contraction."""
        if e.op != "sum" or e.type not in (T.REAL, T.COMPLEX):
            return None
        leaves: list[ir.TensorRead] = []

        def collect(node: ir.Expr) -> bool:
            if isinstance(node, ir.Binary) and node.op == "*":
                return collect(node.left) and collect(node.right)
            if isinstance(node, ir.TensorRead):
                leaves.append(node)
                return True
            return False

        if not collect(e.body) or len(leaves) < 2:
            return None
        used = {i for leaf in leaves for i in leaf.index_ids}
        if not set(bound) <= used:
            return None
        result = tuple(i for i in self.merge(*(leaf.index_ids for leaf in leaves)) if i not in bound)
        operands = ", ".join(self.gen.py(self.module.values[leaf.tensor_id].name) for leaf in leaves)
        spec = ",".join("".join(self.letter(i) for i in leaf.index_ids) for leaf in leaves)
        spec += "->" + "".join(self.letter(i) for i in result)
        return Lowered(f'torch.einsum("{spec}", {operands})', ATOM, result)


def _tuple(items) -> str:
    items = list(items)
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"


def _paren(l: Lowered, min_prec: int) -> str:
    return f"({l.code})" if l.prec < min_prec else l.code


def generate_torch(module: ir.Module, source: str | None = None) -> str:
    """Generate a Python/PyTorch module implementing ``module``."""
    gen = TorchGenerator(module, source)
    try:
        return gen.generate()
    except _Unsupported as e:
        raise CompileError([Diagnostic(Code.UNSUPPORTED, str(e), module.span)], source) from None
