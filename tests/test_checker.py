"""Static checks: names, indices, spaces, types, dimensions, cycles, outputs."""

import pytest

from indexa import analyze, compile_source
from indexa.diagnostics import Code
from indexa.types import ScalarType as T

PRELUDE = """
module M {
    dim N = 768; dim K = 4;
    space Input = Fin(N); space Output = Fin(N); space Small = Fin(K);
    input x : Tensor<Real>[Input];
    input m : Tensor<Bool>[Input];
    input n : Int;
    param w : Tensor<Real>[Input, Small];
    %s
}
"""


def module(body: str) -> str:
    return PRELUDE % body


# --- acceptance test 4: nominal index spaces ------------------------------------


def test_rejects_wrong_nominal_space_with_equal_size(compile_err):
    e = compile_err(module("def y[o : Output] = x[o]; output y;"), filename="model.a")
    assert e.codes == [Code.INDEX_SPACE_MISMATCH]
    text = e.render()
    assert "expected index from `Input`, found `Output`" in text
    assert "both have size 768, but index spaces are nominal" in text
    assert "model.a:" in text


def test_rejects_wrong_space_in_tensor_type_axes(compile_err):
    e = compile_err(module("def y[i : Input, s : Small] = w[s, i]; output y;"))
    assert e.codes == [Code.INDEX_SPACE_MISMATCH, Code.INDEX_SPACE_MISMATCH]


# --- acceptance test 5: unbound index -------------------------------------------


def test_rejects_unbound_index(compile_err):
    e = compile_err(module("def y[i : Input] = x[j]; output y;"))
    assert e.codes == [Code.UNBOUND_INDEX]
    assert "index `j` is not bound here" in e.render()


def test_reduction_index_not_visible_outside_body(compile_err):
    e = compile_err(module("def y[] = sum[i : Input](x[i]) + x[i]; output y;"))
    assert e.codes == [Code.UNBOUND_INDEX]


# --- acceptance test 6: cycles ---------------------------------------------------


def test_rejects_cyclic_pair_of_definitions(compile_err):
    e = compile_err(module("def a[i : Input] = b[i] * 2.0; def b[i : Input] = a[i] + 1.0; output a;"))
    assert e.codes == [Code.DEPENDENCY_CYCLE]
    assert "`a` -> `b` -> `a`" in e.render()


def test_rejects_self_reference(compile_err):
    e = compile_err(module("def a[i : Input] : Real = a[i]; output a;"))
    assert e.codes == [Code.DEPENDENCY_CYCLE]


def test_rejects_cyclic_constants(compile_err):
    e = compile_err(module("const p : Real = q * 2.0; const q : Real = p; def y[] = p; output y;"))
    assert e.codes == [Code.DEPENDENCY_CYCLE]


def test_rejects_cyclic_dimensions(compile_err):
    e = compile_err("module M { dim A = B; dim B = A; space S = Fin(A); input x : Tensor<Real>[S]; output x; }")
    assert e.codes == [Code.DEPENDENCY_CYCLE]


# --- names -------------------------------------------------------------------------


def test_unknown_names(compile_err):
    e = compile_err(module("def y[i : Input] = zz[i] + foo(1.0) + bar; output y;"))
    assert e.codes == [Code.UNKNOWN_NAME, Code.UNKNOWN_NAME, Code.UNKNOWN_NAME]


def test_unknown_space_and_dimension(compile_err):
    e = compile_err("module M { space S = Fin(Nope); input x : Tensor<Real>[T]; output x; }")
    assert e.codes == [Code.UNKNOWN_NAME, Code.UNKNOWN_NAME]


def test_dimension_used_as_space_is_explained(compile_err):
    e = compile_err("module M { dim N = 2; input x : Tensor<Real>[N]; output x; }")
    assert e.codes == [Code.UNKNOWN_NAME]
    assert "is a dimension, not an index space" in e.render()


def test_duplicate_declarations(compile_err):
    e = compile_err(module("input x : Real; dim N; space Input = Fin(N); def y[] = 1.0; def y[] = 2.0; output y;"))
    assert e.codes == [Code.DUPLICATE_DECLARATION] * 4


def test_duplicate_output(compile_err):
    e = compile_err(module("def y[] = 1.0; output y, y;"))
    assert e.codes == [Code.DUPLICATE_DECLARATION]


def test_cannot_redeclare_stdlib_function(compile_err):
    e = compile_err(module("def relu[] = 1.0; output relu;"))
    assert e.codes == [Code.DUPLICATE_DECLARATION]


def test_missing_output(compile_err):
    assert compile_err(module("def y[] = 1.0;")).codes == [Code.MISSING_OUTPUT]
    assert compile_err(module("def y[] = 1.0; output nope;")).codes == [Code.MISSING_OUTPUT]


def test_outputs_may_be_inputs_params_and_consts(compile_ok):
    r = compile_ok(module("const c = 2.0; output x, w, c, n;"))
    assert [r.module.values[o].name for o in r.module.outputs] == ["x", "w", "c", "n"]


# --- shadowing -------------------------------------------------------------------


def test_rejects_shadowing_of_index_by_reduction(compile_err):
    e = compile_err(module("def y[i : Input] = sum[i : Input](x[i]); output y;"))
    assert e.codes == [Code.ILLEGAL_SHADOWING]


def test_rejects_shadowing_by_let(compile_err):
    e = compile_err(module("def y[i : Input] = let i = 1.0 in x[i]; output y;"))
    assert Code.ILLEGAL_SHADOWING in e.codes
    e = compile_err(module("def y[] = let x = 1.0 in x; output y;"))
    assert e.codes == [Code.ILLEGAL_SHADOWING]
    e = compile_err(module("def y[] = let a = 1.0 in let a = 2.0 in a; output y;"))
    assert e.codes == [Code.ILLEGAL_SHADOWING]


def test_duplicate_left_side_index(compile_err):
    e = compile_err(module("def y[i : Input, i : Small] = 1.0; output y;"))
    assert e.codes == [Code.ILLEGAL_SHADOWING]


def test_sibling_reductions_may_reuse_a_name(compile_ok):
    compile_ok(module("def y[] = sum[i : Input](x[i]) + sum[i : Input](x[i] * 2.0); output y;"))


def test_index_named_like_tensor_is_a_warning(compile_ok):
    r = compile_ok(module("def y[x : Input] = 1.0; output y;"))
    assert [w.code for w in r.warnings] == [Code.STYLE]


# --- rank and tensor-ness ---------------------------------------------------------


def test_wrong_rank(compile_err):
    e = compile_err(module("def y[i : Input] = x[i, i] + w[i]; output y;"))
    assert e.codes == [Code.WRONG_RANK, Code.WRONG_RANK]


def test_tensor_used_without_brackets(compile_err):
    e = compile_err(module("def y[i : Input] = x; output y;"))
    assert e.codes == [Code.WRONG_RANK]


def test_scalar_read_with_brackets(compile_err):
    e = compile_err(module("def y[] = n[]; output y;"))
    assert e.codes == [Code.NOT_A_TENSOR]


def test_rank_zero_tensor_read(compile_ok):
    r = compile_ok(module("def loss[] = sum[i : Input](x[i]); def twice[] = loss[] * 2.0; output twice;"))
    assert r.module.value_named("twice").type.axes == ()


# --- scalar typing -------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "x[i] + 1",  # Real + Int
        "n + 1.0",
        "x[i] < n",
        "m[i] + m[i]",
        "!x[i]",
        "-m[i]",
        "x[i] && true",
        "if x[i] then 1.0 else 2.0",
        "if m[i] then 1.0 else 2",
        "relu(n)",
        "scalar_max(x[i])",
        "exp(x[i], x[i])",
        "x[i] == m[i]",
        "relu",
    ],
)
def test_type_mismatches(compile_err, body):
    e = compile_err(module(f"def y[i : Input] = {body}; output y;"))
    assert e.codes == [Code.TYPE_MISMATCH], e.render()


def test_annotation_mismatch(compile_err):
    e = compile_err(module("def y[i : Input] : Int = x[i]; const c : Int = 1.0; output y;"))
    assert e.codes == [Code.TYPE_MISMATCH, Code.TYPE_MISMATCH]


def test_index_variable_is_not_a_scalar(compile_err):
    e = compile_err(module("def y[i : Input] = real(i); output y;"))
    assert e.codes == [Code.TYPE_MISMATCH]


@pytest.mark.parametrize(
    "body",
    ["sum[i : Input](m[i])", "max[i : Input](m[i])", "all[i : Input](x[i])", "any[i : Input](n)"],
)
def test_invalid_reduction_bodies(compile_err, body):
    e = compile_err(module(f"def y[] = {body}; output y;"))
    assert e.codes == [Code.INVALID_REDUCTION_BODY]


def test_inferred_types(compile_ok):
    r = compile_ok(
        module(
            """
            def a[i : Input] = x[i] > 0.0 && m[i];
            def b[] = all[i : Input](a[i]);
            def c[i : Input] = n * 2 + 1;
            def d[i : Input] = real(n) * x[i];
            def e[] = sum[i : Input](c[i]) / 2;
            def f[] = if b[] then 1 else 0;
            output a, b, c, d, e, f;
            """
        )
    )
    types = {v.name: v.element_type for v in r.module.defs()}
    assert types == {"a": T.BOOL, "b": T.BOOL, "c": T.INT, "d": T.REAL, "e": T.INT, "f": T.INT}


def test_let_types(compile_ok):
    r = compile_ok(module("def y[i : Input] = let s = x[i] * 2.0 in let ok = s > 1.0 in if ok then s else 0.0; output y;"))
    assert r.module.value_named("y").element_type is T.REAL


# --- constants ----------------------------------------------------------------------


def test_const_may_use_dims_and_functions(compile_ok):
    r = compile_ok(module("const inv : Real = 1.0 / sqrt(real(N)); const twice = K * 2; def y[] = inv; output y, twice;"))
    assert r.module.value_named("twice").type is T.INT


@pytest.mark.parametrize("body", ["x[i]", "n", "sum[i : Input](1.0)"])
def test_const_restrictions(compile_err, body):
    src = module(f"const c = {body}; def y[] = 1.0; output y;")
    src = src.replace("const c", "def z[i : Input] = 1.0; const c")
    e = compile_err(src)
    assert Code.INVALID_CONST in e.codes


# --- dimensions ---------------------------------------------------------------------


def test_dimension_arithmetic_and_supplied_dims(compile_ok):
    src = """
    module M {
        dim B; dim H = 256; dim Heads = 8; dim Hd = H / Heads; dim T = (B + 2) * 3;
        space S = Fin(Hd); space Q = Fin(T);
        input x : Tensor<Real>[S];
        output x;
    }
    """
    r = compile_ok(src, dims={"B": 6})
    m = r.module
    assert m.dim_named("Hd").value == 32
    assert m.dim_named("T").value == 24
    assert m.dim_named("B").value == 6


@pytest.mark.parametrize(
    "decl, fragment",
    [
        ("dim Z = 0;", "must be positive"),
        ("dim Z = 3 - 5;", "must be positive"),
        ("dim Z = 7 / 2;", "not divisible"),
        ("dim Z = 7 / 0;", "division by zero"),
    ],
)
def test_bad_dimensions(compile_err, decl, fragment):
    e = compile_err(f"module M {{ {decl} space S = Fin(Z); input x : Tensor<Real>[S]; output x; }}")
    assert e.codes == [Code.BAD_DIMENSION]
    assert fragment in e.render()


def test_supplied_dimension_validation(compile_err):
    src = "module M { dim N; dim K = 2; space S = Fin(N); input x : Tensor<Real>[S]; output x; }"
    assert compile_err(src, dims={"Q": 3}).codes == [Code.UNKNOWN_NAME]
    assert compile_err(src, dims={"K": 3}).codes == [Code.BAD_DIMENSION]
    assert compile_err(src, dims={"N": 0}).codes == [Code.BAD_DIMENSION]


def test_constraints_checked_when_known(compile_err, compile_ok):
    base = "module M {{ dim A = 8; dim B{}; constraint A == B * 2; space S = Fin(A); input x : Tensor<Real>[S]; output x; }}"
    compile_ok(base.format(" = 4"))
    e = compile_err(base.format(" = 3"))
    assert e.codes == [Code.UNSATISFIED_CONSTRAINT]
    assert "8 == 6 is false" in e.render()
    # Unknown at compile time: deferred to runtime.
    compile_ok(base.format(""))
    compile_ok(base.format(""), dims={"B": 4})
    assert compile_err(base.format(""), dims={"B": 5}).codes == [Code.UNSATISFIED_CONSTRAINT]


def test_multiple_independent_errors_are_all_reported(compile_err):
    e = compile_err(module("def a[o : Output] = x[o]; def b[i : Input] = x[j]; def c[] = 1 + 1.0; output a, b, c;"))
    assert e.codes == [Code.INDEX_SPACE_MISMATCH, Code.UNBOUND_INDEX, Code.TYPE_MISMATCH]


def test_analyze_returns_ir_with_evaluation_order():
    src = module("def b[i : Input] = a[i] * 2.0; def a[i : Input] = x[i]; output b;")
    m, warnings = analyze(src)
    order = [m.values[v].name for v in m.value_order if m.values[v].kind == "def"]
    assert order == ["a", "b"]
    assert warnings == []


def test_diagnostic_rendering_matches_spec_layout():
    src = "module M {\n    dim N = 768;\n    space Input = Fin(N);\n    space Output = Fin(N);\n    input x : Tensor<Real>[Input];\n    def y[o : Output] = x[o];\n    output y;\n}\n"
    try:
        compile_source(src, "model.a")
    except Exception as e:  # noqa: BLE001
        text = str(e)
    assert text.splitlines()[0] == "error[E0204]: index-space mismatch"
    assert "  --> model.a:6:27" in text
    assert "6 |     def y[o : Output] = x[o];" in text
    assert "^ expected index from `Input`, found `Output`" in text
