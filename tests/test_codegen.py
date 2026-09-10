"""PyTorch lowering: einsum contractions, reductions, broadcasting, axis order,
runtime dimension resolution and numerical agreement with direct PyTorch."""

from pathlib import Path

import pytest
import torch

from indexa import compile_file, compile_source
from indexa.diagnostics import Code

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def forward_line(result, name: str) -> str:
    for line in result.python.splitlines():
        if line.strip().startswith(f"{name} = "):
            return line.strip()
    raise AssertionError(f"no statement for {name} in:\n{result.python}")


# --- acceptance tests 1-3: the MLP example -----------------------------------------


def test_mlp_example_compiles_and_matches_reference():
    result = compile_file(EXAMPLES / "mlp.a")
    assert forward_line(result, "preactivation") == 'preactivation = torch.einsum("ab,bc->ac", x, w1) + b1'
    assert forward_line(result, "hidden") == "hidden = torch.relu(preactivation)"
    assert forward_line(result, "maximum") == "maximum = torch.amax(logits, dim=1)"
    assert forward_line(result, "exponent") == "exponent = torch.exp(logits - maximum[:, None])"
    assert forward_line(result, "denominator") == "denominator = torch.sum(exponent, dim=1)"
    assert forward_line(result, "probability") == "probability = exponent / denominator[:, None]"

    mod = result.load()
    assert mod.INPUTS == ("x",) and mod.PARAMS == ("w1", "b1", "w2", "b2") and mod.OUTPUTS == ("probability",)
    assert mod.DIMS == {"B": None, "I": None, "H": 256, "O": 10} and mod.FREE_DIMS == ("B", "I")

    torch.manual_seed(0)
    B, I = 5, 7
    x, w1, b1, w2, b2 = torch.randn(B, I), torch.randn(I, 256), torch.randn(256), torch.randn(256, 10), torch.randn(10)
    got = mod.forward(x, w1, b1, w2, b2)
    ref = torch.softmax(torch.relu(x @ w1 + b1) @ w2 + b2, dim=1)
    assert got.shape == (B, 10)
    assert torch.allclose(got, ref, atol=1e-6)
    assert mod.resolve_dims(x, w1, b1, w2, b2) == {"B": 5, "I": 7, "H": 256, "O": 10}


def test_mlp_runtime_shape_assertions():
    mod = compile_file(EXAMPLES / "mlp.a").load()
    x = torch.randn(2, 3)
    good = (torch.randn(3, 256), torch.randn(256), torch.randn(256, 10), torch.randn(10))
    with pytest.raises(mod.ShapeError, match="dimension `I` is 3 but `w1` has size 4"):
        mod.forward(x, torch.randn(4, 256), *good[1:])
    with pytest.raises(mod.ShapeError, match=r"param `b2` has shape \(11,\)"):
        mod.forward(x, *good[:3], torch.randn(11))
    with pytest.raises(mod.ShapeError, match="dimension `B` is 8 but `x` has size 2"):
        mod.forward(x, *good, B=8)
    assert mod.forward(x, *good, B=2, I=3).shape == (2, 10)


def test_supplied_dims_are_fixed_in_generated_code():
    result = compile_file(EXAMPLES / "mlp.a", dims={"B": 4, "I": 3})
    mod = result.load()
    assert mod.FREE_DIMS == () and mod.DIMS["B"] == 4
    assert "B=None" not in result.python
    with pytest.raises(mod.ShapeError):
        mod.forward(torch.randn(5, 3), torch.randn(3, 256), torch.randn(256), torch.randn(256, 10), torch.randn(10))


def test_attention_example_matches_reference():
    mod = compile_file(EXAMPLES / "attention.a").load()
    torch.manual_seed(0)
    B, S, D, Dh = 2, 5, 64, 16
    x = torch.randn(B, S, D)
    allowed = torch.tril(torch.ones(S, S, dtype=torch.bool))
    wq, wk, wv = (torch.randn(D, Dh) for _ in range(3))
    out, att = mod.forward(x, allowed, x, wq, wk, wv)
    q, k, v = x @ wq, x @ wk, x @ wv
    score = torch.where(allowed, (q @ k.transpose(1, 2)) / Dh**0.5, torch.tensor(-1e30))
    ref_att = torch.softmax(score, dim=-1)
    assert torch.allclose(att, ref_att, atol=1e-5)
    assert torch.allclose(out, ref_att @ v, atol=1e-4)
    assert mod.resolve_dims(x, allowed, x, wq, wk, wv)["Dh"] == 16


# --- acceptance tests 7-10 -------------------------------------------------------------

BASE = """
module T {
    dim N; dim M; dim K = 3;
    space I = Fin(N); space J = Fin(M); space L = Fin(K);
    input x : Tensor<Real>[I, J];
    input y : Tensor<Real>[J, I];
    input bias : Tensor<Real>[J];
    input sq : Tensor<Real>[I, I];
    param w : Tensor<Real>[J, L];
    %s
}
"""


def run(body: str, **tensors):
    result = compile_source(BASE % body, "t.a")
    mod = result.load()
    out = mod.forward(**tensors)
    return result, (out if isinstance(out, tuple) else (out,))


@pytest.fixture
def data():
    torch.manual_seed(3)
    N, M, K = 4, 6, 3
    return dict(x=torch.randn(N, M), y=torch.randn(M, N), bias=torch.randn(M), sq=torch.randn(N, N), w=torch.randn(M, K))


def test_broadcasting_from_omitted_free_indices(data):
    result, (z, rep) = run(
        "def z[i : I, j : J] = x[i, j] + bias[j];"
        "def rep[i : I, j : J] = bias[j];"
        "output z, rep;",
        **data,
    )
    assert forward_line(result, "z") == "z = x + bias"
    assert "expand(size_I, size_J)" in forward_line(result, "rep")
    assert torch.equal(z, data["x"] + data["bias"])
    assert rep.shape == (4, 6) and torch.equal(rep, data["bias"].expand(4, 6))


def test_leading_broadcast_uses_none_indexing(data):
    result, (z,) = run("def rowsum[i : I] = sum[j : J](x[i, j]); def z[i : I, j : J] = x[i, j] / rowsum[i]; output z;", **data)
    assert forward_line(result, "z") == "z = x / rowsum[:, None]"
    assert torch.allclose(z, data["x"] / data["x"].sum(1, keepdim=True))


def test_matrix_contraction_lowers_to_einsum(data):
    result, (p,) = run("def p[i : I, l : L] = sum[j : J](x[i, j] * w[j, l]); output p;", **data)
    assert forward_line(result, "p") == 'p = torch.einsum("ab,bc->ac", x, w)'
    assert torch.allclose(p, data["x"] @ data["w"], atol=1e-5)


def test_einsum_respects_output_axis_order(data):
    result, (p,) = run("def p[l : L, i : I] = sum[j : J](x[i, j] * w[j, l]); output p;", **data)
    assert forward_line(result, "p") == 'p = torch.einsum("ab,bc->ca", x, w)'
    assert torch.allclose(p, (data["x"] @ data["w"]).T, atol=1e-5)


def test_three_operand_contraction_and_trace(data):
    result, (t, tr) = run(
        "def t[l : L] = sum[i : I, j : J](x[i, j] * y[j, i] * w[j, l]);"
        "def tr[] = sum[i : I](sq[i, i]);"
        "output t, tr;",
        **data,
    )
    assert forward_line(result, "t") == 't = torch.einsum("ab,ba,bc->c", x, y, w)'
    assert forward_line(result, "tr") == 'tr = torch.sum(torch.einsum("aa->a", sq), dim=0)'
    assert torch.allclose(t, torch.einsum("ij,ji,jl->l", data["x"], data["y"], data["w"]), atol=1e-4)
    assert torch.allclose(tr, data["sq"].trace())


def test_sum_and_max_reductions_over_one_or_more_axes(data):
    result, (s1, s2, m1, m2, mn) = run(
        "def s1[i : I] = sum[j : J](x[i, j]);"
        "def s2[] = sum[i : I, j : J](x[i, j]);"
        "def m1[j : J] = max[i : I](x[i, j]);"
        "def m2[] = max[i : I, j : J](x[i, j]);"
        "def mn[i : I] = min[j : J](x[i, j] * 2.0);"
        "output s1, s2, m1, m2, mn;",
        **data,
    )
    x = data["x"]
    assert forward_line(result, "s1") == "s1 = torch.sum(x, dim=1)"
    assert forward_line(result, "s2") == "s2 = torch.sum(x, dim=(0, 1))"
    assert forward_line(result, "m1") == "m1 = torch.amax(x.permute(1, 0), dim=1)"
    assert forward_line(result, "m2") == "m2 = torch.amax(x, dim=(0, 1))"
    assert torch.allclose(s1, x.sum(1)) and torch.allclose(s2, x.sum())
    assert torch.equal(m1, x.amax(0)) and torch.equal(m2, x.max())
    assert torch.allclose(mn, (x * 2).amin(1))


def test_output_axis_order_is_preserved(data):
    result, (t, u) = run(
        "def t[j : J, i : I] = x[i, j];"
        "def u[l : L, i : I, j : J] = x[i, j] * w[j, l];"
        "output t, u;",
        **data,
    )
    assert forward_line(result, "t") == "t = x.permute(1, 0)"
    assert t.shape == (6, 4) and torch.equal(t, data["x"].T)
    assert u.shape == (3, 4, 6)
    assert torch.allclose(u, (data["x"][:, :, None] * data["w"][None, :, :]).permute(2, 0, 1))


def test_prod_all_any_and_bool_ops(data):
    mask = data["x"] > 0
    result = compile_source(
        """
        module B {
            dim N; dim M;
            space I = Fin(N); space J = Fin(M);
            input m : Tensor<Bool>[I, J];
            input v : Tensor<Real>[I, J];
            input flag : Bool;
            def p[i : I] = prod[j : J](v[i, j]);
            def pall[] = prod[i : I, j : J](v[i, j]);
            def a[i : I] = all[j : J](m[i, j] || flag);
            def n[] = any[i : I, j : J](m[i, j] && !flag);
            def cnt[i : I, j : J] = if m[i, j] then 1 else 0;
            output p, pall, a, n, cnt;
        }
        """
    )
    mod = result.load()
    p, pall, a, n, cnt = mod.forward(mask, data["x"], False)
    assert torch.allclose(p, data["x"].prod(1)) and torch.allclose(pall, data["x"].prod())
    assert torch.equal(a, mask.all(1)) and bool(n) == bool(mask.any())
    assert torch.equal(cnt, mask.long())
    assert cnt.dtype == torch.int64


def test_let_if_and_scalar_library(data):
    result, (z,) = run(
        """
        def z[i : I, l : L] =
            let dot = sum[j : J](x[i, j] * w[j, l])
            in let s = sigmoid(dot) - 0.5
            in if s > 0.0 then sqrt(s) ** 2.0 else scalar_max(s, -0.25) + abs(s) * 0.0;
        output z;
        """,
        **data,
    )
    assert forward_line(result, "dot") == 'dot = torch.einsum("ab,bc->ac", x, w)'
    s = torch.sigmoid(data["x"] @ data["w"]) - 0.5
    ref = torch.where(s > 0, torch.sqrt(s) ** 2, torch.maximum(s, torch.tensor(-0.25)))
    assert torch.allclose(z, ref, atol=1e-6)


def test_reduction_over_index_absent_from_body_replicates(data):
    result, (r,) = run("def r[i : I] = sum[j : J](sq[i, i]); output r;", **data)
    assert torch.allclose(r, data["sq"].diagonal() * 6)


def test_int_arithmetic_and_dims_as_scalars():
    result = compile_source(
        """
        module I {
            dim N; dim K = 4;
            space S = Fin(N);
            input c : Tensor<Int>[S];
            input n : Int;
            const scale : Real = 1.0 / real(K);
            def d[i : S] = (c[i] * 3 + n) / 2;
            def r[i : S] = real(c[i]) * scale + real(N);
            def cmp[i : S] = c[i] >= 2 && c[i] != n;
            def neg[i : S] = -c[i] ** 2;
            output d, r, cmp, neg, scale;
        }
        """
    )
    mod = result.load()
    c = torch.tensor([0, 1, 2, 5])
    d, r, cmp, neg, scale = mod.forward(c, 3)
    assert torch.equal(d, torch.div(c * 3 + 3, 2, rounding_mode="floor"))
    assert torch.allclose(r, c.float() / 4 + 4)
    assert torch.equal(cmp, (c >= 2) & (c != 3))
    assert torch.equal(neg, (-c) ** 2)  # unary minus binds tighter than ** (spec section 15)
    assert scale.item() == 0.25 and isinstance(scale, torch.Tensor)


def test_runtime_dimension_derivation_and_constraints():
    result = compile_source(
        """
        module R {
            dim N; dim Half = N / 2; dim Q;
            constraint Q <= N;
            space S = Fin(N); space H = Fin(Half); space QS = Fin(Q);
            input x : Tensor<Real>[S];
            input q : Tensor<Real>[QS];
            def y[h : H] = 1.0;
            output y;
        }
        """
    )
    mod = result.load()
    assert mod.forward(torch.zeros(6), torch.zeros(2)).shape == (3,)
    with pytest.raises(mod.ShapeError, match="not divisible"):
        mod.forward(torch.zeros(5), torch.zeros(2))
    with pytest.raises(mod.ShapeError, match="constraint `Q <= N` is violated: 7 <= 6"):
        mod.forward(torch.zeros(6), torch.zeros(7))


def test_dimension_without_source_needs_keyword():
    mod = compile_source(
        "module D { dim N; space S = Fin(N); input s : Real; def y[i : S] = s; output y; }"
    ).load()
    with pytest.raises(mod.ShapeError, match="cannot be inferred"):
        mod.forward(1.0)
    assert torch.equal(mod.forward(1.5, N=3), torch.full((3,), 1.5))


def test_python_keyword_names_are_mangled():
    mod = compile_source(
        "module K { dim N; space S = Fin(N); input lambda : Tensor<Real>[S]; def torch[i : S] = lambda[i] * 2.0; output torch; }"
    ).load()
    assert torch.equal(mod.forward(torch.ones(2)), torch.full((2,), 2.0))


def test_unused_definitions_are_pruned():
    result = compile_source(
        "module P { dim N; space S = Fin(N); input x : Tensor<Real>[S]; def unused[i : S] = x[i]; def y[i : S] = x[i]; output y; }"
    )
    assert "unused" not in result.python.split("def forward")[1]


def test_scalar_inputs_are_checked():
    mod = compile_source("module S { input s : Real; def y[] = s * 2.0; output y; }").load()
    assert mod.forward(2.0).item() == 4.0
    with pytest.raises(mod.ShapeError, match="input `s` has shape"):
        mod.forward(torch.ones(2))


def test_backend_unsupported_code_exists():
    # Every legal v0.1 construct lowers; the code stays reserved for future backends.
    assert Code.UNSUPPORTED.title == "unsupported construct in the current backend"
