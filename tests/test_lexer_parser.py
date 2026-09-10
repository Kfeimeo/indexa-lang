import pytest

from indexa import ast
from indexa.diagnostics import Code, CompileError
from indexa.lexer import Tok, tokenize
from indexa.parser import parse


def kinds(src):
    return [(t.kind, t.text) for t in tokenize(src) if t.kind is not Tok.EOF]


def test_tokens_and_comments():
    src = "def y[i : I] = x[i] ** 2.5e-1 // comment\n/* block\n comment */ + 3;"
    toks = kinds(src)
    assert toks[0] == (Tok.KEYWORD, "def")
    assert (Tok.PUNCT, "**") in toks
    assert (Tok.REAL, "2.5e-1") in toks
    assert (Tok.INT, "3") in toks
    assert not any(text.startswith("//") or text.startswith("/*") for _, text in toks)


def test_longest_match_punctuation():
    assert [t for _, t in kinds("<= >= == != && || ** < >")] == ["<=", ">=", "==", "!=", "&&", "||", "**", "<", ">"]


@pytest.mark.parametrize("src", ["1.", "1e", "1x", "/* open", "@", "xé"])
def test_lexer_errors(src):
    with pytest.raises(CompileError) as e:
        tokenize(src)
    assert e.value.has(Code.SYNTAX)


def test_future_keywords_are_rejected():
    with pytest.raises(CompileError) as e:
        parse("module M { input fn : Real; output fn; }")
    assert "reserved for a future version" in e.value.render()


def test_precedence():
    m = parse("module M { def y[] = -x ** 2 * 3 + 1 < 4 && !a || b; output y; }")
    body = m.items[0].body
    # || at the top
    assert isinstance(body, ast.Binary) and body.op == "||"
    land = body.left
    assert land.op == "&&"
    lt = land.left
    assert lt.op == "<"
    add = lt.left
    assert add.op == "+"
    mul = add.left
    assert mul.op == "*"
    power = mul.left
    assert power.op == "**"
    # unary binds tighter than ** (spec section 15)
    assert isinstance(power.left, ast.Unary) and power.left.op == "-"
    assert isinstance(land.right, ast.Unary) and land.right.op == "!"


def test_power_is_right_associative():
    body = parse("module M { def y[] = 2.0 ** 3.0 ** 4.0; output y; }").items[0].body
    assert body.op == "**" and body.right.op == "**"


def test_relational_cannot_chain():
    with pytest.raises(CompileError) as e:
        parse("module M { def y[] = 1 < 2 < 3; output y; }")
    assert "chained" in e.value.render()


def test_let_if_and_reduction_shapes():
    m = parse(
        """
        module M {
            def y[b : B] = let d = sum[i : I, j : J](x[b, i, j]) in if d > 0.0 then d else -d;
            output y;
        }
        """
    )
    d = m.items[0]
    assert isinstance(d, ast.DefDecl)
    assert isinstance(d.body, ast.Let)
    assert isinstance(d.body.value, ast.Reduce) and [b.name.name for b in d.body.value.binders] == ["i", "j"]
    assert isinstance(d.body.body, ast.If)


def test_declarations_parse():
    m = parse(
        """
        module M {
            dim N; dim K = (N + 2) * 3 / 2;
            space S = Fin(K);
            constraint K >= N;
            input a : Tensor<Int>[S, S]; input flag : Bool;
            param p : Tensor<Real>[];
            const c : Real = 1.0e-6; const d = 3;
            def t[] = p[];
            output t, c;
        }
        """
    )
    types = [type(i).__name__ for i in m.items]
    assert types == ["DimDecl", "DimDecl", "SpaceDecl", "ConstraintDecl", "InputDecl", "InputDecl",
                     "ParamDecl", "ConstDecl", "ConstDecl", "DefDecl", "OutputDecl"]
    assert m.items[1].value.op == "/"
    assert m.items[6].type.axes == []


@pytest.mark.parametrize(
    "src, fragment",
    [
        ("module M { def y[i] = 1; output y; }", "index binders are written"),
        ("module M { def y[] = sum[](1); output y; }", "at least one index"),
        ("module M { input x : Tensor<Real>; output x; }", "expected `[`"),
        ("module M { def y[] = 1 output y; }", "expected `;`"),
        ("module { }", "expected module name"),
        ("module M { dim N = 1.5; }", "dimensions are integers"),
        ("module M { def y[] = (1; output y; }", "expected `)`"),
    ],
)
def test_parse_errors_have_helpful_messages(src, fragment):
    with pytest.raises(CompileError) as e:
        parse(src)
    assert fragment in e.value.render(), e.value.render()
