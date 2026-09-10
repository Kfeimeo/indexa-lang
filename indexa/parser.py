"""Recursive-descent parser for the A language.

Follows the EBNF in spec sections 7-21 directly.  Operator precedence
(section 15), highest first::

    1. parentheses, reads, calls, reductions
    2. unary + - !
    3. **  (right-associative, binds looser than unary)
    4. * /
    5. + -
    6. < <= > >=   (non-associative)
    7. == !=
    8. &&
    9. ||
    10. if/then/else and let/in
"""

from __future__ import annotations

from . import ast
from .diagnostics import Code, CompileError, Diagnostic, Span
from .lexer import REDUCTION_OPERATORS, Tok, Token, tokenize

SCALAR_TYPES = ("Bool", "Int", "Real", "Complex")
COMPARISON_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")


class Parser:
    def __init__(self, tokens: list[Token], source: str):
        self.tokens = tokens
        self.source = source
        self.i = 0

    # -- token helpers ---------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def _prev(self) -> Token:
        return self.tokens[self.i - 1]

    def _error(self, message: str, span: Span | None = None) -> CompileError:
        return CompileError([Diagnostic(Code.SYNTAX, message, span or self.tok.span)], self.source)

    def _describe(self, tok: Token) -> str:
        if tok.kind is Tok.EOF:
            return "end of input"
        return f"`{tok.text}`"

    def _next(self) -> Token:
        t = self.tok
        if t.kind is not Tok.EOF:
            self.i += 1
        return t

    def _at_punct(self, text: str) -> bool:
        return self.tok.is_punct(text)

    def _at_keyword(self, word: str) -> bool:
        return self.tok.is_keyword(word)

    def _accept_punct(self, text: str) -> Token | None:
        if self._at_punct(text):
            return self._next()
        return None

    def _expect_punct(self, text: str, context: str | None = None) -> Token:
        if not self._at_punct(text):
            where = f" {context}" if context else ""
            raise self._error(f"expected `{text}`{where}, found {self._describe(self.tok)}")
        return self._next()

    def _expect_keyword(self, word: str) -> Token:
        if not self._at_keyword(word):
            raise self._error(f"expected `{word}`, found {self._describe(self.tok)}")
        return self._next()

    def _expect_ident(self, what: str = "identifier") -> ast.Name:
        t = self.tok
        if t.kind is not Tok.IDENT:
            if t.kind is Tok.KEYWORD:
                raise self._error(f"expected {what}, found reserved word `{t.text}`")
            raise self._error(f"expected {what}, found {self._describe(t)}")
        self._next()
        return ast.Name(t.span, t.text)

    # -- module ------------------------------------------------------------
    def parse_module(self) -> ast.Module:
        start = self._expect_keyword("module")
        name = self._expect_ident("module name")
        self._expect_punct("{", "after module name")
        items: list[ast.Item] = []
        while not self._at_punct("}"):
            if self.tok.kind is Tok.EOF:
                raise self._error("unexpected end of input inside module; expected `}`")
            items.append(self._item())
        end = self._next()
        if self.tok.kind is not Tok.EOF:
            raise self._error(f"expected end of input after module, found {self._describe(self.tok)}")
        return ast.Module(start.span.to(end.span), name, items)

    def _item(self) -> ast.Item:
        t = self.tok
        if t.kind is not Tok.KEYWORD:
            raise self._error(
                f"expected a declaration (`dim`, `space`, `constraint`, `input`, `param`, "
                f"`const`, `def` or `output`), found {self._describe(t)}"
            )
        handlers = {
            "dim": self._dim_decl,
            "space": self._space_decl,
            "constraint": self._constraint_decl,
            "input": self._input_decl,
            "param": self._param_decl,
            "const": self._const_decl,
            "def": self._def_decl,
            "output": self._output_decl,
        }
        handler = handlers.get(t.text)
        if handler is None:
            raise self._error(f"`{t.text}` cannot start a declaration")
        return handler()

    def _semicolon(self, start: Token) -> Span:
        end = self._expect_punct(";", "to end the declaration")
        return start.span.to(end.span)

    def _dim_decl(self) -> ast.DimDecl:
        start = self._next()
        name = self._expect_ident("dimension name")
        value = None
        if self._accept_punct("="):
            value = self._dim_expr()
        return ast.DimDecl(self._semicolon(start), name, value)

    def _space_decl(self) -> ast.SpaceDecl:
        start = self._next()
        name = self._expect_ident("space name")
        self._expect_punct("=", "after space name")
        self._expect_keyword("Fin")
        self._expect_punct("(", "after `Fin`")
        size = self._dim_expr()
        self._expect_punct(")", "after `Fin(` dimension")
        return ast.SpaceDecl(self._semicolon(start), name, size)

    def _constraint_decl(self) -> ast.ConstraintDecl:
        start = self._next()
        left = self._dim_expr()
        if self.tok.kind is Tok.PUNCT and self.tok.text in COMPARISON_OPERATORS:
            op = self._next().text
        else:
            raise self._error(f"expected a comparison operator in constraint, found {self._describe(self.tok)}")
        right = self._dim_expr()
        return ast.ConstraintDecl(self._semicolon(start), op, left, right)

    def _input_decl(self) -> ast.InputDecl:
        start = self._next()
        name = self._expect_ident("input name")
        self._expect_punct(":", "after input name")
        ty = self._type()
        return ast.InputDecl(self._semicolon(start), name, ty)

    def _param_decl(self) -> ast.ParamDecl:
        start = self._next()
        name = self._expect_ident("parameter name")
        self._expect_punct(":", "after parameter name")
        ty = self._type()
        return ast.ParamDecl(self._semicolon(start), name, ty)

    def _const_decl(self) -> ast.ConstDecl:
        start = self._next()
        name = self._expect_ident("constant name")
        ty = None
        if self._accept_punct(":"):
            ty = self._scalar_type()
        self._expect_punct("=", "in constant declaration")
        value = self._scalar_expr()
        return ast.ConstDecl(self._semicolon(start), name, ty, value)

    def _def_decl(self) -> ast.DefDecl:
        start = self._next()
        name = self._expect_ident("tensor name")
        self._expect_punct("[", "after tensor name in `def`")
        binders = self._index_binder_list("]")
        self._expect_punct("]", "after index binders")
        ty = None
        if self._accept_punct(":"):
            ty = self._scalar_type()
        self._expect_punct("=", "in tensor definition")
        body = self._scalar_expr()
        return ast.DefDecl(self._semicolon(start), name, binders, ty, body)

    def _output_decl(self) -> ast.OutputDecl:
        start = self._next()
        names = [self._expect_ident("output name")]
        while self._accept_punct(","):
            names.append(self._expect_ident("output name"))
        return ast.OutputDecl(self._semicolon(start), names)

    def _index_binder_list(self, closer: str) -> list[ast.IndexBinder]:
        binders: list[ast.IndexBinder] = []
        if self._at_punct(closer):
            return binders
        while True:
            name = self._expect_ident("index name")
            self._expect_punct(":", "after index name (index binders are written `i : Space`)")
            space = self._expect_ident("index-space name")
            binders.append(ast.IndexBinder(name.span.to(space.span), name, space))
            if not self._accept_punct(","):
                return binders

    # -- dimension expressions --------------------------------------------
    def _dim_expr(self) -> ast.DimNode:
        left = self._dim_mul()
        while self.tok.kind is Tok.PUNCT and self.tok.text in ("+", "-"):
            op = self._next().text
            right = self._dim_mul()
            left = ast.DimBinary(left.span.to(right.span), op, left, right)
        return left

    def _dim_mul(self) -> ast.DimNode:
        left = self._dim_primary()
        while self.tok.kind is Tok.PUNCT and self.tok.text in ("*", "/"):
            op = self._next().text
            right = self._dim_primary()
            left = ast.DimBinary(left.span.to(right.span), op, left, right)
        return left

    def _dim_primary(self) -> ast.DimNode:
        t = self.tok
        if t.kind is Tok.INT:
            self._next()
            return ast.DimLit(t.span, int(t.text))
        if t.kind is Tok.IDENT:
            self._next()
            return ast.DimName(t.span, t.text)
        if t.is_punct("("):
            self._next()
            inner = self._dim_expr()
            self._expect_punct(")", "to close the parenthesized dimension expression")
            return inner
        if t.kind is Tok.REAL:
            raise self._error("dimensions are integers; real literals are not allowed here")
        raise self._error(f"expected a dimension expression, found {self._describe(t)}")

    # -- types -------------------------------------------------------------
    def _scalar_type(self) -> ast.ScalarTypeNode:
        t = self.tok
        if t.kind is Tok.KEYWORD and t.text in SCALAR_TYPES:
            self._next()
            return ast.ScalarTypeNode(t.span, t.text)
        raise self._error(f"expected a scalar type (`Bool`, `Int`, `Real` or `Complex`), found {self._describe(t)}")

    def _type(self) -> ast.TypeNode:
        t = self.tok
        if t.is_keyword("Tensor"):
            self._next()
            self._expect_punct("<", "after `Tensor`")
            element = self._scalar_type()
            self._expect_punct(">", "after tensor element type")
            self._expect_punct("[", "to start the tensor axis list")
            axes: list[ast.Name] = []
            if not self._at_punct("]"):
                while True:
                    axes.append(self._expect_ident("index-space name"))
                    if not self._accept_punct(","):
                        break
            end = self._expect_punct("]", "to close the tensor axis list")
            return ast.TensorTypeNode(t.span.to(end.span), element, axes)
        return self._scalar_type()

    # -- scalar expressions -----------------------------------------------
    def _scalar_expr(self) -> ast.Expr:
        if self._at_keyword("let"):
            start = self._next()
            name = self._expect_ident("local name")
            self._expect_punct("=", "after `let` name")
            value = self._scalar_expr()
            self._expect_keyword("in")
            body = self._scalar_expr()
            return ast.Let(start.span.to(body.span), name, value, body)
        if self._at_keyword("if"):
            start = self._next()
            cond = self._scalar_expr()
            self._expect_keyword("then")
            then = self._scalar_expr()
            self._expect_keyword("else")
            otherwise = self._scalar_expr()
            return ast.If(start.span.to(otherwise.span), cond, then, otherwise)
        return self._or()

    def _or(self) -> ast.Expr:
        left = self._and()
        while self._at_punct("||"):
            self._next()
            right = self._and()
            left = ast.Binary(left.span.to(right.span), "||", left, right)
        return left

    def _and(self) -> ast.Expr:
        left = self._equality()
        while self._at_punct("&&"):
            self._next()
            right = self._equality()
            left = ast.Binary(left.span.to(right.span), "&&", left, right)
        return left

    def _equality(self) -> ast.Expr:
        left = self._relational()
        while self.tok.kind is Tok.PUNCT and self.tok.text in ("==", "!="):
            op = self._next().text
            right = self._relational()
            left = ast.Binary(left.span.to(right.span), op, left, right)
        return left

    def _relational(self) -> ast.Expr:
        left = self._additive()
        if self.tok.kind is Tok.PUNCT and self.tok.text in ("<", "<=", ">", ">="):
            op = self._next().text
            right = self._additive()
            left = ast.Binary(left.span.to(right.span), op, left, right)
            if self.tok.kind is Tok.PUNCT and self.tok.text in ("<", "<=", ">", ">="):
                raise self._error("relational operators cannot be chained; parenthesize the comparison")
        return left

    def _additive(self) -> ast.Expr:
        left = self._multiplicative()
        while self.tok.kind is Tok.PUNCT and self.tok.text in ("+", "-"):
            op = self._next().text
            right = self._multiplicative()
            left = ast.Binary(left.span.to(right.span), op, left, right)
        return left

    def _multiplicative(self) -> ast.Expr:
        left = self._power()
        while self.tok.kind is Tok.PUNCT and self.tok.text in ("*", "/"):
            op = self._next().text
            right = self._power()
            left = ast.Binary(left.span.to(right.span), op, left, right)
        return left

    def _power(self) -> ast.Expr:
        base = self._unary()
        if self._at_punct("**"):
            self._next()
            exponent = self._power()  # right-associative
            return ast.Binary(base.span.to(exponent.span), "**", base, exponent)
        return base

    def _unary(self) -> ast.Expr:
        t = self.tok
        if t.kind is Tok.PUNCT and t.text in ("+", "-", "!"):
            self._next()
            operand = self._unary()
            return ast.Unary(t.span.to(operand.span), t.text, operand)
        return self._primary()

    def _primary(self) -> ast.Expr:
        t = self.tok
        if t.kind is Tok.INT:
            self._next()
            return ast.IntLit(t.span, t.text)
        if t.kind is Tok.REAL:
            self._next()
            return ast.RealLit(t.span, t.text)
        if t.is_keyword("true") or t.is_keyword("false"):
            self._next()
            return ast.BoolLit(t.span, t.text == "true")
        if t.kind is Tok.KEYWORD and t.text in REDUCTION_OPERATORS:
            return self._reduction()
        if t.is_punct("("):
            self._next()
            inner = self._scalar_expr()
            end = self._expect_punct(")", "to close the parenthesized expression")
            inner.span = t.span.to(end.span)
            return inner
        if t.kind is Tok.IDENT:
            self._next()
            name = ast.Name(t.span, t.text)
            if self._at_punct("["):
                self._next()
                indices: list[ast.Name] = []
                if not self._at_punct("]"):
                    while True:
                        indices.append(self._expect_ident("index name"))
                        if not self._accept_punct(","):
                            break
                end = self._expect_punct("]", "to close the tensor read")
                return ast.TensorRead(t.span.to(end.span), name, indices)
            if self._at_punct("("):
                self._next()
                args: list[ast.Expr] = []
                if not self._at_punct(")"):
                    while True:
                        args.append(self._scalar_expr())
                        if not self._accept_punct(","):
                            break
                end = self._expect_punct(")", "to close the argument list")
                return ast.Call(t.span.to(end.span), name, args)
            return ast.Ident(t.span, t.text)
        if t.kind is Tok.KEYWORD:
            raise self._error(f"unexpected reserved word `{t.text}` in expression")
        raise self._error(f"expected an expression, found {self._describe(t)}")

    def _reduction(self) -> ast.Reduce:
        start = self._next()
        self._expect_punct("[", f"after `{start.text}` (reductions are written `{start.text}[i : Space](body)`)")
        binders = self._index_binder_list("]")
        if not binders:
            raise self._error("a reduction must bind at least one index", start.span.to(self.tok.span))
        self._expect_punct("]", "after reduction binders")
        self._expect_punct("(", "to start the reduction body")
        body = self._scalar_expr()
        end = self._expect_punct(")", "to close the reduction body")
        return ast.Reduce(start.span.to(end.span), start.text, binders, body)


def parse(source: str, filename: str = "<input>") -> ast.Module:
    """Parse a complete A module from source text."""
    return Parser(tokenize(source, filename), source).parse_module()
