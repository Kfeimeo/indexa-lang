"""Lexer for the A language (spec sections 4-6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .diagnostics import Code, CompileError, Diagnostic, Span


class Tok(Enum):
    IDENT = auto()
    INT = auto()
    REAL = auto()
    KEYWORD = auto()
    PUNCT = auto()
    EOF = auto()


#: Words reserved in v0.1 (spec section 5).
KEYWORDS = frozenset(
    """
    module dim space constraint input param const def output let in
    if then else sum prod max min all any true false
    Bool Int Real Complex Tensor Fin
    """.split()
)

#: Words reserved for future versions.  They are rejected as identifiers so that
#: programs written today keep working when the words gain meaning.
FUTURE_KEYWORDS = frozenset("fn return indexmap get default import as grad wrt scan while Dim Space".split())

REDUCTION_OPERATORS = ("sum", "prod", "max", "min", "all", "any")

# Longest punctuators first so that ``**`` wins over ``*`` and ``<=`` over ``<``.
_PUNCT = sorted(
    ["{", "}", "(", ")", "[", "]", "<", ">", ",", ";", ":", "=", "+", "-", "*", "/", "**",
     "==", "!=", "<=", ">=", "&&", "||", "!"],
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class Token:
    kind: Tok
    text: str
    span: Span

    def is_keyword(self, word: str) -> bool:
        return self.kind is Tok.KEYWORD and self.text == word

    def is_punct(self, text: str) -> bool:
        return self.kind is Tok.PUNCT and self.text == text

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, {self.span})"


class Lexer:
    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1

    # -- helpers ---------------------------------------------------------
    def _peek(self, offset: int = 0) -> str:
        i = self.pos + offset
        return self.source[i] if i < len(self.source) else ""

    def _advance(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pos >= len(self.source):
                return
            if self.source[self.pos] == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1

    def _span_from(self, start_pos: int, start_line: int, start_col: int) -> Span:
        return Span(self.filename, start_line, start_col, self.line, self.col, start_pos, self.pos)

    def _error(self, message: str, span: Span) -> CompileError:
        return CompileError([Diagnostic(Code.SYNTAX, message, span)], self.source)

    # -- main loop -------------------------------------------------------
    def tokens(self) -> list[Token]:
        out: list[Token] = []
        while True:
            self._skip_trivia()
            start = (self.pos, self.line, self.col)
            ch = self._peek()
            if ch == "":
                out.append(Token(Tok.EOF, "", self._span_from(*start)))
                return out
            if (ch.isalpha() or ch == "_") and ch.isascii():
                out.append(self._identifier(start))
            elif not ch.isascii():
                self._advance()
                raise self._error(f"unexpected non-ASCII character {ch!r}", self._span_from(*start))
            elif ch.isdigit():
                out.append(self._number(start))
            else:
                out.append(self._punct(start))

    def _skip_trivia(self) -> None:
        while True:
            ch = self._peek()
            if ch and ch in " \t\r\n\f\v":
                self._advance()
            elif ch == "/" and self._peek(1) == "/":
                while self._peek() not in ("", "\n"):
                    self._advance()
            elif ch == "/" and self._peek(1) == "*":
                start = (self.pos, self.line, self.col)
                self._advance(2)
                while not (self._peek() == "*" and self._peek(1) == "/"):
                    if self._peek() == "":
                        raise self._error("unterminated block comment", self._span_from(*start))
                    self._advance()
                self._advance(2)
            else:
                return

    def _identifier(self, start: tuple[int, int, int]) -> Token:
        if not self._peek().isascii():
            self._advance()
            raise self._error("identifiers must be ASCII in v0.1", self._span_from(*start))
        while True:
            ch = self._peek()
            if ch and ch.isascii() and (ch.isalnum() or ch == "_"):
                self._advance()
            else:
                break
        text = self.source[start[0] : self.pos]
        span = self._span_from(*start)
        if text in KEYWORDS:
            return Token(Tok.KEYWORD, text, span)
        if text in FUTURE_KEYWORDS:
            raise self._error(f"`{text}` is reserved for a future version of the language", span)
        return Token(Tok.IDENT, text, span)

    def _number(self, start: tuple[int, int, int]) -> Token:
        def digits() -> None:
            while self._peek().isdigit():
                self._advance()

        digits()
        is_real = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_real = True
            self._advance()
            digits()
        elif self._peek() == ".":
            self._advance()
            raise self._error("a real literal needs digits after the decimal point", self._span_from(*start))
        if self._peek() in ("e", "E"):
            save = (self.pos, self.line, self.col)
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            if self._peek().isdigit():
                is_real = True
                digits()
            else:
                # Not an exponent after all (e.g. `2e` followed by junk): report clearly.
                self.pos, self.line, self.col = save
                raise self._error("malformed exponent in numeric literal", self._span_from(*start))
        nxt = self._peek()
        if nxt and (nxt.isalpha() or nxt == "_"):
            self._advance()
            raise self._error("numeric literal immediately followed by an identifier character", self._span_from(*start))
        text = self.source[start[0] : self.pos]
        return Token(Tok.REAL if is_real else Tok.INT, text, self._span_from(*start))

    def _punct(self, start: tuple[int, int, int]) -> Token:
        for p in _PUNCT:
            if self.source.startswith(p, self.pos):
                self._advance(len(p))
                return Token(Tok.PUNCT, p, self._span_from(*start))
        self._advance()
        ch = self.source[start[0]]
        raise self._error(f"unexpected character {ch!r}", self._span_from(*start))


def tokenize(source: str, filename: str = "<input>") -> list[Token]:
    return Lexer(source, filename).tokens()
