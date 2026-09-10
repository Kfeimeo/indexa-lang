"""The v0.1 scalar standard library (spec section 20).

Each entry maps a function name to the list of admissible signatures,
``(argument types) -> result type``.  Names here are ordinary identifiers, not
keywords; a module may not redefine them because they live in the value
namespace.
"""

from __future__ import annotations

from .types import ScalarType as T

Signature = tuple[tuple[T, ...], T]

_REAL1: list[Signature] = [((T.REAL,), T.REAL)]
_NUM1: list[Signature] = [((T.INT,), T.INT), ((T.REAL,), T.REAL)]
_NUM2: list[Signature] = [((T.INT, T.INT), T.INT), ((T.REAL, T.REAL), T.REAL)]

FUNCTIONS: dict[str, list[Signature]] = {
    "abs": _NUM1,
    "sign": _NUM1,
    "exp": _REAL1,
    "log": _REAL1,
    "sqrt": _REAL1,
    "sin": _REAL1,
    "cos": _REAL1,
    "tanh": _REAL1,
    "relu": _REAL1,
    "sigmoid": _REAL1,
    "floor": _REAL1,
    "ceil": _REAL1,
    "scalar_min": _NUM2,
    "scalar_max": _NUM2,
    "real": [((T.INT,), T.REAL), ((T.REAL,), T.REAL)],
}


def signature_text(name: str) -> str:
    return " | ".join(
        f"{name}({', '.join(str(a) for a in args)}) -> {ret}" for args, ret in FUNCTIONS[name]
    )
