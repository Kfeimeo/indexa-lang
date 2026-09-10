"""Scalar and tensor types (spec section 11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScalarType(Enum):
    BOOL = "Bool"
    INT = "Int"
    REAL = "Real"
    COMPLEX = "Complex"
    #: Internal sentinel used after a type error to suppress cascading errors.
    ERROR = "<error>"

    def __str__(self) -> str:
        return self.value

    @property
    def is_numeric(self) -> bool:
        return self in (ScalarType.INT, ScalarType.REAL, ScalarType.COMPLEX)

    @property
    def is_ordered(self) -> bool:
        return self in (ScalarType.INT, ScalarType.REAL)

    @property
    def is_error(self) -> bool:
        return self is ScalarType.ERROR

    @staticmethod
    def parse(name: str) -> "ScalarType":
        return ScalarType(name)


@dataclass(frozen=True)
class TensorType:
    """``Tensor<element>[axes]`` where axes are space ids (order is significant)."""

    element: ScalarType
    axes: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.axes)
