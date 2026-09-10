"""Code-generation backends.  v0.1 ships a PyTorch backend only."""

from .torch_codegen import generate_torch

__all__ = ["generate_torch"]
