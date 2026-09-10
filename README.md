# indexa-lang

**INDEXed tensor Algebra language** — a compiler for **A**, a pure, statically
typed, declarative language for indexed tensor algebra, targeting PyTorch.

This repository implements the
[A Language Specification v0.1](docs/A_LANGUAGE_SPEC_v0.1.md) (MVP scope,
sections 2–30).

```a
module MLP {
    dim B;                         // uninitialized: resolved from runtime shapes
    dim I;
    dim H = 256;
    dim O = 10;

    space Batch  = Fin(B);         // nominal index spaces
    space Input  = Fin(I);
    space Hidden = Fin(H);
    space Output = Fin(O);

    input x : Tensor<Real>[Batch, Input];
    param w1 : Tensor<Real>[Input, Hidden];
    param b1 : Tensor<Real>[Hidden];
    param w2 : Tensor<Real>[Hidden, Output];
    param b2 : Tensor<Real>[Output];

    def preactivation[b : Batch, h : Hidden] =
        sum[i : Input](x[b, i] * w1[i, h]) + b1[h];
    def hidden[b : Batch, h : Hidden] = relu(preactivation[b, h]);
    def logits[b : Batch, o : Output] =
        sum[h : Hidden](hidden[b, h] * w2[h, o]) + b2[o];
    def maximum[b : Batch] = max[o : Output](logits[b, o]);
    def exponent[b : Batch, o : Output] = exp(logits[b, o] - maximum[b]);
    def denominator[b : Batch] = sum[o : Output](exponent[b, o]);
    def probability[b : Batch, o : Output] = exponent[b, o] / denominator[b];

    output probability;
}
```

compiles to

```python
def forward(x, w1, b1, w2, b2, *, B=None, I=None):
    ...  # dimension resolution and shape assertions
    preactivation = torch.einsum("ab,bc->ac", x, w1) + b1
    hidden = torch.relu(preactivation)
    logits = torch.einsum("ab,bc->ac", hidden, w2) + b2
    maximum = torch.amax(logits, dim=1)
    exponent = torch.exp(logits - maximum[:, None])
    denominator = torch.sum(exponent, dim=1)
    probability = exponent / denominator[:, None]
    return probability
```

## Installation

The compiler itself has no dependencies beyond Python 3.10+. PyTorch is only
needed to *run* generated code (and the tests).

```sh
pip install -e ".[test]"          # compiler + pytest + torch
# or, without torch:
pip install -e .
```

## Command line

```sh
indexa check   examples/mlp.a                     # parse, resolve, type-check
indexa ir      examples/mlp.a                     # print the resolved core IR
indexa compile examples/mlp.a -o mlp_torch.py     # generate PyTorch code
indexa compile examples/mlp.a --dim B=32 --dim I=784   # specialize free dims
```

`python -m indexa ...` works without installing the console script.

Errors are reported with source spans in the layout of spec section 29:

```text
error[E0204]: index-space mismatch
  --> model.a:6:27
  |
6 |     def y[o : Output] = x[o];
  |                           ^ expected index from `Input`, found `Output`
  |
  = note: `Input` and `Output` both have size 768, but index spaces are nominal
```

## Python API

```python
import torch
from indexa import compile_file, compile_source

result = compile_file("examples/mlp.a")      # CompileResult
result.python                                # generated source text
result.module                                # core IR (indexa.ir.Module)
result.warnings                              # style warnings

mod = result.load()                          # exec the generated code
probability = mod.forward(x, w1, b1, w2, b2)
mod.resolve_dims(x, w1, b1, w2, b2)          # {'B': 5, 'I': 7, 'H': 256, 'O': 10}
```

`indexa.CompileError` carries a list of `Diagnostic`s (`.codes`, `.render()`).

## The generated module

Every compiled module is a self-contained Python file exposing:

| Name | Meaning |
| --- | --- |
| `forward(*inputs, *params, **free_dims)` | evaluates the module; returns the output tensor, or a tuple in `output` order |
| `resolve_dims(...)` | infers uninitialized dimensions from tensor shapes (or keyword arguments), evaluates derived dimensions with exact division, checks every `constraint`, returns a dict |
| `INPUTS`, `PARAMS`, `OUTPUTS`, `DIMS`, `FREE_DIMS` | interface metadata |
| `ShapeError` | raised for any runtime dimension, constraint or shape violation |

Arguments are inputs in declaration order followed by parameters in
declaration order. Every interface tensor's shape is asserted against its
declared type before any computation runs. Scalar-typed inputs are accepted as
Python numbers or 0-d tensors; scalar results (rank-zero tensors, constants)
are returned as 0-d tensors.

## Compiler pipeline

```text
source -> lexer -> parser AST -> checker -> core IR -> DAG validation -> PyTorch codegen
```

| Module | Stage |
| --- | --- |
| `indexa/lexer.py` | tokens, comments, reserved words (spec §4–6) |
| `indexa/parser.py` | recursive descent over the EBNF (spec §7–21) |
| `indexa/checker.py` | name resolution, dimension & constraint evaluation, index-space and scalar typing, shadowing rules (spec §8–23) → `indexa/ir.py` |
| `indexa/dag.py` | dependency cycle detection, evaluation order, dead-definition pruning (spec §25) |
| `indexa/backend/torch_codegen.py` | contraction recognition and PyTorch code generation (spec §26) |
| `indexa/cli.py`, `indexa/compiler.py` | command line and Python entry points |

The core IR matches spec section 28: `Literal`, `ScalarRef`, `LocalRef`,
`TensorRead`, `Unary`, `Binary`, `Call`, `If`, `Let`, `Reduce`, plus
`DimValue` for a dimension used as an `Int` scalar (`real(N)`). Names are
resolved to integer ids before type checking and code generation.

### Lowering strategy

Each definition gets a *canonical index order*: its left-hand-side indices in
declared order, then reduction binders in order of appearance. Every lowered
sub-expression keeps its axes in that order, so operands align with `None`
indexing only (`maximum[:, None]`) and PyTorch's right-aligned broadcasting;
the only transposes are on reads whose axis order differs from the canonical
one (`x.permute(1, 0)`).

- `sum[...]` over a product of `Real`/`Complex` tensor reads is recognized as a
  contraction and lowered to `torch.einsum` (the letters are backend-local and
  never identify A spaces).
- Other reductions lower to `torch.sum`, `torch.amax`, `torch.amin`,
  `torch.prod`, `torch.all`, `torch.any` over the bound axes.
- An index omitted on the right-hand side is replicated with `expand`, so
  declared output axis order is always preserved.
- `let` locals become temporaries, `if` becomes `torch.where`, repeated read
  indices such as `x[i, i]` become `torch.einsum("aa->a", x)`.
- Definitions not reachable from an `output` are type-checked but not emitted.

## Semantics notes (decisions the spec leaves to the implementation)

- **No implicit numeric conversion.** `x[i] + 1` with `Real` `x` is a type
  error; write `x[i] + 1.0` or `real(n)`. Literals are `Int` or `Real` by form.
- **`Int / Int`** is floor division (`torch.div(..., rounding_mode="floor")`).
  `Real` division is ordinary division.
- **Dimensions as scalars.** A dimension name may appear in any scalar
  expression as an `Int` (`real(H)`), which is how constants refer to sizes.
- **Shadowing.** A `let` name may not shadow an index, another local, a module
  value, a dimension or a standard-library function; a reduction binder may not
  shadow an in-scope index or local. Sibling reductions may reuse a name. An
  index spelled like a tensor is a style warning (`W0001`).
- **Constants** may use literals, other constants, dimensions and standard
  library functions only; they are computed at runtime after dimensions are
  resolved and returned as 0-d tensors when they are outputs.
- **`Real`** lowers to PyTorch's default floating dtype; `Tensor<Real>[]` and
  `Real` differ at the interface (a 0-d tensor vs. a scalar input) but share a
  representation.
- The standard library is exactly spec section 20: `abs sign exp log sqrt sin
  cos tanh relu sigmoid floor ceil scalar_min scalar_max real`.

## Diagnostics

| Code | Meaning |
| --- | --- |
| E0001 | syntax error |
| E0101 | unknown name |
| E0102 | duplicate declaration |
| E0103 | illegal lexical shadowing |
| E0201 | undefined or unbound index |
| E0202 | wrong tensor rank |
| E0203 | value is not a tensor |
| E0204 | index-space mismatch |
| E0301 | scalar type mismatch |
| E0302 | invalid reduction body |
| E0303 | invalid constant expression |
| E0401 | non-positive or non-integral dimension |
| E0402 | unsatisfied dimension constraint |
| E0501 | dependency cycle |
| E0502 | missing output |
| E0601 | unsupported construct in the current backend |
| W0001 | style warning |

All errors of a stage are reported together.

## Tests

```sh
pytest -q
```

`tests/test_codegen.py` covers the ten MVP acceptance tests of spec section 30
(the MLP example compiles, imports and agrees numerically with a direct
PyTorch reference; nominal-space, unbound-index and cycle rejection;
broadcasting inference; einsum lowering; `sum`/`max` over one or more axes;
output axis order). `tests/test_checker.py` covers the minimum diagnostics of
section 29, and `tests/test_lexer_parser.py` the grammar and precedence.

## Examples

- `examples/mlp.a` — the two-layer MLP of spec section 27.
- `examples/attention.a` — causal single-head attention with derived
  dimensions, a `constraint`, `let`, `if` and multi-axis reductions.

## Not in v0.1 (Post-MVP, spec §31–34)

Index maps and partial reads, user-defined tensor functions, Einstein
shorthand, automatic differentiation, and additional backends.
