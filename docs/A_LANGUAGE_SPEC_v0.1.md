# A Language Specification v0.1

Status: Draft for MVP implementation  
Purpose: Typed indexed tensor-algebra IR and source language  
Primary lowering target: PyTorch  

## 1. Overview

A is a pure, statically typed, declarative language for indexed tensor algebra.
It describes **what tensor computation means**, not how that computation is
scheduled or executed.

The intended compilation architecture is:

```text
M -> A
A x B -> Program
```

- `M` is the mathematical or algorithmic layer.
- `A` is the deterministic indexed tensor computation layer.
- `B` is the engineering policy: target language, device, layout, precision,
  parallelism, kernel selection, approximation, visualization, and related
  implementation choices.

An A program denotes a pure function:

```text
(inputs, parameters) -> outputs
```

The compiler derives a data-flow DAG from tensor definitions. Free indices
expose pointwise parallelism; reduction indices expose reduction structure.

### 1.1 Core design decisions

1. Indices and index spaces are first-class static objects.
2. Index spaces use nominal typing: equal sizes do not imply equal spaces.
3. Tensor definitions are immutable equations, not assignments.
4. Reductions are explicit in the core language.
5. Traditional repeated-index Einstein notation is optional surface sugar and
   is not part of the v0.1 MVP parser.
6. All v0.1 dependencies must form an acyclic graph.
7. A has exact mathematical semantics; numerical approximation belongs to B.

## 2. Scope of v0.1

The MVP must support:

- symbolic and fixed positive dimensions;
- nominal finite index spaces;
- scalar and tensor types;
- inputs, parameters, constants, tensor definitions, and outputs;
- typed tensor-element reads;
- scalar arithmetic, comparisons, Boolean expressions, and scalar calls;
- explicit finite reductions;
- local scalar bindings and conditional expressions;
- static name, type, index, shape, and DAG validation;
- lowering recognized contractions to `torch.einsum`;
- lowering remaining equations to valid PyTorch tensor operations.

The MVP does not need to support:

- mutation or state;
- general recursion, `while`, or runtime-dependent loops;
- automatic differentiation syntax;
- user-defined tensor functions;
- partial index maps, convolution padding, gather, or scatter;
- imports, packages, modules spanning multiple files;
- implicit Einstein summation;
- devices, layouts, dtypes, distribution, training loops, or initialization.

Sections marked **Post-MVP** specify intended extensions and need not be
implemented in the first compiler.

## 3. Source files and encoding

- Recommended extension: `.a`
- Source encoding: UTF-8
- Identifiers in v0.1: ASCII letters, digits, and underscore
- The language is case-sensitive.

Each source file contains exactly one module.

## 4. Lexical grammar

The grammar in this document uses EBNF notation.

```ebnf
letter      ::= "A" | ... | "Z" | "a" | ... | "z" ;
digit       ::= "0" | ... | "9" ;
identifier  ::= (letter | "_") { letter | digit | "_" } ;

integer     ::= digit { digit } ;
exponent    ::= ("e" | "E") [ "+" | "-" ] integer ;
real        ::= integer "." integer [ exponent ]
              | integer exponent ;

line_comment  ::= "//" { any-character-except-newline } ;
block_comment ::= "/*" { any-character } "*/" ;
```

Nested block comments are not supported in v0.1.

Whitespace and comments separate tokens but otherwise have no meaning.

## 5. Reserved words

The following words are reserved in v0.1:

```text
module
dim space constraint
input param const def output
let in
if then else
sum prod max min all any
true false
Bool Int Real Complex Tensor Fin
```

The following words are reserved for future versions:

```text
fn return
indexmap get default
import as
grad wrt
scan while
Dim Space
```

Names such as `relu`, `exp`, `log`, `sqrt`, and `sigmoid` are standard-library
identifiers, not keywords.

## 6. Punctuation and operators

```text
{ } ( ) [ ] < >
, ; : =
+ - * / **
== != < <= > >=
&& || !
```

`=` introduces a definition. `==` tests equality or states an equality
constraint. A program never uses `=` as mutation.

## 7. Module grammar

```ebnf
program       ::= module-declaration ;

module-declaration
              ::= "module" identifier "{"
                    { module-item }
                  "}" ;

module-item   ::= dimension-declaration
                | space-declaration
                | constraint-declaration
                | input-declaration
                | parameter-declaration
                | constant-declaration
                | tensor-definition
                | output-declaration ;
```

Example:

```a
module Example {
    dim N = 10;
    space Item = Fin(N);

    input x : Tensor<Real>[Item];

    def y[i : Item] = x[i] * 2.0;

    output y;
}
```

Declarations are order-independent. The compiler resolves names for the whole
module and then checks that value dependencies form a DAG.

## 8. Dimensions

A dimension is a compile-time positive natural number.

```a
dim BatchSize;
dim InputSize = 784;
dim HiddenSize = 256;
dim HeadSize = HiddenSize / 8;
```

An uninitialized dimension is a required module parameter. It must receive a
positive integer before code generation or runtime specialization.

```ebnf
dimension-declaration
              ::= "dim" identifier [ "=" dimension-expression ] ";" ;

dimension-expression
              ::= dimension-additive ;

dimension-additive
              ::= dimension-multiplicative
                  { ("+" | "-") dimension-multiplicative } ;

dimension-multiplicative
              ::= dimension-primary
                  { ("*" | "/") dimension-primary } ;

dimension-primary
              ::= integer
                | identifier
                | "(" dimension-expression ")" ;
```

Dimension division is exact integer division. If the dividend is not divisible
by the divisor, the program is rejected when dimensions become known.

## 9. Index spaces

An index space is a nominal finite type.

```a
space Batch  = Fin(BatchSize);
space Input  = Fin(InputSize);
space Output = Fin(InputSize);
```

Although `Input` and `Output` have equal cardinality above, they are distinct
types and their indices are not interchangeable.

```ebnf
space-declaration
              ::= "space" identifier "=" "Fin"
                  "(" dimension-expression ")" ";" ;
```

For `space S = Fin(N)`, the semantic index set is:

```text
{0, 1, ..., N - 1}
```

## 10. Dimension constraints

```a
constraint ModelSize == NumHeads * HeadSize;
constraint KernelSize <= InputSize;
```

```ebnf
constraint-declaration
              ::= "constraint" dimension-relation ";" ;

dimension-relation
              ::= dimension-expression comparison-operator
                  dimension-expression ;

comparison-operator
              ::= "==" | "!=" | "<" | "<=" | ">" | ">=" ;
```

Constraints are checked after dimension substitution and before lowering.

## 11. Types

```ebnf
type          ::= scalar-type | tensor-type ;

scalar-type   ::= "Bool" | "Int" | "Real" | "Complex" ;

tensor-type   ::= "Tensor" "<" scalar-type ">"
                  "[" [ space-name-list ] "]" ;

space-name-list
              ::= identifier { "," identifier } ;
```

Examples:

```a
Real
Tensor<Real>[Batch, Input]
Tensor<Int>[Batch]
Tensor<Bool>[Batch, Sequence]
Tensor<Real>[]
```

`Tensor<Real>[]` is a rank-zero tensor. It is distinct at the interface level
from the scalar type `Real`, although a backend may represent them similarly.

The order of tensor axes is significant.

## 12. Interface and constant declarations

```ebnf
input-declaration
              ::= "input" identifier ":" type ";" ;

parameter-declaration
              ::= "param" identifier ":" type ";" ;

constant-declaration
              ::= "const" identifier [ ":" scalar-type ]
                  "=" scalar-expression ";" ;
```

Example:

```a
input x : Tensor<Real>[Batch, Input];
input training : Bool;

param weight : Tensor<Real>[Input, Hidden];
param bias : Tensor<Real>[Hidden];

const epsilon : Real = 1.0e-6;
```

`input` is runtime data. `param` is mathematically also an input, but carries
metadata for training and code generation. Parameter initialization is not
part of A.

In v0.1, a `const` expression may refer only to literals, earlier or later
constants, dimensions converted by standard-library functions, and pure scalar
functions. Constant dependencies must also be acyclic.

## 13. Tensor definitions

Every computed tensor is defined by one immutable indexed equation.

```ebnf
tensor-definition
              ::= "def" identifier
                  "[" [ index-binder-list ] "]"
                  [ ":" scalar-type ]
                  "=" scalar-expression ";" ;

index-binder-list
              ::= index-binder { "," index-binder } ;

index-binder  ::= identifier ":" identifier ;
```

The second identifier in an index binder must name an index space.

Example:

```a
def z[b : Batch, h : Hidden] : Real =
    sum[i : Input](x[b, i] * weight[i, h]) + bias[h];
```

The compiler infers:

```text
z : Tensor<Real>[Batch, Hidden]
```

The optional scalar annotation checks the element type. It does not annotate
the whole tensor.

A scalar-valued result is written as a rank-zero tensor definition:

```a
def loss[] = sum[b : Batch](sample_loss[b]);
```

Every `def` name may be defined exactly once.

## 14. Tensor reads

```ebnf
tensor-read   ::= identifier "[" [ index-expression-list ] "]" ;

index-expression-list
              ::= index-expression { "," index-expression } ;

index-expression
              ::= identifier ;
```

v0.1 index expressions are index variables only. Arithmetic index expressions
and index maps are Post-MVP features.

Examples:

```a
x[b, i]
weight[i, h]
bias[h]
loss[]
```

A read is valid only if:

1. the number of supplied indices equals the tensor rank;
2. each index belongs to exactly the space required by the corresponding axis;
3. every index is in lexical scope.

## 15. Scalar expressions

Tensor equations are pointwise: their right-hand sides are scalar expressions.

```ebnf
scalar-expression
              ::= let-expression ;

let-expression
              ::= "let" identifier "=" scalar-expression
                  "in" scalar-expression
                | if-expression ;

if-expression ::= "if" scalar-expression
                  "then" scalar-expression
                  "else" scalar-expression
                | logical-or-expression ;

logical-or-expression
              ::= logical-and-expression
                  { "||" logical-and-expression } ;

logical-and-expression
              ::= equality-expression
                  { "&&" equality-expression } ;

equality-expression
              ::= relational-expression
                  { ("==" | "!=") relational-expression } ;

relational-expression
              ::= additive-expression
                  [ ("<" | "<=" | ">" | ">=") additive-expression ] ;

additive-expression
              ::= multiplicative-expression
                  { ("+" | "-") multiplicative-expression } ;

multiplicative-expression
              ::= power-expression
                  { ("*" | "/") power-expression } ;

power-expression
              ::= unary-expression [ "**" power-expression ] ;

unary-expression
              ::= ("+" | "-" | "!") unary-expression
                | primary-expression ;

primary-expression
              ::= integer
                | real
                | "true"
                | "false"
                | identifier
                | tensor-read
                | function-call
                | reduction-expression
                | "(" scalar-expression ")" ;

function-call ::= identifier "(" [ argument-list ] ")" ;

argument-list ::= scalar-expression { "," scalar-expression } ;
```

Operator precedence, from highest to lowest:

1. parentheses, reads, calls, reductions;
2. unary `+`, unary `-`, `!`;
3. `**` (right-associative);
4. `*`, `/`;
5. `+`, `-`;
6. `<`, `<=`, `>`, `>=`;
7. `==`, `!=`;
8. `&&`;
9. `||`;
10. `if ... then ... else ...` and `let ... in ...`.

## 16. Reductions

Reductions explicitly bind one or more indices.

```ebnf
reduction-expression
              ::= reduction-operator
                  "[" index-binder-list "]"
                  "(" scalar-expression ")" ;

reduction-operator
              ::= "sum" | "prod" | "max" | "min" | "all" | "any" ;
```

Examples:

```a
sum[i : Input](x[b, i] * weight[i, h])

sum[i : I, j : J](a[i, j] * b[i, j])

max[o : Output](logits[b, o])

all[i : Input](mask[b, i])
```

The bound index is in scope only inside the reduction body. A reduction removes
its bound indices from the free-index set of its body.

Reduction typing:

| Operator | Body type | Result type | Empty-space identity |
| --- | --- | --- | --- |
| `sum` | numeric | same numeric type | `0` |
| `prod` | numeric | same numeric type | `1` |
| `max` | ordered numeric | same type | invalid in v0.1 |
| `min` | ordered numeric | same type | invalid in v0.1 |
| `all` | `Bool` | `Bool` | `true` |
| `any` | `Bool` | `Bool` | `false` |

Because all v0.1 dimensions are positive, empty reductions normally cannot
occur.

## 17. Free indices and broadcasting

The indices declared on the left side of a tensor definition are its free
indices.

```a
def y[b : Batch, o : Output] = x[b, o] + bias[o];
```

`bias[o]` does not depend on `b`, so it is mathematically replicated for every
`b`. No separate broadcasting syntax is required.

For a definition:

```a
def y[i1 : S1, ..., in : Sn] = expression;
```

the static rules are:

1. every free index in `expression` must be declared on the left;
2. every other index must be bound by a reduction;
3. the expression may omit any left-side index, producing replication along
   that axis;
4. no index is implicitly summed in the v0.1 core language.

## 18. Local scalar bindings

```a
def y[b : Batch, h : Hidden] =
    let dot = sum[i : Input](x[b, i] * weight[i, h])
    in relu(dot + bias[h]);
```

`let` binds an immutable scalar. The binding is visible only after `in`.
Shadowing an existing name in an overlapping scope is rejected in v0.1.

## 19. Conditional expressions

```a
def result[b : Batch, i : Input] =
    if mask[b, i]
    then x[b, i]
    else 0.0;
```

The condition must have type `Bool`. Both branches must have the same type.
This is a pointwise value selection, not a statement-level control-flow branch.

## 20. Scalar standard library

The v0.1 implementation should provide at least:

```text
abs sign
exp log sqrt
sin cos tanh
relu sigmoid
floor ceil
scalar_min scalar_max
real
```

Example signatures:

```text
relu(Real) -> Real
exp(Real) -> Real
sqrt(Real) -> Real
scalar_max(Real, Real) -> Real
real(Int) -> Real
```

Higher-level tensor operations such as `linear`, `matmul`, `softmax`,
`layernorm`, `conv2d`, and `attention` are not core primitives. They should be
expressed as indexed equations or later provided as hygienic macros/functions
that expand into indexed equations.

## 21. Output declarations

```ebnf
output-declaration
              ::= "output" identifier { "," identifier } ";" ;
```

Example:

```a
output probability, loss;
```

Each output name must resolve to an input, parameter, constant scalar, or
defined tensor. Normally outputs are defined tensors.

## 22. Name resolution and scopes

A module contains three logical namespaces:

1. compile-time names: dimensions and spaces;
2. value names: inputs, parameters, constants, defined tensors, and functions;
3. lexical index names: left-side index binders and reduction binders.

An index variable may have the same spelling as a tensor because their
syntactic positions distinguish them, but implementations should emit a style
warning.

Duplicate names within the same namespace are errors. Undefined names are
errors. v0.1 rejects lexical shadowing.

## 23. Static type and well-formedness rules

A module is valid only if all of the following hold:

1. Every dimension is positive after specialization.
2. Every dimension expression is integral and satisfies all constraints.
3. Every tensor axis names a declared index space.
4. Every tensor read has the correct rank and space types.
5. Every index variable is lexically bound.
6. Every expression type-checks under scalar operator and function signatures.
7. The right side of every `def` has the annotated element type, if present.
8. Every tensor has exactly one definition.
9. All value and dimension dependencies are acyclic.
10. Every declared output resolves to a valid value.

Equal axis sizes never bypass nominal space checking.

## 24. Denotational semantics

For each declared space `S = Fin(N)`, let:

```text
[[S]] = {0, ..., N - 1}
```

A tensor of type:

```text
Tensor<T>[S1, ..., Sn]
```

denotes a total function:

```text
[[S1]] x ... x [[Sn]] -> [[T]]
```

For:

```a
def y[i1 : S1, ..., in : Sn] = e;
```

the denotation is:

```text
[[y]](i1, ..., in) = [[e]] under those index bindings
```

For a finite sum:

```a
sum[j : S](e)
```

the denotation is:

```text
the sum of [[e]] for every j in [[S]]
```

The semantics of `Real` is mathematical real arithmetic. A backend that lowers
it to floating-point arithmetic implements an approximation policy supplied by
B. Algebraic rewrites that are invalid under strict floating-point evaluation
must therefore be controlled by B rather than silently changing A.

## 25. Dependency graph and parallelism

Every tensor definition creates one logical DAG node. A definition depends on
every value read by its right-hand side.

For each equation:

- different tuples of left-side free indices are pointwise independent;
- indices bound by reductions identify reduction dimensions;
- shared subexpressions may be represented explicitly as additional `def`
  nodes;
- the backend may recognize contraction patterns and select optimized kernels.

A contains the information needed to infer possible parallelism, but it does
not choose a schedule.

## 26. PyTorch lowering requirements

The first backend should generate a Python module using PyTorch.

The backend may lower a recognized contraction such as:

```a
def y[b : Batch, o : Output] =
    sum[i : Input](x[b, i] * weight[i, o]);
```

to:

```python
y = torch.einsum("bi,io->bo", x, weight)
```

The `einsum` letters are backend-generated local symbols. They are never the
identity of A index spaces.

Broadcasting such as:

```a
def z[b : Batch, o : Output] = y[b, o] + bias[o];
```

may lower to:

```python
z = y + bias
```

The generated code must preserve declared axis order and must include runtime
shape assertions for unresolved module dimensions in the MVP.

If the backend cannot recognize a legal A expression as one optimized
operation, it may lower it compositionally. Optimization does not affect
language validity.

## 27. Complete example: two-layer MLP

```a
module MLP {
    dim B;
    dim I;
    dim H = 256;
    dim O = 10;

    space Batch  = Fin(B);
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

    def hidden[b : Batch, h : Hidden] =
        relu(preactivation[b, h]);

    def logits[b : Batch, o : Output] =
        sum[h : Hidden](hidden[b, h] * w2[h, o]) + b2[o];

    def maximum[b : Batch] =
        max[o : Output](logits[b, o]);

    def exponent[b : Batch, o : Output] =
        exp(logits[b, o] - maximum[b]);

    def denominator[b : Batch] =
        sum[o : Output](exponent[b, o]);

    def probability[b : Batch, o : Output] =
        exponent[b, o] / denominator[b];

    output probability;
}
```

Expected conceptual lowering:

```python
preactivation = torch.einsum("bi,ih->bh", x, w1) + b1
hidden = torch.relu(preactivation)
logits = torch.einsum("bh,ho->bo", hidden, w2) + b2
maximum = torch.amax(logits, dim=1)
exponent = torch.exp(logits - maximum[:, None])
denominator = torch.sum(exponent, dim=1)
probability = exponent / denominator[:, None]
```

## 28. Required compiler pipeline

The MVP implementation should use the following conceptual stages:

```text
Source text
  -> Lexer
  -> Concrete syntax tree or parser AST
  -> Name resolution
  -> Dimension and constraint checking
  -> Index-space and scalar type checking
  -> Explicit indexed core IR
  -> Dependency DAG validation
  -> Contraction-pattern recognition
  -> PyTorch code generation
```

The indexed core IR should contain explicit reduction binders even after the
surface language later gains Einstein shorthand.

A minimal expression IR is:

```text
Expr =
    Literal(value, scalar_type)
  | ScalarRef(name)
  | TensorRead(tensor_id, index_ids)
  | Unary(op, expression)
  | Binary(op, left, right)
  | Call(function_id, arguments)
  | If(condition, then_expression, else_expression)
  | Let(local_id, value, body)
  | Reduce(operator, bound_indices, body)
```

A tensor definition IR is:

```text
TensorDefinition = {
    tensor_id,
    free_indices: [(index_id, space_id)],
    element_type,
    body: Expr
}
```

Names should be resolved to stable internal IDs before type checking and code
generation.

## 29. Minimum diagnostics

Errors should include source spans and a short explanation. The MVP must detect
at least:

- unknown name;
- duplicate declaration;
- illegal lexical shadowing;
- undefined or unbound index;
- wrong tensor rank;
- index-space mismatch;
- scalar type mismatch;
- invalid reduction body;
- non-positive or non-integral dimension;
- unsatisfied dimension constraint;
- dependency cycle;
- missing output;
- unsupported construct in the current backend.

Example diagnostic:

```text
error[E0204]: index-space mismatch
  --> model.a:18:23
   |
18 |     def y[o : Output] = x[o];
   |                              ^ expected index from `Input`, found `Output`
   |
   = note: `Input` and `Output` both have size 768, but index spaces are nominal
```

## 30. MVP acceptance tests

The implementation is complete only when it can:

1. Parse and type-check the MLP example in section 27.
2. Generate importable PyTorch code for that example.
3. Numerically compare generated output with a direct PyTorch reference.
4. Reject a tensor read with the right size but wrong nominal index space.
5. Reject an unbound index.
6. Reject a cyclic pair of tensor definitions.
7. Infer broadcasting from omitted free indices.
8. Lower a matrix contraction to `torch.einsum`.
9. Lower `sum` and `max` reductions over one or more axes.
10. Preserve output axis order.

## 31. Post-MVP: explicit index maps

Future versions may add:

```a
indexmap conv_h(
    oh : OutputHeight,
    kh : KernelHeight
) -> InputHeight? =
    oh * stride + kh * dilation - padding;
```

`?` marks a partial index map. A partial read must state its out-of-bounds
value:

```a
get x[b, c, conv_h(oh, kh)] default 0.0
```

This provides a typed foundation for convolution, padding, slicing, gather,
and structured sparsity.

## 32. Post-MVP: user-defined tensor functions

Future syntax may use space-polymorphic pure functions:

```a
fn linear<B : Space, I : Space, O : Space>(
    x : Tensor<Real>[B, I],
    w : Tensor<Real>[I, O],
    bias : Tensor<Real>[O]
) -> Tensor<Real>[B, O] {
    def result[b : B, o : O] =
        sum[i : I](x[b, i] * w[i, o]) + bias[o];

    return result;
}
```

Such functions should elaborate into the same indexed core IR; they must not
become opaque backend operators.

## 33. Post-MVP: Einstein surface sugar

A future parser may accept restricted repeated-index notation such as:

```a
def y[b : Batch, o : Output] =
    x[b, i : Input] * weight[i, o];
```

and elaborate it to:

```a
def y[b : Batch, o : Output] =
    sum[i : Input](x[b, i] * weight[i, o]);
```

Implicit contraction should initially be restricted to a single multiplicative
term. It must not infer a reduction scope across addition, conditionals, local
bindings, or nonlinear calls. For example, the following should remain
illegal:

```a
relu(x[b, i] * weight[i, h] + bias[h])
```

The unambiguous form is:

```a
relu(sum[i : Input](x[b, i] * weight[i, h]) + bias[h])
```

## 34. Post-MVP: automatic differentiation

Automatic differentiation should be an A-to-A program transformation rather
than a scalar runtime primitive:

```text
grad : A -> A
```

Possible future surface syntax:

```a
grad loss wrt [w1, b1, w2, b2];
```

The transformation should generate ordinary indexed tensor equations, or the
backend may discharge it through a verified target autograd facility.

## 35. Non-goals and backend boundary

The following do not belong to the semantic core of A:

```text
CPU, GPU, NPU, or accelerator selection
PyTorch, JAX, CUDA, or other target APIs
physical dtype and mixed precision
memory layout and tensor strides
thread count and vector width
kernel tiling and fusion policy
distributed execution
parameter initialization
optimizer and training-loop configuration
data loading, logging, and visualization
```

These are B-level decisions. If B chooses an approximate implementation, it
must do so explicitly under an approximation or numerical policy; it must not
silently redefine the A program.

## 36. Recommended initial implementation strategy

For the first implementation, use a handwritten or parser-generator frontend
in a familiar implementation language. Do not implement LLVM or a custom
runtime initially.

The first milestone should compile only:

```text
dimensions + spaces + interfaces + constants
+ indexed definitions + sum/max reductions
+ scalar standard-library calls
-> PyTorch source
```

Once this subset is stable, add in order:

1. full scalar typing and diagnostics;
2. more reductions;
3. index maps and safe reads;
4. hygienic tensor functions/macros;
5. restricted Einstein sugar;
6. A-to-A transformations such as differentiation and algebraic optimization;
7. additional backends and B-policy integration.

