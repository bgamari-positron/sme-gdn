# Single-head gated delta update

## What this object represents

`gdn_head<D, Scalar>` stores the recurrent memory for one Gated DeltaNet head in one sequence. Keep
the object between tokens: each call to `update()` changes the memory used by the next call. Use a
separate object for each head and each independent sequence.

`D` is the length of the query, key, value, and output vectors. This implementation requires the key
and value dimensions to be equal. Its state is therefore a `D × D` matrix, stored as
`state[key_component][value_component]`. For example, `gdn_head<128, float>` stores 128 rows of 128
floats: 64 KiB of recurrent state. A new object initializes that state to zero.

This code implements the recurrent memory operation only. The caller must supply the projections,
convolution, normalization, and other operations required by the surrounding model layer.

## Updating the memory for a token

Call `update(q, k, v, alpha, beta, output)` with:

| Argument | Meaning                                                        |
| -------- | -------------------------------------------------------------- |
| `q`      | Query vector: selects what to read from the updated memory.    |
| `k`      | Key vector: identifies the association to correct in memory.   |
| `v`      | Value vector: the value that memory should associate with `k`. |
| `alpha`  | Scalar decay multiplier applied to the old state.              |
| `beta`   | Scalar multiplier controlling the correction strength.         |
| `output` | Destination vector, overwritten with the readout at `q`.       |

The update first multiplies the old state by `alpha`. It then reads the decayed memory at `k` and
compares that prediction with `v`. The difference, multiplied by `beta`, is the correction written
into memory. Finally, it reads the corrected memory at `q`. **The output includes the current
token's update.**

With `S` denoting the stored matrix, the calculation is:

```text
decayed_state   = alpha * S
predicted_value = transpose(decayed_state) * k
correction      = beta * (v - predicted_value)
S               = decayed_state + outer(k, correction)
output          = transpose(S) * q
```

The gates are multipliers, not logits or logarithms. The caller supplies finite inputs and any key
normalization or query scaling required by the model. State and output must not overlap each other
or the input vectors; the three read-only inputs may overlap.

Here is a two-component example that shows how memory persists between calls:

```cpp
#include "gated_delta.h"

using Head = gated_delta::gdn_head<2, float>;
Head head; // The state starts at zero.
Head::vector_type q{1, 0}, k{1, 0}, v{3, 4}, output{};

head.update(q, k, v, 1.0f, 1.0f, output);
// The memory now associates k with {3, 4}. output is {3, 4}.

v = {5, 6};
head.update(q, k, v, 1.0f, 0.5f, output);
// Keep the old memory and apply half the correction. output is {4, 5}.
```

## Choosing an implementation

All implementations use the same methods and argument layout:

| Template argument     | Implementation                                                       |
| --------------------- | -------------------------------------------------------------------- |
| `float` (the default) | Generic reference calculation with FP32 state.                       |
| `double`              | The same reference code with FP64 state.                             |
| `sme_matrix_float`    | FP32 update using SME matrix instructions and ZA storage.            |
| `sme_vector_float`    | FP32 update using streaming vector instructions, without ZA storage. |

For example, select the vector implementation with `gdn_head<128, gated_delta::sme_vector_float>`.

## Other operations

- `predict(key)` returns the value currently associated with `key`, without changing the state.
  Component `j` of the result is `sum_i(key[i] * state[i][j])`. Passing a query vector performs the
  same readout as the final update step.
- `decay(alpha)` multiplies every state entry by `alpha`, without producing an output. `alpha = 1`
  preserves the state; `alpha = 0` clears it. An update already applies its decay, so do not call
  this first unless you intend an additional decay.
- `prefetch()` requests cache hints for a future update. It does not modify the memory or guarantee
  that data is cached. The scalar implementations do nothing; the SME implementations issue
  range-prefetch instructions.

The SME implementations inherit the non-streaming reference versions of `predict()` and `decay()`.
Separate calls add memory traffic and can incur CPU/CME coherence overhead. Use the fused `update()`
when you need a normal token update; its output already contains the query readout.

## SME implementation

`sme_matrix_float` is a backend tag, not a scalar wrapper. Its specialization inherits FP32 state, types,
prediction, and decay from the generic float head. It overrides `update` with the fused SME kernel
and `prefetch` with range hints. There are no virtual functions or runtime backend dispatch.
Selecting `sme_matrix_float` on a non-SME compilation target fails to compile.

The fused kernel uses FP32 throughout and handles arbitrary positive dimensions with predicated
tails. It computes residuals before modifying the corresponding state columns. It loads decayed
state blocks into ZA0.S and applies the correction with `FMOPA`. Updated rows supply both the stored
state and the output accumulation.

The kernel uses one ZA tile, two state reads, and no scratch arrays. It does not schedule prefetches
automatically. This is a correctness baseline, not a measured performance optimization.
Floating-point contraction and decay ordering can cause small differences between backends.

Both SME backends require hardware and OS support for streaming mode. The ZA-based backend also uses
SME FP32 outer products and private ZA storage. `sme_vector_float` inherits state and prefetch
behavior from `sme_matrix_float`, but replaces the update with streaming vector FMAs. It reads the state
twice and writes it once, without ZA allocation, slice transfers, or `FMOPA`. Selecting either SME
backend on a non-SME compilation target fails to compile. Standalone prediction and decay use the
inherited non-streaming scalar implementation. Calling them separately adds state traffic and can
cause CPU/CME coherence overhead; prefer the fused update when applicable.

## Prefetching

For an SME head, call `next_head.prefetch()` before independent work to hint a later update. The
method enters streaming mode and issues `RPRFM PSTKEEP` for nonoverlapping blocks of at most 1 KiB.
Each hint describes one contiguous block, with an unknown reuse distance. `PSTKEEP` requests
read-modify-write intent and temporal retention.

```cpp
next_head.prefetch();
current_head.update(q, k, v, alpha, beta, output);
// Later, update next_head with its own inputs.
```

C1 CME prefetches at most 1 KiB per instruction; a 128-dimensional FP32 head needs 64 hints. These
hints target the memory hierarchy, not ZA, and have no completion guarantee. An implementation
without FEAT_RPRFM treats the instruction as a NOP. Benchmark the lead time on hardware: whole-head
prefetches can evict useful data or exceed prefetch resources. Do not issue another prefetch for an
overlapping range before consuming the first.

## Build and validate with Nix

The flake pins Nixpkgs and provides an AArch64 cross-compiler, Make, and QEMU user-mode emulation.
It supports x86-64 and AArch64 Linux hosts. From the repository root:

```sh
nix develop "path:$PWD/examples/gated_delta" \
  --command make -C examples/gated_delta check
```

For an interactive shell:

```sh
cd examples/gated_delta
nix develop "path:$PWD"
make check
```

The Makefile builds a static AArch64 executable, so QEMU needs no runtime sysroot. `make check`
tests all four backends at streaming vector lengths of 16, 32, 64, 128, and 256 bytes. Use
`make check SVL_BYTES=64` to run one length, or `make clean` to remove build outputs. QEMU validates
instruction behavior, not CME performance or prefetch timing.

## Scalar-only check

On a host with a C++17 compiler:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  examples/gated_delta/test.cpp -o /tmp/gated-delta-test
/tmp/gated-delta-test
```

The same test code exercises float, double, and SME through their shared interface. The FP64 oracle
is independent of the generic head implementation. Tests cover initialization, prediction, decay,
prefetch execution, state updates, and outputs. They cover dimensions 1, 3, 16, 17, 32, 65, and 128,
repeated updates, and zero/one gate values. The double backend uses a tighter comparison tolerance
than FP32 backends. Do not use `-ffast-math` for these checks.
