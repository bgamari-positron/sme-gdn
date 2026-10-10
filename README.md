# Single-head gated delta update

`gdn_head<D, Scalar = float>` owns one square state as a nested `std::array`. Choose `float` or
`double` for the generic reference implementation. Choose `sme_float` for the FP32 SME
implementation. All backends provide the same interface:

- `predict(key) const` returns `transpose(state) * key` without modifying the state.
- `decay(alpha)` scales the state in place.
- `update(q, k, v, alpha, beta, output)` updates the state and writes the output.
- `prefetch() const` hints a later state update. It is a no-op for scalar backends.

An update computes:

```text
r      = beta * (v - alpha * transpose(S) * k)
S      = alpha * S + outer(k, r)
output = transpose(S) * q
```

`alpha` and `beta` are scalar gates, not logits or log-decays. The caller supplies key normalization
and query scaling. State and output must not overlap each other or the input vectors. The three
read-only vectors may overlap.

```cpp
#include "gated_delta.h"

using Head = gated_delta::gdn_head<128, double>;
// Other choices: gdn_head<128, float> or gdn_head<128, gated_delta::sme_float>.
Head head; // Zero-initialized state.
Head::vector_type q{}, k{}, v{}, output{};
// Fill q, k, and v before the call.
head.prefetch();
head.update(q, k, v, 0.99, 0.5, output);
const auto prediction = head.predict(q);
```

## SME implementation

`sme_float` is a backend tag, not a scalar wrapper. Its specialization inherits FP32 state, types,
prediction, and decay from the generic float head. It overrides `update` with the fused SME kernel
and `prefetch` with range hints. There are no virtual functions or runtime backend dispatch.
Selecting `sme_float` on a non-SME compilation target fails to compile.

The fused kernel uses FP32 throughout and handles arbitrary positive dimensions with predicated
tails. It computes residuals before modifying the corresponding state columns. It loads decayed
state blocks into ZA0.S and applies the correction with `FMOPA`. Updated rows supply both the stored
state and the output accumulation.

The kernel uses one ZA tile, two state reads, and no scratch arrays. It does not schedule prefetches
automatically. This is a correctness baseline, not a measured performance optimization.
Floating-point contraction and decay ordering can cause small differences between backends.

The SME methods require hardware and OS support for SME FP32 outer products, streaming mode, and ZA.
The compiler manages streaming-mode transitions and private ZA storage through ACLE attributes.
Standalone prediction and decay use the inherited scalar implementation, not SME instructions.

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
tests all three backends at streaming vector lengths of 16, 32, 64, 128, and 256 bytes. Use
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
