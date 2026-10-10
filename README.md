# Single-head gated delta update

`gdn_head<D>` owns one square FP32 state as a nested `std::array`.
Its `update_reference` and `update_sme` methods process one token and return the output through a `std::array` argument.
Both update the state in place:

```text
r      = beta * (v - alpha * transpose(S) * k)
S      = alpha * S + outer(k, r)
output = transpose(S) * q
```

`alpha` and `beta` are scalar gates, not logits or log-decays.
The caller supplies key normalization and query scaling.
State and output must not overlap each other or the input vectors.
The three read-only vectors may overlap.

```cpp
#include "gated_delta.h"

using Head = gated_delta::gdn_head<64>;
Head head; // Zero-initialized state.
Head::vector_type q{}, k{}, v{}, output{};
// Fill q, k, and v before the call.
head.update_reference(q, k, v, 0.99f, 0.5f, output);
// On an SME target, call head.update_sme(...) instead.
```

## SME implementation

The kernel uses FP32 throughout. It supports arbitrary positive dimensions and streaming vector lengths, with predicated tails.
It computes a vector of residuals before modifying the corresponding state columns.
It then loads decayed state blocks into ZA0.S and applies the correction with `FMOPA`.
Updated rows supply both the stored state and the output accumulation.

The kernel uses one ZA tile, two state reads, and no scratch arrays.
It does not include range prefetch or target-specific scheduling.
This is a correctness baseline, not a measured performance optimization.
Floating-point contraction can cause small differences between implementations.

The SME function requires SME FP32 outer products and OS support for streaming mode and ZA.
The compiler manages private ZA storage and streaming-mode transitions through ACLE attributes.
The API has no runtime feature dispatch. Call it only on a supported target.

## Build and validate with Nix

The flake pins Nixpkgs and provides an AArch64 cross-compiler, Make, and QEMU user-mode emulation.
It supports x86-64 and AArch64 Linux hosts.
From the repository root:

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

The Makefile builds a static AArch64 executable, so QEMU needs no runtime sysroot.
`make check` enables SME and tests streaming vector lengths of 16, 32, 64, 128, and 256 bytes.
Use `make check SVL_BYTES=64` to run one length, or `make clean` to remove build outputs.
QEMU validates instruction behavior, not CME performance or prefetch timing.

## Reference-only check

On a host with a C++17 compiler:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  examples/gated_delta/test.cpp -o /tmp/gated-delta-test
/tmp/gated-delta-test
```

The tests compare state and output against an independent FP64 recurrence after each update.
They cover dimensions 1, 3, 16, 17, 32, 65, and 128, repeated updates, and zero/one gate values.
Do not use `-ffast-math` for these checks.
