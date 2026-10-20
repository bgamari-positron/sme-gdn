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

| Template argument      | Implementation                                                                 |
| ---------------------- | ------------------------------------------------------------------------------ |
| `float` (the default)  | Generic reference calculation with FP32 state.                                 |
| `double`               | The same reference code with FP64 state.                                       |
| `sme_matrix_float`     | FP32 update using SME matrix instructions and ZA storage.                      |
| `sme_vector_float`     | FP32 update using streaming vector instructions, without ZA storage.           |
| `sme_blocked_float<B>` | FP32 streaming-vector update that processes B adjacent vectors together (1–4). |

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

`sme_matrix_float` is a backend tag, not a scalar wrapper. Its specialization inherits FP32 state,
types, prediction, and decay from the generic float head. It overrides `update` with the fused SME
kernel and `prefetch` with range hints. There are no virtual functions or runtime backend dispatch.
Selecting `sme_matrix_float` on a non-SME compilation target fails to compile.

The fused kernel uses FP32 throughout and handles arbitrary positive dimensions with predicated
tails. It computes residuals before modifying the corresponding state columns. It loads decayed
state blocks into ZA0.S and applies the correction with `FMOPA`. Updated rows supply both the stored
state and the output accumulation.

The kernel uses one ZA tile, two state reads, and no scratch arrays. It does not schedule prefetches
automatically. This is a correctness baseline, not a measured performance optimization.
Floating-point contraction can cause small differences between backends.

Both SME backends require hardware and OS support for streaming mode. The ZA-based backend also uses
SME FP32 outer products and private ZA storage. `sme_vector_float` inherits state and prefetch
behavior from `sme_matrix_float`, but replaces the update with streaming vector FMAs. It reads the
state twice and writes it once, without ZA allocation, slice transfers, or `FMOPA`. Selecting either
SME backend on a non-SME compilation target fails to compile. Standalone prediction and decay use
the inherited non-streaming scalar implementation. Calling them separately adds state traffic and
can cause CPU/CME coherence overhead; prefer the fused update when applicable.

## Blocked streaming-vector implementation

Select a block size at compile time:

```cpp
using Head = gated_delta::gdn_head<128, gated_delta::sme_blocked_float<4>>;
```

`B` counts adjacent streaming vectors, not scalar elements. Each vector contains `svcntw()` floats;
on C1's 512-bit streaming engine, this is 16 floats. Thus `B=1`, `2`, `3`, or `4` processes 16, 32,
48, or 64 value columns per block on C1. Other streaming vector lengths work without recompilation,
and predicates handle incomplete final vectors and blocks.

The prediction loop loads each key component once and reuses it across B independent prediction
vectors. The correction loop likewise shares each key and query load across B state vectors. Each
output element still accumulates state rows in the original order. The update uses no ZA state and
retains two state reads and one state write. It inherits scalar prediction/decay and range prefetch
from the other SME backends.

With SME2 enabled, full B=2 and B=4 blocks load adjacent state vectors with a single grouped `LD1W`.
Both the prediction pass and the correction pass use these loads. The full-block check runs once per
block, outside the state-row loops. Incomplete blocks and B=1/3 retain predicated single-vector
loads and stores. An SME1 compilation uses single-vector loads and stores for every block.

The Makefile now targets `-march=armv9-a+sme2`, matching C1's SME2 support. The benchmark banner
identifies whether grouped loads are compiled in. For an SME1 comparison, clean first and rebuild
with the older target:

```sh
make clean
make benchmark CXXFLAGS='-O3 -g -std=c++17 -march=armv9-a+sme -Wall -Wextra -Werror'
```

Run these commands from the example directory inside its Nix shell. Clean again before returning to
the default SME2 build; Make does not track command-line flag changes.

The SME2 full-block paths are separate `detail::sme_full_block<D, 2>` and `detail::sme_full_block<D, 4>` specializations.
Each spells out its prediction and output accumulators directly, without conditional code for the other width.
Only the small per-vector arithmetic helpers are shared between them.
This reduces register copies while keeping both kernels straightforward to read.
Full blocks use matching two- or four-vector SME2 loads and state stores.
Output stores remain individual: grouping them increased hot-loop register shuffling in the checked GCC build.
Grouped state stores also require some register copies; hardware measurements must establish whether the reduced store instruction count helps.
The public `sme_blocked_float<B>` interface and the generic tail path are unchanged.

The generic tail path uses four-vector SVE tuples, with compile-time indices selecting the active
vectors. This limits B to 1–4. Small inline template helpers let the compiler unroll the tuple
operations. The helpers avoid a GCC 15 compiler crash encountered with streaming lambdas. The
checked GCC D=128 builds contain no helper calls or vector spills for any supported block size.
Hardware measurements must determine the best block size; QEMU does not establish a speedup.

## Updating several heads in one streaming region

`update_batch()` processes one token for each head in an array.
All heads in a batch have the same dimension and backend.
Queries, keys, values, gates, and outputs are parallel arrays: element `i` belongs to head `i`.

```cpp
using Head = gated_delta::gdn_head<128, gated_delta::sme_blocked_float<4>>;
constexpr std::size_t count = 3;
std::array<Head, count> heads{};
std::array<Head::vector_type, count> q{}, k{}, v{}, output{};
std::array<Head::scalar_type, count> alpha{}, beta{};
alpha.fill(0.99f);
beta.fill(0.5f);
for (std::size_t i = 0; i < count; ++i) {
    q[i][0] = k[i][0] = 1.0f;
    v[i][0] = float(i + 1);
}
gated_delta::update_batch(heads.data(), q.data(), k.data(), v.data(),
                         alpha.data(), beta.data(), output.data(), count);
// After this first update, output[i][0] is 0.5 * (i + 1).
// Keep heads between tokens and supply the next token's input arrays.
```

Every pointer must address at least `count` elements, with the same non-overlap requirements as a single-head update.
A zero count is a no-op, and all pointers may be null in that case.
The function also accepts storage supplied by `std::vector` or another contiguous container.

For SME backends, the batch enters streaming mode once and returns to the caller's mode after all heads finish.
The matrix backend allocates one private ZA context for the batch.
Vector and blocked backends do not allocate ZA.
For float and double, the function calls the generic update for each head without entering streaming mode.
The batch does not issue prefetch hints automatically.

SME heads also expose `update_streaming()` and `prefetch_streaming()` for custom streaming loops.
Call these from a function declared with the appropriate ACLE streaming attributes, not from ordinary non-streaming code.
The matrix `update_streaming()` additionally requires the caller to provide active ZA via ACLE attributes and uses it as scratch.
Vector updates and prefetches do not use ZA.
Existing `update()` and `prefetch()` methods remain the ordinary, single-head entry points.
Their mode-changing wrappers are deliberately out of line to avoid GCC 15 streaming-inlining failures.

The tests compare batches of zero through three heads against individual calls, including untouched heads beyond the requested count.
They cover both 17- and 128-dimensional state and execute streaming prefetch calls separately.
The current benchmark still measures individual calls; it does not measure the batching benefit.

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

## Prefill

`prefill.h` provides `gdn_prefill<D, Backend>` for consecutive tokens of one head.
The object retains its final state between calls. Select one of these backends:

- `reference_prefill<GenerationBackend>` calls an existing generation backend per token.
- `rowwise_prefill<ChunkSize, Scalar>` performs portable row-wise WY formation.
- `sme_rowwise_prefill<ChunkSize>` performs FP32 WY formation with SME.

```cpp
#include "prefill.h"
using Prefill = gated_delta::gdn_prefill<128, gated_delta::sme_rowwise_prefill<64>>;
Prefill head;
// q, k, v, and output point to token_count vector_type elements.
// alpha and beta point to token_count scalar_type elements.
head.update(q, k, v, alpha, beta, output, token_count);
```

The arrays use token order, not head order. Each output includes its token's update.
The same finite-input and non-overlap requirements apply as for generation.
A zero token count permits null pointers and leaves state unchanged.
Chunked backends handle partial final chunks and zero decay gates without division.
Floating-point evaluation order differs from sequential generation.

The SME backend uses one streaming region and private ZA context per call.
ZA0.S accumulates tiled Gram, attention, state-read, and state-update products.
Streaming vectors perform row-wise forward substitution.
The backend requires SME FP32 outer products, but not SME2.
Selecting it without an SME compilation target fails to compile.

Scratch storage is on the stack: `4*C*D + 2*C*C + C` floats for chunk size `C`.
For `D=128, C=64`, this is 160.25 KiB, excluding the object state and compiler spills.
Packing is scalar and the two chunk matrices use dense storage.
This is a correctness baseline; hardware measurements must establish performance.
The existing benchmark measures generation calls, not this prefill backend.

## Build and validate with Nix

The flake pins Nixpkgs and provides an AArch64 cross-compiler, Make, and QEMU user-mode emulation.
It supports x86-64 and AArch64 Linux hosts. From the repository root:

```sh
nix develop "path:$PWD" --command make check
```

For an interactive shell:

```sh
nix develop "path:$PWD"
make check
```

The Makefile builds a static AArch64 executable, so QEMU needs no runtime sysroot. `make check`
tests all backends, including block sizes 1–4, at streaming vector lengths of 16, 32, 64, 128, and
256 bytes. Use `make check SVL_BYTES=64` to run one length, or `make clean` to remove build outputs.
QEMU validates instruction behavior, not CME performance or prefetch timing.

## Benchmarking 128-dimensional heads

`benchmark.cpp` compares float, double, matrix SME, vector SME, and blocked SME with block sizes
1–4. It reports median and minimum nanoseconds per update, plus a checksum of the final state and
output. Build the static AArch64 executable from the repository root:

```sh
nix develop "path:$PWD" --command make benchmark
```

Copy `build/benchmark` to an SME2-capable AArch64 system for the default build.
On that system, run:

```sh
./benchmark 10000 5
```

The arguments are updates per trial and number of trials; the defaults are 10000 and 5. Use enough
updates to make each trial long compared with timer resolution and scheduling noise. On Linux, use
`taskset -c <cpu> ./benchmark 10000 5` to pin the process to the intended CPU. Control CPU/CME
clocks and competing work when comparing hardware results.

Each trial starts with zero state and performs 64 untimed warmup updates. The timed loop updates one
persistent head using 16 repeating, deterministic query/key/value inputs. All backends receive the
same numeric inputs, including float-to-double conversion for the double backend. State and input
storage are aligned to 64 bytes. Allocation, initialization, warmup, and checksum computation are
outside the timed interval. Compiler barriers and the checksum prevent the state updates from being
discarded. Explicit prefetch is disabled, and the SME measurements include per-call streaming-mode
transitions. This measures repeated single-head API calls, not a multi-head scheduler or a
persistent streaming batch. Run the correctness tests separately; a finite checksum alone does not
establish correctness.

For a short QEMU execution check:

```sh
nix develop "path:$PWD" --command make benchmark-smoke
```

This runs 32 updates per trial and three trials at a 64-byte streaming vector length. **Do not use
QEMU timings to select the fastest C1 kernel.** They measure emulator behavior, not CME execution. A
host-only build can compare the scalar backends:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  benchmark.cpp -o /tmp/gdn-benchmark
/tmp/gdn-benchmark 10000 5
```

## Scalar-only check

On a host with a C++17 compiler:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  test.cpp -o /tmp/gated-delta-test
/tmp/gated-delta-test
```

The same test code exercises float, double, and SME through their shared interface. The FP64 oracle
is independent of the generic head implementation. Tests cover initialization, prediction, decay,
prefetch execution, state updates, and outputs. They cover dimensions 1, 3, 16, 17, 32, 65, and 128,
repeated updates, and zero/one gate values. The double backend uses a tighter comparison tolerance
than FP32 backends. Do not use `-ffast-math` for these checks.
