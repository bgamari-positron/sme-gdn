#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

namespace gated_delta {

// Selects the FP32 matrix implementation: gdn_head<D, sme_matrix_float>.
// update() uses streaming mode, ZA matrix storage, and FMOPA outer products.
// This empty tag selects code; state and vector elements remain ordinary floats.
// Requires an SME compilation target and compatible hardware/OS support.
// Standalone predict() and decay() still use the non-streaming reference code.
struct sme_matrix_float {};

// Selects the FP32 vector implementation: gdn_head<D, sme_vector_float>.
// update() uses streaming vector loads, multiplies, and fused multiply-adds.
// It does not use ZA or matrix instructions, but still requires SME streaming mode.
// Stores ordinary floats and shares the matrix backend's range-prefetch method.
// Standalone predict() and decay() still use the non-streaming reference code.
struct sme_vector_float {};

// Selects a streaming-vector update that handles B adjacent vectors per block.
// B counts vectors, not elements: on C1, B=4 covers 64 value columns.
// Supports 1 through 4 vectors and reuses each key/query load across the block.
// State remains FP32; prediction/decay and prefetch follow sme_vector_float.
template <std::size_t B>
struct sme_blocked_float {
    static_assert(B >= 1 && B <= 4, "A block must contain 1 through 4 streaming vectors");
};

// Stores the recurrent memory for one Gated DeltaNet head in one sequence.
// Keep this object between tokens. Each update() call processes the next token
// and changes the memory used by subsequent calls.
//
// D is the number of elements in each query, key, value, and output vector.
// This implementation requires the key and value dimensions to be equal.
// The state therefore has D rows and D columns. state[i][j] connects key
// component i to value component j; each row is contiguous in memory.
// New objects start with an all-zero state.
//
// For each token, update(q, k, v, alpha, beta, output) performs these steps:
//   1. Multiply the old state by alpha to decay previous memories.
//   2. Read the decayed memory at key k to predict its associated value.
//   3. Compute the correction: beta * (v - predicted_value).
//   4. Add the outer product of k and that correction to the state.
//   5. Read the updated memory at query q and write the result to output.
//
// q, k, and v are the current token's query, key, and value vectors.
// alpha is the decay multiplier; beta controls the correction strength.
// Both gates are scalar multipliers, not logits or logarithms.
// Inputs must be finite. The caller supplies any key normalization and query
// scaling required by the model; these methods do not perform either operation.
//
// State and output must not overlap each other or q, k, v.
// The three read-only input vectors may overlap.
// Use separate objects for different heads or independent sequences.
// This implements the recurrent update, not the surrounding neural-network layer.
template <std::size_t D, class Scalar = float>
struct gdn_head {
    static_assert(D > 0, "A head must have at least one dimension");
    static_assert(std::is_floating_point_v<Scalar>, "Use a floating-point scalar or an SME specialization");
    using scalar_type = Scalar;
    static constexpr bool uses_sme = false;
    static constexpr bool uses_za = false;
    using vector_type = std::array<scalar_type, D>;
    using state_type = std::array<vector_type, D>;

    state_type state{};

    // Read the value associated with key without changing the state.
    // Result component j is sum_i(key[i] * state[i][j]).
    // Passing a query vector instead reads the output for that query.
    // SME backends inherit this non-streaming scalar implementation.
    // A separate call reads the state again; update() already computes its output.
    // Switching state access between the CPU and CME can add coherence overhead.
    vector_type predict(const vector_type& key) const {
        vector_type prediction{};
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                prediction[j] += key[i] * state[i][j];
        return prediction;
    }

    // Multiply every state entry by alpha. Does not compute an output.
    // alpha=1 preserves the state; alpha=0 clears it.
    // update() already applies its alpha, so do not decay twice for one token.
    // SME backends inherit this non-streaming scalar implementation.
    // A separate call adds a full state read/write pass and can cause CPU/CME
    // coherence overhead. Prefer update(), which fuses decay into the update.
    void decay(scalar_type alpha) {
        for (auto& row : state)
            for (scalar_type& value : row)
                value *= alpha;
    }

    // Request that the state be fetched before a later update. Does not change it.
    // Scalar backends do nothing; SME backends issue best-effort cache hints.
    void prefetch() const noexcept {}

    // Incorporate this token's (k, v) association, then read the result at q.
    // Mutates state and overwrites output. The output includes this token's update.
    // q, k, v, and output each contain D elements; alpha and beta are scalars.
    void update(const vector_type& q, const vector_type& k,
                const vector_type& v, scalar_type alpha, scalar_type beta,
                vector_type& output) {
        decay(alpha);
        vector_type residual = predict(k);
        for (std::size_t j = 0; j < D; ++j)
            residual[j] = beta * (v[j] - residual[j]);
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                state[i][j] += k[i] * residual[j];
        output = predict(q);
    }
};

// Incomplete on non-SME targets: selecting this backend must not silently fall back.
template <std::size_t D>
struct gdn_head<D, sme_matrix_float>;
template <std::size_t D>
struct gdn_head<D, sme_vector_float>;
template <std::size_t D, std::size_t B>
struct gdn_head<D, sme_blocked_float<B>>;

#if defined(__ARM_FEATURE_SME)
// Keep mode-changing wrappers out of line: GCC 15 can lose streaming context
// when it inlines them into non-streaming callers. Streaming entry points inline
// within a batch so that there is no mode switch or function call per head.
template <std::size_t D>
struct gdn_head<D, sme_matrix_float> : gdn_head<D, float> {
    using base_type = gdn_head<D, float>;
    using typename base_type::vector_type;
    using base_type::state;
    static constexpr bool uses_sme = true;
    static constexpr bool uses_za = true;

    // Standalone predict/decay reuse the scalar methods; update remains fused.
    // Hint the state for a later read-modify-write on the CME.
    // C1 prefetches at most 1 KiB per RPRFM. Timing remains caller-controlled.
    __arm_locally_streaming
    __attribute__((noinline))
    void prefetch() const noexcept {
        prefetch_streaming();
    }

    // For callers already in streaming mode; does not allocate or change ZA.
    __attribute__((always_inline))
    void prefetch_streaming() const noexcept __arm_streaming {
        const auto* base = reinterpret_cast<const unsigned char*>(&state);
        for (std::size_t offset = 0; offset < sizeof(state); offset += 1024) {
            const std::size_t remaining = sizeof(state) - offset;
            // Single contiguous block: count-1, stride, and reuse distance are zero.
            const uint64_t metadata = remaining < 1024 ? remaining : 1024;
            asm volatile("rprfm pstkeep, %x[metadata], [%[address]]"
                         :
                         : [metadata] "r" (metadata), [address] "r" (base + offset)
                         : "memory");
        }
    }

    // Requires SME FP32 outer products and OS support for streaming mode/ZA.
    // The compiler enters streaming mode and manages private ZA state.
    // ponytail: one ZA tile and two state reads; tune residency/prefetch only
    // after measuring this kernel on the target CPU or CME.
    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void update(const vector_type& q, const vector_type& k,
                const vector_type& v, float alpha, float beta, vector_type& output) {
        update_streaming(q, k, v, alpha, beta, output);
    }

    // Uses the caller's active ZA as scratch. ZA contents are not preserved.
    __attribute__((always_inline))
    void update_streaming(const vector_type& q, const vector_type& k,
                          const vector_type& v, float alpha, float beta,
                          vector_type& output) __arm_streaming __arm_inout("za") {
        const std::size_t lanes = svcntw(); // Streaming length, not normal VL.
        const svfloat32_t zero = svdup_f32(0.0f);

        for (std::size_t col = 0; col < D; col += lanes) {
            const svbool_t columns = svwhilelt_b32_u64(col, D);
            svfloat32_t prediction = zero;
            for (std::size_t i = 0; i < D; ++i) {
                const svfloat32_t s = svld1_f32(columns, state[i].data() + col);
                prediction = svmla_n_f32_m(columns, prediction, s, k[i]);
            }
            const svfloat32_t decayed_prediction =
                svmul_n_f32_x(columns, prediction, alpha);
            const svfloat32_t error = svsub_f32_x(
                columns, svld1_f32(columns, v.data() + col), decayed_prediction);
            const svfloat32_t residual = svmul_n_f32_x(columns, error, beta);

            svfloat32_t y = zero;
            for (std::size_t row = 0; row < D; row += lanes) {
                const svbool_t rows = svwhilelt_b32_u64(row, D);
                const std::size_t height = D - row < lanes ? D - row : lanes;

                // Fill the active rectangle of ZA0.S with the decayed state.
                for (std::size_t i = 0; i < height; ++i) {
                    const svfloat32_t s = svmul_n_f32_x(
                        columns, svld1_f32(columns, state[row + i].data() + col), alpha);
                    svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), columns, s);
                }

                // ZA0.S += k[row:row+lanes] * residual^T.
                svmopa_za32_f32_m(0, rows, columns,
                                 svld1_f32(rows, k.data() + row), residual);

                // Read each updated row for both output and state storage.
                for (std::size_t i = 0; i < height; ++i) {
                    const svfloat32_t s = svread_hor_za32_f32_m(
                        zero, columns, 0, static_cast<uint32_t>(i));
                    svst1_f32(columns, state[row + i].data() + col, s);
                    y = svmla_n_f32_m(columns, y, s, q[row + i]);
                }
            }
            svst1_f32(columns, output.data() + col, y);
        }
    }
};

// Reuse state, scalar predict/decay, and range prefetch from the SME backend.
template <std::size_t D>
struct gdn_head<D, sme_vector_float> : gdn_head<D, sme_matrix_float> {
    using base_type = gdn_head<D, sme_matrix_float>;
    using typename base_type::vector_type;
    using base_type::state;
    static constexpr bool uses_za = false;

    // Two state reads and one write, with no ZA allocation or slice transfers.
    __arm_locally_streaming
    __attribute__((noinline))
    void update(const vector_type& q, const vector_type& k,
                const vector_type& v, float alpha, float beta, vector_type& output) {
        update_streaming(q, k, v, alpha, beta, output);
    }

    // Streaming entry point for batches; does not use ZA.
    __attribute__((always_inline))
    void update_streaming(const vector_type& q, const vector_type& k,
                          const vector_type& v, float alpha, float beta,
                          vector_type& output) __arm_streaming {
        const std::size_t lanes = svcntw();
        const svfloat32_t zero = svdup_f32(0.0f);

        for (std::size_t col = 0; col < D; col += lanes) {
            const svbool_t columns = svwhilelt_b32_u64(col, D);
            svfloat32_t prediction = zero;
            for (std::size_t i = 0; i < D; ++i) {
                const svfloat32_t s = svld1_f32(columns, state[i].data() + col);
                prediction = svmla_n_f32_m(columns, prediction, s, k[i]);
            }
            const svfloat32_t decayed_prediction =
                svmul_n_f32_x(columns, prediction, alpha);
            const svfloat32_t error = svsub_f32_x(
                columns, svld1_f32(columns, v.data() + col), decayed_prediction);
            const svfloat32_t residual = svmul_n_f32_x(columns, error, beta);

            svfloat32_t y = zero;
            for (std::size_t i = 0; i < D; ++i) {
                svfloat32_t s = svld1_f32(columns, state[i].data() + col);
                s = svmul_n_f32_x(columns, s, alpha);
                s = svmla_n_f32_m(columns, s, residual, k[i]);
                svst1_f32(columns, state[i].data() + col, s);
                y = svmla_n_f32_m(columns, y, s, q[i]);
            }
            svst1_f32(columns, output.data() + col, y);
        }
    }
};

#if defined(__ARM_FEATURE_SME2)
namespace detail {

__attribute__((always_inline))
inline svfloat32_t residual_vector(svbool_t pg, svfloat32_t prediction,
                                  const float* value, float alpha, float beta) __arm_streaming {
    prediction = svmul_n_f32_x(pg, prediction, alpha);
    const svfloat32_t error = svsub_f32_x(pg, svld1_f32(pg, value), prediction);
    return svmul_n_f32_x(pg, error, beta);
}

__attribute__((always_inline))
inline svfloat32_t correct_vector(svbool_t pg, svfloat32_t previous,
                                 svfloat32_t residual, svfloat32_t key, float alpha) __arm_streaming {
    const svfloat32_t decayed = svmul_n_f32_x(pg, previous, alpha);
    return svmla_f32_x(pg, decayed, residual, key);
}

// Full-block kernels are deliberately separate: SVE vectors cannot form arrays,
// and named accumulators avoid the register copies generated for tuple updates.
// The caller must ensure that the entire block fits; grouped stores write every lane.
template <std::size_t D, std::size_t B>
struct sme_full_block;

template <std::size_t D>
struct sme_full_block<D, 2> {
    using head_type = gdn_head<D, float>;
    using vector_type = typename head_type::vector_type;
    using state_type = typename head_type::state_type;

    __attribute__((always_inline))
    static inline void update(state_type& state, const vector_type& q, const vector_type& k,
                              const vector_type& v, float alpha, float beta,
                              vector_type& output, std::size_t col, std::size_t lanes) __arm_streaming {
        const svbool_t pg0 = svwhilelt_b32_u64(col, D);
        const svbool_t pg1 = svwhilelt_b32_u64(col + lanes, D);
        const svfloat32_t zero = svdup_f32(0.0f);
        svfloat32_t r0 = zero, r1 = zero;
        for (std::size_t row = 0; row < D; ++row) {
            const svfloat32x2_t values = svld1_f32_x2(svptrue_c32(), state[row].data() + col);
            const svfloat32_t key = svdup_f32(k[row]);
            r0 = svmla_f32_m(pg0, r0, svget2_f32(values, 0), key);
            r1 = svmla_f32_m(pg1, r1, svget2_f32(values, 1), key);
        }
        r0 = residual_vector(pg0, r0, v.data() + col, alpha, beta);
        r1 = residual_vector(pg1, r1, v.data() + col + lanes, alpha, beta);

        svfloat32_t y0 = zero, y1 = zero;
        for (std::size_t row = 0; row < D; ++row) {
            const svfloat32x2_t values = svld1_f32_x2(svptrue_c32(), state[row].data() + col);
            const svfloat32_t key = svdup_f32(k[row]);
            const svfloat32_t query = svdup_f32(q[row]);
            const svfloat32_t s0 = correct_vector(pg0, svget2_f32(values, 0), r0, key, alpha);
            const svfloat32_t s1 = correct_vector(pg1, svget2_f32(values, 1), r1, key, alpha);
            svst1_f32_x2(svptrue_c32(), state[row].data() + col, svcreate2_f32(s0, s1));
            y0 = svmla_f32_m(pg0, y0, s0, query);
            y1 = svmla_f32_m(pg1, y1, s1, query);
        }
        svst1_f32(pg0, output.data() + col, y0);
        svst1_f32(pg1, output.data() + col + lanes, y1);
    }
};

template <std::size_t D>
struct sme_full_block<D, 4> {
    using head_type = gdn_head<D, float>;
    using vector_type = typename head_type::vector_type;
    using state_type = typename head_type::state_type;

    __attribute__((always_inline))
    static inline void update(state_type& state, const vector_type& q, const vector_type& k,
                              const vector_type& v, float alpha, float beta,
                              vector_type& output, std::size_t col, std::size_t lanes) __arm_streaming {
        const svbool_t pg0 = svwhilelt_b32_u64(col, D);
        const svbool_t pg1 = svwhilelt_b32_u64(col + lanes, D);
        const svbool_t pg2 = svwhilelt_b32_u64(col + 2 * lanes, D);
        const svbool_t pg3 = svwhilelt_b32_u64(col + 3 * lanes, D);
        const svfloat32_t zero = svdup_f32(0.0f);
        svfloat32_t r0 = zero, r1 = zero, r2 = zero, r3 = zero;
        for (std::size_t row = 0; row < D; ++row) {
            const svfloat32x4_t values = svld1_f32_x4(svptrue_c32(), state[row].data() + col);
            const svfloat32_t key = svdup_f32(k[row]);
            r0 = svmla_f32_m(pg0, r0, svget4_f32(values, 0), key);
            r1 = svmla_f32_m(pg1, r1, svget4_f32(values, 1), key);
            r2 = svmla_f32_m(pg2, r2, svget4_f32(values, 2), key);
            r3 = svmla_f32_m(pg3, r3, svget4_f32(values, 3), key);
        }
        r0 = residual_vector(pg0, r0, v.data() + col, alpha, beta);
        r1 = residual_vector(pg1, r1, v.data() + col + lanes, alpha, beta);
        r2 = residual_vector(pg2, r2, v.data() + col + 2 * lanes, alpha, beta);
        r3 = residual_vector(pg3, r3, v.data() + col + 3 * lanes, alpha, beta);

        svfloat32_t y0 = zero, y1 = zero, y2 = zero, y3 = zero;
        for (std::size_t row = 0; row < D; ++row) {
            const svfloat32x4_t values = svld1_f32_x4(svptrue_c32(), state[row].data() + col);
            const svfloat32_t key = svdup_f32(k[row]);
            const svfloat32_t query = svdup_f32(q[row]);
            const svfloat32_t s0 = correct_vector(pg0, svget4_f32(values, 0), r0, key, alpha);
            const svfloat32_t s1 = correct_vector(pg1, svget4_f32(values, 1), r1, key, alpha);
            const svfloat32_t s2 = correct_vector(pg2, svget4_f32(values, 2), r2, key, alpha);
            const svfloat32_t s3 = correct_vector(pg3, svget4_f32(values, 3), r3, key, alpha);
            svst1_f32_x4(svptrue_c32(), state[row].data() + col, svcreate4_f32(s0, s1, s2, s3));
            y0 = svmla_f32_m(pg0, y0, s0, query);
            y1 = svmla_f32_m(pg1, y1, s1, query);
            y2 = svmla_f32_m(pg2, y2, s2, query);
            y3 = svmla_f32_m(pg3, y3, s3, query);
        }
        svst1_f32(pg0, output.data() + col, y0);
        svst1_f32(pg1, output.data() + col + lanes, y1);
        svst1_f32(pg2, output.data() + col + 2 * lanes, y2);
        svst1_f32(pg3, output.data() + col + 3 * lanes, y3);
    }
};

} // namespace detail
#endif

template <std::size_t D, std::size_t B>
struct gdn_head<D, sme_blocked_float<B>> : gdn_head<D, sme_vector_float> {
    static_assert(B >= 1 && B <= 4, "A block must contain 1 through 4 streaming vectors");
    using base_type = gdn_head<D, sme_vector_float>;
    using typename base_type::vector_type;
    using base_type::state;

private:
    // SVE vectors cannot be array elements. These helpers unroll constant tuple
    // indices without capturing sizeless vectors in lambdas, which crashes GCC 15.

    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat32x4_t predict_row(svfloat32x4_t acc, const vector_type& row,
                                            std::size_t col, std::size_t lanes,
                                            svfloat32_t key) __arm_streaming {
        const std::size_t column = col + I * lanes;
        const svbool_t pg = svwhilelt_b32_u64(column, D);
        const std::size_t offset = column < D ? column : D;
        const svfloat32_t s = svld1_f32(pg, row.data() + offset);
        acc = svset4_f32(acc, I, svmla_f32_m(pg, svget4_f32(acc, I), s, key));
        if constexpr (I + 1 < B)
            return predict_row<I + 1>(acc, row, col, lanes, key);
        return acc;
    }

    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat32x4_t update_row(svfloat32x4_t y, vector_type& row,
                                          svfloat32x4_t residual, std::size_t col,
                                          std::size_t lanes, float alpha,
                                          svfloat32_t key, svfloat32_t query) __arm_streaming {
        const std::size_t column = col + I * lanes;
        const svbool_t pg = svwhilelt_b32_u64(column, D);
        const std::size_t offset = column < D ? column : D;
        svfloat32_t s = svmul_n_f32_x(pg, svld1_f32(pg, row.data() + offset), alpha);
        s = svmla_f32_m(pg, s, svget4_f32(residual, I), key);
        svst1_f32(pg, row.data() + offset, s);
        y = svset4_f32(y, I, svmla_f32_m(pg, svget4_f32(y, I), s, query));
        if constexpr (I + 1 < B)
            return update_row<I + 1>(y, row, residual, col, lanes, alpha, key, query);
        return y;
    }


    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat32x4_t make_residual(svfloat32x4_t acc, const vector_type& v,
                                             std::size_t col, std::size_t lanes,
                                             float alpha, float beta) __arm_streaming {
        const std::size_t column = col + I * lanes;
        const svbool_t pg = svwhilelt_b32_u64(column, D);
        const std::size_t offset = column < D ? column : D;
        const svfloat32_t prediction = svmul_n_f32_x(pg, svget4_f32(acc, I), alpha);
        const svfloat32_t error = svsub_f32_x(pg, svld1_f32(pg, v.data() + offset), prediction);
        acc = svset4_f32(acc, I, svmul_n_f32_x(pg, error, beta));
        if constexpr (I + 1 < B)
            return make_residual<I + 1>(acc, v, col, lanes, alpha, beta);
        return acc;
    }


    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline void store_output(vector_type& output, svfloat32x4_t y,
                                    std::size_t col, std::size_t lanes) __arm_streaming {
        const std::size_t column = col + I * lanes;
        const svbool_t pg = svwhilelt_b32_u64(column, D);
        const std::size_t offset = column < D ? column : D;
        svst1_f32(pg, output.data() + offset, svget4_f32(y, I));
        if constexpr (I + 1 < B)
            store_output<I + 1>(output, y, col, lanes);
    }


public:
    // B adjacent vectors share each key/query broadcast and row-loop iteration.
    // Full two/four-vector blocks use SME2 grouped loads when available.
    // Tails and SME1 builds retain predicated single-vector loads.
    __arm_locally_streaming
    __attribute__((noinline))
    void update(const vector_type& q, const vector_type& k,
                const vector_type& v, float alpha, float beta, vector_type& output) {
        update_streaming(q, k, v, alpha, beta, output);
    }

    // Streaming entry point for batches; does not use ZA.
    __attribute__((always_inline))
    void update_streaming(const vector_type& q, const vector_type& k,
                          const vector_type& v, float alpha, float beta,
                          vector_type& output) __arm_streaming {
        const std::size_t lanes = svcntw();
        for (std::size_t col = 0; col < D; col += B * lanes) {
#if defined(__ARM_FEATURE_SME2)
            if constexpr (B == 2 || B == 4) {
                // Dispatch once per block, not inside the state-row loops.
                if (D - col >= B * lanes) {
                    detail::sme_full_block<D, B>::update(
                        state, q, k, v, alpha, beta, output, col, lanes);
                    continue;
                }
            }
#endif
            const svfloat32_t zero = svdup_f32(0.0f);
            svfloat32x4_t residual = svcreate4_f32(zero, zero, zero, zero);
            for (std::size_t row = 0; row < D; ++row)
                residual = predict_row(residual, state[row], col, lanes, svdup_f32(k[row]));
            residual = make_residual(residual, v, col, lanes, alpha, beta);

            svfloat32x4_t y = svcreate4_f32(zero, zero, zero, zero);
            for (std::size_t row = 0; row < D; ++row) {
                const svfloat32_t key = svdup_f32(k[row]);
                const svfloat32_t query = svdup_f32(q[row]);
                y = update_row(y, state[row], residual, col, lanes, alpha, key, query);
            }
            store_output(output, y, col, lanes);
        }
    }
};

namespace detail {

// Matrix batches create one private ZA context for all heads.
template <class Head>
__arm_new("za")
void matrix_batch(Head* heads, const typename Head::vector_type* q,
                  const typename Head::vector_type* k, const typename Head::vector_type* v,
                  const typename Head::scalar_type* alpha, const typename Head::scalar_type* beta,
                  typename Head::vector_type* output, std::size_t count) __arm_streaming {
    for (std::size_t i = 0; i < count; ++i)
        heads[i].update_streaming(q[i], k[i], v[i], alpha[i], beta[i], output[i]);
}

template <class Head>
__arm_locally_streaming
__attribute__((noinline))
void streaming_batch(Head* heads, const typename Head::vector_type* q,
                     const typename Head::vector_type* k, const typename Head::vector_type* v,
                     const typename Head::scalar_type* alpha, const typename Head::scalar_type* beta,
                     typename Head::vector_type* output, std::size_t count) {
    if constexpr (Head::uses_za) {
        matrix_batch(heads, q, k, v, alpha, beta, output, count);
    } else {
        for (std::size_t i = 0; i < count; ++i)
            heads[i].update_streaming(q[i], k[i], v[i], alpha[i], beta[i], output[i]);
    }
}

} // namespace detail
#endif

// Process one token for each of count independent heads, in array order.
// Every pointer must address count elements. State and output arrays must not
// overlap any input array or each other; the read-only input arrays may overlap.
// With count=0, pointers may be null; the function returns without accessing them.
// SME backends enter streaming mode once; vector backends do not allocate ZA.
// Matrix batches allocate ZA once. No prefetch is issued automatically.
template <class Head>
void update_batch(Head* heads, const typename Head::vector_type* q,
                  const typename Head::vector_type* k, const typename Head::vector_type* v,
                  const typename Head::scalar_type* alpha, const typename Head::scalar_type* beta,
                  typename Head::vector_type* output, std::size_t count) {
    if (count == 0)
        return;
#if defined(__ARM_FEATURE_SME)
    if constexpr (Head::uses_sme) {
        detail::streaming_batch(heads, q, k, v, alpha, beta, output, count);
    } else
#endif
    {
        for (std::size_t i = 0; i < count; ++i)
            heads[i].update(q[i], k[i], v[i], alpha[i], beta[i], output[i]);
    }
}

} // namespace gated_delta
