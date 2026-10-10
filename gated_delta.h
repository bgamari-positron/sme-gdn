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

#if defined(__ARM_FEATURE_SME)
template <std::size_t D>
struct gdn_head<D, sme_matrix_float> : gdn_head<D, float> {
    using base_type = gdn_head<D, float>;
    using typename base_type::vector_type;
    using base_type::state;

    // Standalone predict/decay reuse the scalar methods; update remains fused.
    // Hint the state for a later read-modify-write on the CME.
    // C1 prefetches at most 1 KiB per RPRFM. Timing remains caller-controlled.
    __arm_locally_streaming
    void prefetch() const noexcept {
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
    void update(const vector_type& q, const vector_type& k,
                    const vector_type& v, float alpha, float beta,
                    vector_type& output) {
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

    // Two state reads and one write, with no ZA allocation or slice transfers.
    __arm_locally_streaming
    void update(const vector_type& q, const vector_type& k,
                const vector_type& v, float alpha, float beta,
                vector_type& output) {
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
#endif

} // namespace gated_delta
