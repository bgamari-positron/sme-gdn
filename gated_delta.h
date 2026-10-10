#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

namespace gated_delta {

// Backend tag: the SME specialization stores float, not wrapped scalar values.
struct sme_float {};

// One token, one square head. State uses [key][value] row-major order.
// r = beta * (v - alpha * S^T k)
// S = alpha * S + k r^T; output = S^T q
// Inputs are finite. The caller supplies scalar gates, key normalization,
// and query scaling. State and output must not overlap each other or q, k, v.
// The three read-only inputs may overlap.
template <std::size_t D, class Scalar = float>
struct gdn_head {
    static_assert(D > 0, "A head must have at least one dimension");
    static_assert(std::is_floating_point_v<Scalar>, "Use a floating-point scalar or an SME specialization");
    using scalar_type = Scalar;
    using vector_type = std::array<scalar_type, D>;
    using state_type = std::array<vector_type, D>;

    state_type state{};

    vector_type predict(const vector_type& key) const {
        vector_type prediction{};
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                prediction[j] += key[i] * state[i][j];
        return prediction;
    }

    void decay(scalar_type alpha) {
        for (auto& row : state)
            for (scalar_type& value : row)
                value *= alpha;
    }

    // No portable range-prefetch operation; scalar backends leave this as a hint.
    void prefetch() const noexcept {}

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
struct gdn_head<D, sme_float>;

#if defined(__ARM_FEATURE_SME)
template <std::size_t D>
struct gdn_head<D, sme_float> : gdn_head<D, float> {
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
#endif

} // namespace gated_delta
