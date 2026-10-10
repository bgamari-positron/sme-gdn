#pragma once

#include <array>
#include <cstddef>

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif

namespace gated_delta {

// One token, one square head. State uses [key][value] row-major order.
// r = beta * (v - alpha * S^T k)
// S = alpha * S + k r^T; output = S^T q
// Inputs are finite FP32 values. The caller supplies scalar gates,
// key normalization, and query scaling. State and output must not overlap
// each other or q, k, v. The three read-only inputs may overlap.
template <std::size_t D>
struct gdn_head {
    static_assert(D > 0, "A head must have at least one dimension");
    using vector_type = std::array<float, D>;
    using state_type = std::array<vector_type, D>;

    state_type state{};

    void update_reference(const vector_type& q, const vector_type& k,
                          const vector_type& v, float alpha, float beta,
                          vector_type& output) {
        for (std::size_t j = 0; j < D; ++j) {
            float prediction = 0.0f;
            for (std::size_t i = 0; i < D; ++i)
                prediction += k[i] * state[i][j];
            const float residual = beta * (v[j] - alpha * prediction);
            float y = 0.0f;
            for (std::size_t i = 0; i < D; ++i) {
                state[i][j] = alpha * state[i][j] + k[i] * residual;
                y += q[i] * state[i][j];
            }
            output[j] = y;
        }
    }

#if defined(__ARM_FEATURE_SME)
    // Requires SME FP32 outer products and OS support for streaming mode/ZA.
    // The compiler enters streaming mode and manages private ZA state.
    // ponytail: one ZA tile and two state reads; tune residency/prefetch only
    // after measuring this kernel on the target CPU or CME.
    __arm_locally_streaming __arm_new("za")
    void update_sme(const vector_type& q, const vector_type& k,
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
#endif
};

} // namespace gated_delta
