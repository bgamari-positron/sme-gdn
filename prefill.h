#pragma once

#include "gated_delta.h"

namespace gated_delta {

// Selects sequential prefill through an existing generation backend.
template <class GenerationBackend = float>
struct reference_prefill {};

// Selects chunked WY formation by row-wise forward substitution.
template <std::size_t ChunkSize = 64, class Scalar = float>
struct rowwise_prefill {
    static_assert(ChunkSize > 0, "A chunk must contain at least one token");
};

// Selects FP32 row-wise WY with streaming vectors and ZA matrix products.
template <std::size_t ChunkSize = 64>
struct sme_rowwise_prefill {
    static_assert(ChunkSize > 0, "A chunk must contain at least one token");
};

// Selects blocked forward substitution. BlockSize counts tokens, not lanes.
template <std::size_t ChunkSize = 64, std::size_t BlockSize = 16>
struct sme_blocked_prefill {
    static_assert(ChunkSize > 0 && BlockSize > 0, "Chunk and block sizes must be positive");
};

// Selects blocked substitution with joint U/W updates in two ZA tiles.
template <std::size_t ChunkSize = 64, std::size_t BlockSize = 16>
struct sme_blocked_joint_prefill {
    static_assert(ChunkSize > 0 && BlockSize > 0, "Chunk and block sizes must be positive");
};

// Stores one head's state between prefill calls. Backend selects the strategy.
// Specializations expose scalar_type, vector_type, state_type, state, and update().
template <std::size_t D, class Backend = reference_prefill<>>
struct gdn_prefill;

template <std::size_t D, class GenerationBackend>
struct gdn_prefill<D, reference_prefill<GenerationBackend>>
    : gdn_head<D, GenerationBackend> {
    using base_type = gdn_head<D, GenerationBackend>;
    using typename base_type::scalar_type;
    using typename base_type::vector_type;
    using base_type::update;

    // Process count consecutive tokens for this head, not independent heads.
    // Each pointer addresses count elements. Gates and vectors are per-token.
    // State and output must not overlap each other or any input array.
    // Read-only inputs may overlap. Input requirements match gdn_head::update().
    // With count=0, pointers may be null and state remains unchanged.
    // Output includes each token's update; state contains the final token's state.
    void update(const vector_type* q, const vector_type* k, const vector_type* v,
                const scalar_type* alpha, const scalar_type* beta,
                vector_type* output, std::size_t count) {
        // ponytail: use existing per-token calls; streaming-region reuse is deferred.
        for (std::size_t i = 0; i < count; ++i)
            base_type::update(q[i], k[i], v[i], alpha[i], beta[i], output[i]);
    }
};

template <std::size_t D, std::size_t C, class Scalar>
struct gdn_prefill<D, rowwise_prefill<C, Scalar>> : gdn_head<D, Scalar> {
    static_assert(C > 0, "A chunk must contain at least one token");
    using base_type = gdn_head<D, Scalar>;
    using typename base_type::scalar_type;
    using typename base_type::vector_type;
    using base_type::state;
    using base_type::update;

    // Same token-array and non-overlap contract as reference_prefill.
    // ponytail: portable FP arithmetic first; no matrix kernels or packed scratch.
    void update(const vector_type* q, const vector_type* k, const vector_type* v,
                const scalar_type* alpha, const scalar_type* beta,
                vector_type* output, std::size_t count) {
        if (count == 0)
            return;
        // O(C*D) stack scratch. No C-by-C Gram or triangular matrix is stored.
        std::array<vector_type, C> u, w;
        std::array<scalar_type, C> gamma;
        for (std::size_t start = 0; start < count;) {
            const std::size_t n = count - start < C ? count - start : C;
            scalar_type prefix = 1;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t t = start + i;
                gamma[i] = prefix *= alpha[t];
                for (std::size_t d = 0; d < D; ++d) {
                    u[i][d] = beta[t] * v[t][d];
                    w[i][d] = beta[t] * gamma[i] * k[t][d];
                }
                // Solve A[U,W] = [B V, B diag(gamma) K], where
                // A_ij = beta_i * dot(k_i,k_j) * product(alpha[j+1:i+1]).
                // The diagonal is one; only earlier rows enter the solve.
                scalar_type decay = 1;
                for (std::size_t j = i; j > 0;) {
                    decay *= alpha[start + j];
                    --j;
                    scalar_type gram = 0;
                    for (std::size_t d = 0; d < D; ++d)
                        gram += k[t][d] * k[start + j][d];
                    const scalar_type a = beta[t] * decay * gram;
                    for (std::size_t d = 0; d < D; ++d) {
                        u[i][d] -= a * u[j][d];
                        w[i][d] -= a * w[j][d];
                    }
                }
            }

            // All solves are complete. Reuse U for R = U - W*S_in.
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t d = 0; d < D; ++d) {
                    scalar_type prediction = 0;
                    for (std::size_t a = 0; a < D; ++a)
                        prediction += w[i][a] * state[a][d];
                    u[i][d] -= prediction;
                }

            // O = diag(gamma)*Q*S_in + ((Q*K^T) .* decay)*R.
            // Include the diagonal: each output sees its own token's correction.
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t t = start + i;
                output[t].fill(0);
                for (std::size_t a = 0; a < D; ++a)
                    for (std::size_t d = 0; d < D; ++d)
                        output[t][d] += q[t][a] * (gamma[i] * state[a][d]);
                scalar_type decay = 1;
                for (std::size_t j = i + 1; j > 0;) {
                    --j;
                    scalar_type attention = 0;
                    for (std::size_t a = 0; a < D; ++a)
                        attention += q[t][a] * k[start + j][a];
                    for (std::size_t d = 0; d < D; ++d)
                        output[t][d] += (decay * attention) * u[j][d];
                    decay *= alpha[start + j];
                }
            }

            // Do not change incoming state until every output has read it.
            base_type::decay(gamma[n - 1]);
            scalar_type decay = 1;
            for (std::size_t j = n; j > 0;) {
                --j;
                for (std::size_t a = 0; a < D; ++a)
                    for (std::size_t d = 0; d < D; ++d)
                        state[a][d] += (decay * k[start + j][a]) * u[j][d];
                decay *= alpha[start + j];
            }
            start += n;
        }
    }
};

#if defined(__ARM_FEATURE_SME)
namespace detail {

// B=0 retains row-wise formation; B>0 selects blocked substitution.
template <std::size_t D, std::size_t C, std::size_t B, bool Joint = false>
struct sme_prefill : gdn_head<D, sme_matrix_float> {
    static_assert(C > 0, "A chunk must contain at least one token");
    using base_type = gdn_head<D, sme_matrix_float>;
    using typename base_type::vector_type;
    using base_type::state;
    using base_type::update;

private:
    // dst = (accumulate ? dst : 0) + scale * transpose(left) * right.
    // Only the active rectangle is read or written. ZA0.S is scratch.
    template <std::size_t M, std::size_t K, std::size_t N>
    __attribute__((always_inline))
    static inline void product(const std::array<std::array<float, M>, K>& left,
                               const std::array<std::array<float, N>, K>& right,
                               std::array<float, N>* dst, std::size_t rows,
                               std::size_t columns, std::size_t inner,
                               float scale, bool accumulate, std::size_t row_offset = 0)
        __arm_streaming __arm_inout("za") {
        const std::size_t lanes = svcntw();
        const svfloat32_t zero = svdup_f32(0);
        for (std::size_t row = 0; row < rows; row += lanes) {
            const svbool_t rp = svwhilelt_b32_u64(row, rows);
            const std::size_t height = rows - row < lanes ? rows - row : lanes;
            for (std::size_t col = 0; col < columns; col += lanes) {
                const svbool_t cp = svwhilelt_b32_u64(col, columns);
                for (std::size_t i = 0; i < height; ++i) {
                    const svfloat32_t initial = accumulate ?
                        svld1_f32(cp, dst[row + i].data() + col) : zero;
                    svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), cp, initial);
                }
                for (std::size_t a = 0; a < inner; ++a) {
                    const svfloat32_t l = svmul_n_f32_x(
                        rp, svld1_f32(rp, left[a].data() + row_offset + row), scale);
                    svmopa_za32_f32_m(0, rp, cp, l,
                                     svld1_f32(cp, right[a].data() + col));
                }
                for (std::size_t i = 0; i < height; ++i)
                    svst1_f32(cp, dst[row + i].data() + col,
                              svread_hor_za32_f32_m(zero, cp, 0, static_cast<uint32_t>(i)));
            }
        }
    }

    // [U,W]_block -= A_block,previous * [U,W]_previous.
    // ZA0.S and ZA1.S share each coefficient load. No joint packing is needed.
    // Only rows before begin are read as sources; destination rows follow them.
    __attribute__((always_inline))
    static inline void joint_update(const std::array<std::array<float, C>, C>& at,
                                    std::array<vector_type, C>& u,
                                    std::array<vector_type, C>& w,
                                    std::size_t begin, std::size_t rows)
        __arm_streaming __arm_inout("za") {
        const std::size_t lanes = svcntw();
        const svfloat32_t zero = svdup_f32(0);
        for (std::size_t row = 0; row < rows; row += lanes) {
            const svbool_t rp = svwhilelt_b32_u64(row, rows);
            const std::size_t height = rows - row < lanes ? rows - row : lanes;
            for (std::size_t col = 0; col < D; col += lanes) {
                const svbool_t cp = svwhilelt_b32_u64(col, D);
                for (std::size_t i = 0; i < height; ++i) {
                    svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), cp,
                                         svld1_f32(cp, u[begin + row + i].data() + col));
                    svwrite_hor_za32_f32_m(1, static_cast<uint32_t>(i), cp,
                                         svld1_f32(cp, w[begin + row + i].data() + col));
                }
                for (std::size_t a = 0; a < begin; ++a) {
                    const svfloat32_t coefficient = svneg_f32_x(
                        rp, svld1_f32(rp, at[a].data() + begin + row));
                    svmopa_za32_f32_m(0, rp, cp, coefficient,
                                     svld1_f32(cp, u[a].data() + col));
                    svmopa_za32_f32_m(1, rp, cp, coefficient,
                                     svld1_f32(cp, w[a].data() + col));
                }
                for (std::size_t i = 0; i < height; ++i) {
                    svst1_f32(cp, u[begin + row + i].data() + col,
                              svread_hor_za32_f32_m(zero, cp, 0, static_cast<uint32_t>(i)));
                    svst1_f32(cp, w[begin + row + i].data() + col,
                              svread_hor_za32_f32_m(zero, cp, 1, static_cast<uint32_t>(i)));
                }
            }
        }
    }

public:
    // Same token-array contract as reference_prefill. One streaming/ZA region
    // covers all chunks. Requires SME FP32 outer products and OS ZA support.
    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void update(const vector_type* q, const vector_type* k, const vector_type* v,
                const float* alpha, const float* beta,
                vector_type* output, std::size_t count) {
        if (count == 0)
            return;
        // ponytail: dense stack scratch and scalar packing; tune on hardware.
        // Scratch holds 4*C*D + 2*C*C + C floats, independent of token count.
        std::array<vector_type, C> u, w;
        std::array<std::array<float, C>, D> kt, xt;
        std::array<std::array<float, C>, C> gram, attention;
        std::array<float, C> gamma;
        const std::size_t lanes = svcntw();
        for (std::size_t start = 0; start < count;) {
            const std::size_t n = count - start < C ? count - start : C;
            float prefix = 1;
            for (std::size_t i = 0; i < n; ++i) {
                gamma[i] = prefix *= alpha[start + i];
                for (std::size_t d = 0; d < D; ++d) {
                    kt[d][i] = k[start + i][d];
                    xt[d][i] = q[start + i][d];
                }
            }
            product(kt, kt, gram.data(), n, n, D, 1, false);
            product(xt, kt, attention.data(), n, n, D, 1, false);

            if constexpr (B == 0) {
                // Forward substitution in the same gated system as rowwise_prefill.
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t t = start + i;
                    for (std::size_t d = 0; d < D; d += lanes) {
                        const svbool_t pg = svwhilelt_b32_u64(d, D);
                        svfloat32_t ui = svmul_n_f32_x(pg, svld1_f32(pg, v[t].data() + d), beta[t]);
                        svfloat32_t wi = svmul_n_f32_x(
                            pg, svld1_f32(pg, k[t].data() + d), beta[t] * gamma[i]);
                        float decay = 1;
                        for (std::size_t j = i; j > 0;) {
                            decay *= alpha[start + j];
                            --j;
                            const float a = beta[t] * decay * gram[i][j];
                            ui = svmla_n_f32_m(pg, ui, svld1_f32(pg, u[j].data() + d), -a);
                            wi = svmla_n_f32_m(pg, wi, svld1_f32(pg, w[j].data() + d), -a);
                        }
                        svst1_f32(pg, u[i].data() + d, ui);
                        svst1_f32(pg, w[i].data() + d, wi);
                    }
                }
            } else {
                // Store transpose(A)'s strict upper triangle in Gram's unused
                // upper triangle. Source entries below the diagonal stay intact.
                for (std::size_t i = 0; i < n; ++i) {
                    float decay = 1;
                    for (std::size_t j = i; j > 0;) {
                        decay *= alpha[start + j];
                        --j;
                        gram[j][i] = beta[start + i] * decay * gram[i][j];
                    }
                }
                for (std::size_t begin = 0; begin < n;) {
                    const std::size_t height = n - begin < B ? n - begin : B;
                    const std::size_t end = begin + height;
                    for (std::size_t i = begin; i < end; ++i) {
                        const std::size_t t = start + i;
                        for (std::size_t d = 0; d < D; d += lanes) {
                            const svbool_t pg = svwhilelt_b32_u64(d, D);
                            svst1_f32(pg, u[i].data() + d, svmul_n_f32_x(
                                pg, svld1_f32(pg, v[t].data() + d), beta[t]));
                            svst1_f32(pg, w[i].data() + d, svmul_n_f32_x(
                                pg, svld1_f32(pg, k[t].data() + d), beta[t] * gamma[i]));
                        }
                    }
                    // X_block = RHS_block - A_block,previous * X_previous.
                    // Read only solved rows; destination rows do not overlap them.
                    if (begin != 0) {
                        if constexpr (Joint) {
                            joint_update(gram, u, w, begin, height);
                        } else {
                            product(gram, u, u.data() + begin, height, D, begin, -1, true, begin);
                            product(gram, w, w.data() + begin, height, D, begin, -1, true, begin);
                        }
                    }
                    // Solve the unit-lower-triangular diagonal block with vectors.
                    for (std::size_t i = begin; i < end; ++i) {
                        for (std::size_t d = 0; d < D; d += lanes) {
                            const svbool_t pg = svwhilelt_b32_u64(d, D);
                            svfloat32_t ui = svld1_f32(pg, u[i].data() + d);
                            svfloat32_t wi = svld1_f32(pg, w[i].data() + d);
                            for (std::size_t j = i; j > begin;) {
                                --j;
                                const float a = gram[j][i];
                                ui = svmla_n_f32_m(pg, ui, svld1_f32(pg, u[j].data() + d), -a);
                                wi = svmla_n_f32_m(pg, wi, svld1_f32(pg, w[j].data() + d), -a);
                            }
                            svst1_f32(pg, u[i].data() + d, ui);
                            svst1_f32(pg, w[i].data() + d, wi);
                        }
                    }
                    begin = end;
                }
            }

            // Read incoming state for output before reusing X for transpose(W).
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < n; ++i)
                    xt[d][i] *= gamma[i];
            product(xt, state, output + start, n, D, D, 1, false);
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < n; ++i)
                    xt[d][i] = w[i][d];
            product(xt, state, u.data(), n, D, D, -1, true);

            // Reuse Gram storage for transpose(masked, decayed attention).
            // Direct interval products also support alpha=0 inside a chunk.
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i + 1; j < n; ++j)
                    gram[j][i] = 0;
                float decay = 1;
                for (std::size_t j = i + 1; j > 0;) {
                    --j;
                    gram[j][i] = decay * attention[i][j];
                    decay *= alpha[start + j];
                }
            }
            product(gram, u, output + start, n, D, n, 1, true);

            // No output or residual still needs the incoming state.
            for (std::size_t a = 0; a < D; ++a)
                for (std::size_t d = 0; d < D; d += lanes) {
                    const svbool_t pg = svwhilelt_b32_u64(d, D);
                    svst1_f32(pg, state[a].data() + d, svmul_n_f32_x(
                        pg, svld1_f32(pg, state[a].data() + d), gamma[n - 1]));
                }
            float decay = 1;
            for (std::size_t j = n; j > 0;) {
                --j;
                for (std::size_t d = 0; d < D; d += lanes) {
                    const svbool_t pg = svwhilelt_b32_u64(d, D);
                    svst1_f32(pg, w[j].data() + d, svmul_n_f32_x(
                        pg, svld1_f32(pg, k[start + j].data() + d), decay));
                }
                decay *= alpha[start + j];
            }
            product(w, u, state.data(), D, D, n, 1, true);
            start += n;
        }
    }
};

} // namespace detail

template <std::size_t D, std::size_t C>
struct gdn_prefill<D, sme_rowwise_prefill<C>> : detail::sme_prefill<D, C, 0> {};

template <std::size_t D, std::size_t C, std::size_t B>
struct gdn_prefill<D, sme_blocked_prefill<C, B>> : detail::sme_prefill<D, C, B> {
    static_assert(B > 0, "A block must contain at least one token");
};

template <std::size_t D, std::size_t C, std::size_t B>
struct gdn_prefill<D, sme_blocked_joint_prefill<C, B>> : detail::sme_prefill<D, C, B, true> {
    static_assert(B > 0, "A block must contain at least one token");
};
#endif

} // namespace gated_delta
