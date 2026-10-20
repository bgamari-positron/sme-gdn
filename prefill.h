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

} // namespace gated_delta
