#pragma once

#include "gated_delta.h"

namespace gated_delta {

// Selects sequential prefill through an existing generation backend.
template <class GenerationBackend = float>
struct reference_prefill {};

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

} // namespace gated_delta
