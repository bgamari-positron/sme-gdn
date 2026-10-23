#include "fp16.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace gated_delta;
static double maximum_error = 0;
static std::size_t current_dimension = 0, current_chunk = 0;

// Compare to FP64 recurrence on the same quantized inputs, not FP32 inputs.
// This bound covers these normalized-key test sequences, not arbitrary models.
void check_half(float16 got, double expected, const char* name, const char* field) {
    const double error = std::abs(double(got) - expected);
    maximum_error = std::max(maximum_error, error);
    if (!std::isfinite(double(got)) || !std::isfinite(expected) || error > 0.004 + 0.03 * std::abs(expected)) {
        std::fprintf(stderr, "%s D=%zu C=%zu %s: %.9g != %.9g (error %.9g)\n",
                     name, current_dimension, current_chunk, field, double(got), expected, error);
        std::abort();
    }
}

template <std::size_t D>
void inputs(std::mt19937& rng, std::array<float16, D>& q,
            std::array<float16, D>& k, std::array<float16, D>& v) {
    std::uniform_real_distribution<float> sample(-1, 1);
    double norm = 0;
    for (std::size_t d = 0; d < D; ++d) {
        q[d] = float16(sample(rng) / std::sqrt(float(D)));
        k[d] = float16(sample(rng));
        v[d] = float16(sample(rng));
        norm += double(k[d]) * k[d];
    }
    for (auto& x : k) x = float16(double(x) / std::sqrt(norm));
}

template <std::size_t D>
std::array<double, D> widen(const std::array<float16, D>& x) {
    std::array<double, D> result{};
    for (std::size_t i = 0; i < D; ++i) result[i] = x[i];
    return result;
}

template <std::size_t D, class Backend>
void decode(const char* name, unsigned steps = 96) {
    current_dimension = D; current_chunk = 0;
    using Head = gdn_head<D, Backend>;
    Head head;
    gdn_head<D, double> oracle;
    typename Head::vector_type q{}, k{}, v{}, output{};
    typename gdn_head<D, double>::vector_type expected{};
    std::mt19937 rng(123 + D);
    std::uniform_real_distribution<float> sample(-0.25, 0.25);
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j) {
            if (head.state[i][j] != 0) std::abort();
            oracle.state[i][j] = head.state[i][j] = float16(sample(rng));
        }
    for (unsigned t = 0; t < steps; ++t) {
        inputs(rng, q, k, v);
        const float16 alpha = t == steps / 2 ? float16(0) : t % 3 == 0 ? float16(1) : float16(0.97f);
        const float16 beta = t % 7 == 0 ? float16(0) : t % 5 == 0 ? float16(1) : float16(0.6f);
        oracle.update(widen(q), widen(k), widen(v), alpha, beta, expected);
        head.prefetch();
        head.update(q, k, v, alpha, beta, output);
        const auto predicted = head.predict(q);
        for (std::size_t j = 0; j < D; ++j) {
            check_half(output[j], expected[j], name, "decode output");
            check_half(predicted[j], expected[j], name, "prediction");
            for (std::size_t i = 0; i < D; ++i)
                check_half(head.state[i][j], oracle.state[i][j], name, "decode state");
        }
    }
    const auto old = head.state;
    head.decay(float16(0.5));
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            if (head.state[i][j] != float16(float(old[i][j]) * 0.5f)) std::abort();
}

template <class Backend>
void batch() {
    using Head = gdn_head<17, Backend>;
    std::array<Head, 3> batched{}, separate{};
    std::array<typename Head::vector_type, 3> q{}, k{}, v{}, got{}, expected{};
    std::array<float16, 3> alpha{float16(0.97), 0, 1}, beta{0, float16(0.6), 1};
    std::mt19937 rng(42);
    update_batch<Head>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    for (std::size_t count : {3u, 1u, 0u, 2u}) {
        for (std::size_t h = 0; h < 3; ++h) inputs(rng, q[h], k[h], v[h]);
        for (std::size_t h = 0; h < count; ++h)
            separate[h].update(q[h], k[h], v[h], alpha[h], beta[h], expected[h]);
        update_batch(batched.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(), got.data(), count);
        if (got != expected) std::abort();
        for (std::size_t h = 0; h < 3; ++h)
            if (batched[h].state != separate[h].state) std::abort();
    }
}

template <std::size_t D, std::size_t C, class Backend>
void prefill(const char* name) {
    current_dimension = D; current_chunk = C;
    using Head = gdn_prefill<D, Backend>;
    constexpr std::size_t count = 2 * C + 3;
    for (bool reset : {false, true}) {
        Head head, split;
        gdn_head<D, double> oracle;
        std::array<typename Head::vector_type, count> q{}, k{}, v{}, got{}, parts{};
        std::array<typename gdn_head<D, double>::vector_type, count> expected{};
        std::array<float16, count> alpha{}, beta{};
        std::mt19937 rng(567 + D + C);
        std::uniform_real_distribution<float> sample(-0.25, 0.25);
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                oracle.state[i][j] = head.state[i][j] = split.state[i][j] = float16(sample(rng));
        for (std::size_t t = 0; t < count; ++t) {
            inputs(rng, q[t], k[t], v[t]);
            alpha[t] = reset && (t == C / 2 || t == C) ? float16(0) :
                       t % 3 == 0 ? float16(1) : float16(0.97f);
            beta[t] = t % 7 == 0 ? float16(0) : t % 5 == 0 ? float16(1) : float16(0.6f);
            oracle.update(widen(q[t]), widen(k[t]), widen(v[t]), alpha[t], beta[t], expected[t]);
        }
        const auto saved = head.state;
        head.update(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
        if (saved != head.state) std::abort();
        head.update(q.data(), k.data(), v.data(), alpha.data(), beta.data(), got.data(), count);
        split.update(q.data(), k.data(), v.data(), alpha.data(), beta.data(), parts.data(), 1);
        split.update(q.data() + 1, k.data() + 1, v.data() + 1, alpha.data() + 1, beta.data() + 1, parts.data() + 1, C);
        constexpr std::size_t tail = C + 1;
        split.update(q.data() + tail, k.data() + tail, v.data() + tail,
                     alpha.data() + tail, beta.data() + tail, parts.data() + tail, count - tail);
        for (std::size_t t = 0; t < count; ++t)
            for (std::size_t d = 0; d < D; ++d) {
                check_half(got[t][d], expected[t][d], name, "prefill output");
                check_half(parts[t][d], expected[t][d], name, "split output");
            }
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j) {
                check_half(head.state[i][j], oracle.state[i][j], name, "prefill state");
                check_half(split.state[i][j], oracle.state[i][j], name, "split state");
            }
    }
}

template <class Backend>
void decode_suite(const char* name) {
    maximum_error = 0;
    decode<1, Backend>(name); decode<17, Backend>(name); decode<128, Backend>(name);
    // Full B=4 grouped loads plus a tail even at the largest tested SVL.
    decode<513, Backend>(name, 3);
    batch<Backend>();
    prefill<17, 17, reference_prefill<Backend>>(name);
    std::printf("PASS: %s decode/batch/reference-prefill; max abs error %.6g\n", name, maximum_error);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    decode_suite<float16>("scalar_half");
    prefill<17, 17, rowwise_prefill<17, float16>>("portable_half");
#if defined(__ARM_FEATURE_SME)
    decode_suite<sme_matrix_half>("sme_matrix_half");
    decode_suite<sme_vector_half>("sme_vector_half");
    decode_suite<sme_blocked_half<1>>("sme_blocked_half<1>");
    decode_suite<sme_blocked_half<2>>("sme_blocked_half<2>");
    decode_suite<sme_blocked_half<3>>("sme_blocked_half<3>");
    decode_suite<sme_blocked_half<4>>("sme_blocked_half<4>");
    maximum_error = 0;
    prefill<1, 1, sme_rowwise_half_prefill<1>>("rowwise_half");
    prefill<17, 17, sme_rowwise_half_prefill<17>>("rowwise_half");
    prefill<128, 64, sme_rowwise_half_prefill<64>>("rowwise_half");
    prefill<1, 1, sme_blocked_half_prefill<1, 1>>("blocked_half");
    prefill<17, 17, sme_blocked_half_prefill<17, 1>>("blocked_half");
    prefill<17, 4, sme_blocked_half_prefill<4, 16>>("blocked_half");
    prefill<65, 33, sme_blocked_half_prefill<33, 17>>("blocked_half");
    prefill<128, 64, sme_blocked_half_prefill<64, 16>>("blocked_half");
    prefill<128, 64, sme_blocked_half_prefill<64, 8>>("blocked_half_B8");
    prefill<128, 64, sme_blocked_half_prefill<64, 32>>("blocked_half_B32");
    prefill<17, 17, sme_blocked_half_prefill<17, 17>>("blocked_half_full");
    prefill<17, 17, sme_blocked_joint_half_prefill<17, 1>>("joint_half");
    prefill<17, 4, sme_blocked_joint_half_prefill<4, 16>>("joint_half");
    prefill<65, 33, sme_blocked_joint_half_prefill<33, 17>>("joint_half");
    prefill<128, 64, sme_blocked_joint_half_prefill<64, 16>>("joint_half");
    prefill<128, 64, sme_blocked_joint_half_prefill<64, 8>>("joint_half_B8");
    prefill<128, 64, sme_blocked_joint_half_prefill<64, 32>>("joint_half_B32");
    prefill<17, 17, sme_blocked_joint_half_prefill<17, 17>>("joint_half_full");
    std::printf("PASS: FP16 SME WY prefill; max abs error %.6g\n", maximum_error);
#endif
}
