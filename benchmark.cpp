#include "gated_delta.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr std::size_t dimension = 128;
constexpr std::size_t input_count = 16;
constexpr std::size_t warmup_updates = 64;

template <class Scalar>
struct token {
    std::array<Scalar, dimension> q, k, v;
};

std::array<token<float>, input_count> make_inputs() {
    std::array<token<float>, input_count> inputs{};
    std::mt19937 random(42);
    std::uniform_real_distribution<float> sample(-1.0f, 1.0f);
    for (auto& input : inputs) {
        double norm = 0;
        for (std::size_t i = 0; i < dimension; ++i) {
            input.q[i] = sample(random) / std::sqrt(float(dimension));
            input.k[i] = sample(random);
            input.v[i] = sample(random);
            norm += double(input.k[i]) * input.k[i];
        }
        for (float& value : input.k)
            value = static_cast<float>(value / std::sqrt(norm));
    }
    return inputs;
}

// Keep state writes observable to the compiler without adding CPU instructions.
inline void observe(const void* state, const void* output) {
    asm volatile("" : : "g"(state), "g"(output) : "memory");
}

template <class Backend>
void benchmark(const char* name, const std::array<token<float>, input_count>& source,
               std::size_t updates, std::size_t trials) {
    using head_type = gated_delta::gdn_head<dimension, Backend>;
    using scalar_type = typename head_type::scalar_type;
    using clock = std::chrono::steady_clock;
    // Convert the same FP32 input values for every backend, including double.
    alignas(64) std::array<token<scalar_type>, input_count> inputs{};
    for (std::size_t t = 0; t < input_count; ++t)
        for (std::size_t i = 0; i < dimension; ++i) {
            inputs[t].q[i] = source[t].q[i];
            inputs[t].k[i] = source[t].k[i];
            inputs[t].v[i] = source[t].v[i];
        }
    std::vector<double> times;
    times.reserve(trials);
    double checksum = 0;
    for (std::size_t trial = 0; trial < trials; ++trial) {
        alignas(64) head_type head;
        typename head_type::vector_type output{};
        const scalar_type alpha = scalar_type(0.99f);
        const scalar_type beta = scalar_type(0.6f);
        for (std::size_t i = 0; i < warmup_updates; ++i) {
            const auto& input = inputs[i % input_count];
            head.update(input.q, input.k, input.v, alpha, beta, output);
        }
        observe(&head.state, &output);
        const auto begin = clock::now();
        for (std::size_t i = 0; i < updates; ++i) {
            const auto& input = inputs[i % input_count];
            head.update(input.q, input.k, input.v, alpha, beta, output);
        }
        observe(&head.state, &output);
        const auto end = clock::now();
        times.push_back(std::chrono::duration<double, std::nano>(end - begin).count() / updates);

        // Outside the timed region. Include every state and output element.
        double digest = 0;
        for (const auto& row : head.state)
            for (scalar_type value : row)
                digest += double(value) * value;
        for (scalar_type value : output)
            digest += double(value) * value;
        if (!std::isfinite(digest)) {
            std::fprintf(stderr, "%s produced a non-finite checksum\n", name);
            std::exit(1);
        }
        checksum += digest;
    }
    std::sort(times.begin(), times.end());
    const double median = times.size() % 2 ? times[times.size() / 2] :
        (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    std::printf("%-23s %14.1f %14.1f %18.9g\n", name, median, times.front(), checksum / trials);
}

std::size_t positive_number(const char* text) {
    std::size_t result = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end || result == 0) {
        std::fprintf(stderr, "Expected a positive integer, got: %s\n", text);
        std::exit(2);
    }
    return result;
}

#if defined(__ARM_FEATURE_SME)
__arm_locally_streaming
std::size_t streaming_bytes() {
    return svcntb();
}
#endif
} // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: %s [updates-per-trial=10000] [trials=5]\n", argv[0]);
        return 2;
    }
    const std::size_t updates = argc > 1 ? positive_number(argv[1]) : 10000;
    const std::size_t trials = argc > 2 ? positive_number(argv[2]) : 5;
    const auto inputs = make_inputs();
    std::printf("D=128, %zu updates/trial, %zu trials, %zu warmup updates/trial\n",
                updates, trials, warmup_updates);
    std::puts("One persistent head per trial; 16 repeating inputs; explicit prefetch disabled.");
    std::puts("Includes per-call streaming transitions. Emulated timings are NOT C1 performance estimates.");
#if defined(__ARM_FEATURE_SME)
    std::printf("Streaming vector length: %zu bytes\n", streaming_bytes());
#if defined(__ARM_FEATURE_SME2)
    std::puts("Blocked kernels: SME2 grouped loads enabled for full B=2/4 blocks.");
#else
    std::puts("Blocked kernels: SME1 single-vector loads.");
#endif
#else
    std::puts("Scalar-only build: SME kernels are not compiled.");
#endif
    std::printf("%-23s %14s %14s %18s\n", "kernel", "median ns/op", "min ns/op", "checksum");
    benchmark<float>("float", inputs, updates, trials);
    benchmark<double>("double", inputs, updates, trials);
#if defined(__ARM_FEATURE_SME)
    benchmark<gated_delta::sme_matrix_float>("sme_matrix_float", inputs, updates, trials);
    benchmark<gated_delta::sme_vector_float>("sme_vector_float", inputs, updates, trials);
    benchmark<gated_delta::sme_blocked_float<1>>("sme_blocked_float<1>", inputs, updates, trials);
    benchmark<gated_delta::sme_blocked_float<2>>("sme_blocked_float<2>", inputs, updates, trials);
    benchmark<gated_delta::sme_blocked_float<3>>("sme_blocked_float<3>", inputs, updates, trials);
    benchmark<gated_delta::sme_blocked_float<4>>("sme_blocked_float<4>", inputs, updates, trials);
#endif
}
