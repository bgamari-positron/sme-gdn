#include "prefill.h"

#include <algorithm>
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
constexpr std::size_t chunk_size = 64;
using vector_type = gated_delta::gdn_head<dimension>::vector_type;

struct inputs {
    std::vector<vector_type> q, k, v;
};

inputs make_inputs(std::size_t count) {
    inputs result{std::vector<vector_type>(count), std::vector<vector_type>(count),
                  std::vector<vector_type>(count)};
    std::mt19937 random(42);
    std::uniform_real_distribution<float> sample(-1, 1);
    for (std::size_t t = 0; t < count; ++t) {
        double norm = 0;
        for (std::size_t d = 0; d < dimension; ++d) {
            result.q[t][d] = sample(random) / std::sqrt(float(dimension));
            result.k[t][d] = sample(random);
            result.v[t][d] = sample(random);
            norm += double(result.k[t][d]) * result.k[t][d];
        }
        for (float& value : result.k[t])
            value = static_cast<float>(value / std::sqrt(norm));
    }
    return result;
}

// Keep all state and output writes observable without CPU instructions.
inline void observe(const void* state, const void* output) {
    asm volatile("" : : "g"(state), "g"(output) : "memory");
}

template <class Backend>
void benchmark(const char* name, const inputs& source, std::size_t trials) {
    using head_type = gated_delta::gdn_prefill<dimension, Backend>;
    using scalar_type = typename head_type::scalar_type;
    using clock = std::chrono::steady_clock;
    const std::size_t count = source.q.size();
    std::vector<typename head_type::vector_type> q(count), k(count), v(count), output(count);
    std::vector<scalar_type> alpha(count, scalar_type(0.99f)), beta(count, scalar_type(0.6f));
    // Every backend receives the same FP32 inputs, including the FP64 reference.
    for (std::size_t t = 0; t < count; ++t)
        for (std::size_t d = 0; d < dimension; ++d) {
            q[t][d] = source.q[t][d];
            k[t][d] = source.k[t][d];
            v[t][d] = source.v[t][d];
        }
    std::vector<double> times;
    times.reserve(trials);
    double checksum = 0;
    alignas(64) head_type head;
    for (std::size_t trial = 0; trial < trials; ++trial) {
        head.state = {};
        head.update(q.data(), k.data(), v.data(), alpha.data(), beta.data(),
                    output.data(), std::min(count, chunk_size));
        observe(&head.state, output.data());
        head.state = {}; // Each timed prompt starts from zero, not the warmup state.
        observe(&head.state, output.data());
        const auto begin = clock::now();
        head.update(q.data(), k.data(), v.data(), alpha.data(), beta.data(), output.data(), count);
        observe(&head.state, output.data());
        const auto end = clock::now();
        times.push_back(std::chrono::duration<double, std::nano>(end - begin).count() / count);

        // Outside timing: include the entire output sequence and final state.
        double digest = 0;
        for (const auto& row : head.state)
            for (scalar_type value : row)
                digest += double(value) * value;
        for (const auto& token : output)
            for (scalar_type value : token)
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
    std::printf("%-26s %15.1f %15.1f %15.1f %18.9g\n", name, median, times.front(),
                median > 0 ? 1e9 / median : 0, checksum / trials);
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
__arm_locally_streaming __attribute__((noinline))
std::size_t streaming_bytes() {
    return svcntb();
}
#endif
} // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: %s [tokens-per-prompt=4096] [trials=5]\n", argv[0]);
        return 2;
    }
    const std::size_t count = argc > 1 ? positive_number(argv[1]) : 4096;
    const std::size_t trials = argc > 2 ? positive_number(argv[2]) : 5;
    const auto source = make_inputs(count);
    std::printf("D=%zu, chunk=%zu, %zu tokens/prompt, %zu trials, %zu warmup tokens/trial\n",
                dimension, chunk_size, count, trials, std::min(count, chunk_size));
    std::puts("One whole-prompt call per trial, starting from zero state; all token outputs retained.");
    std::puts("Allocation, input generation, warmup, state reset, and checksum are outside timing.");
    std::puts("Includes packing, WY formation, state updates, and streaming transitions; no explicit prefetch.");
    std::puts("Emulated timings are NOT C1 performance estimates. Run make check for correctness.");
#if defined(__ARM_FEATURE_SME)
    std::printf("Streaming vector length: %zu bytes\n", streaming_bytes());
#else
    std::puts("Scalar-only build: SME kernels are not compiled.");
#endif
    std::printf("%-26s %15s %15s %15s %18s\n", "kernel", "median ns/token", "min ns/token",
                "tokens/s", "checksum");
    using namespace gated_delta;
    benchmark<reference_prefill<float>>("reference_float", source, trials);
    benchmark<reference_prefill<double>>("reference_double", source, trials);
    benchmark<rowwise_prefill<chunk_size, float>>("rowwise_float", source, trials);
#if defined(__ARM_FEATURE_SME)
    benchmark<reference_prefill<sme_blocked_float<4>>>("reference_sme_vector_B4", source, trials);
    benchmark<sme_rowwise_prefill<chunk_size>>("sme_rowwise", source, trials);
    benchmark<sme_blocked_prefill<chunk_size, 8>>("sme_blocked_B8", source, trials);
    benchmark<sme_blocked_prefill<chunk_size, 16>>("sme_blocked_B16", source, trials);
    benchmark<sme_blocked_prefill<chunk_size, 32>>("sme_blocked_B32", source, trials);
#endif
}
