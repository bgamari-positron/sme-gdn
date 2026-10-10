#include "gated_delta.h"

#if defined(GATED_DELTA_COMPILE_ONLY)
// Freestanding cross-compilation checks instantiate the actual SME body.
#if !defined(__ARM_FEATURE_SME)
#error "Compile this check for an SME target"
#endif
template struct gated_delta::gdn_head<1>;
template struct gated_delta::gdn_head<17>;
template struct gated_delta::gdn_head<128>;
#else
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void check(float got, double expected, size_t dimension, unsigned step,
                  const char* implementation, const char* field) {
    if (!std::isfinite(got) ||
        std::abs(got - expected) > 3e-5 + 3e-4 * std::abs(expected)) {
        std::fprintf(stderr, "%s D=%zu step=%u %s: %.9g != %.17g\n",
                     implementation, dimension, step, field, got, expected);
        std::abort();
    }
}

template <size_t D>
void run() {
    using head_type = gated_delta::gdn_head<D>;
    head_type reference;
    typename head_type::vector_type q{}, k{}, v{}, output{};
    std::array<std::array<double, D>, D> oracle{};
#if defined(__ARM_FEATURE_SME)
    head_type actual;
    typename head_type::vector_type sme_output{};
#endif
    std::mt19937 random(42 + D);
    std::uniform_real_distribution<float> sample(-1.0f, 1.0f);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            oracle[i][j] = reference.state[i][j] = sample(random);
#if defined(__ARM_FEATURE_SME)
            actual.state[i][j] = reference.state[i][j];
#endif
        }
    }
    for (unsigned step = 0; step < 128; ++step) {
        double norm = 0;
        for (size_t i = 0; i < D; ++i) {
            q[i] = sample(random) / std::sqrt(static_cast<float>(D));
            k[i] = sample(random);
            v[i] = sample(random);
            norm += static_cast<double>(k[i]) * k[i];
        }
        for (float& x : k)
            x = static_cast<float>(x / std::sqrt(norm));
        // Include no decay, complete decay, no correction, and full correction.
        const float alpha = step % 11 == 0 ? 0.0f :
                            step % 3 == 0 ? 1.0f : 0.97f;
        const float beta = step % 7 == 0 ? 0.0f :
                           step % 5 == 0 ? 1.0f : 0.6f;

        // Independent FP64 oracle: decay, predict, correct, then read output.
        std::array<double, D> prediction{}, expected{};
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j < D; ++j) {
                oracle[i][j] *= alpha;
                prediction[j] += k[i] * oracle[i][j];
            }
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j < D; ++j) {
                oracle[i][j] += k[i] * (beta * (v[j] - prediction[j]));
                expected[j] += q[i] * oracle[i][j];
            }

        reference.update_reference(q, k, v, alpha, beta, output);
#if defined(__ARM_FEATURE_SME)
        actual.prefetch_sme(); // Instruction smoke test, not a timing benchmark.
        actual.update_sme(q, k, v, alpha, beta, sme_output);
#endif
        for (size_t j = 0; j < D; ++j) {
            check(output[j], expected[j], D, step, "reference", "output");
#if defined(__ARM_FEATURE_SME)
            check(sme_output[j], expected[j], D, step, "SME", "output");
#endif
            for (size_t i = 0; i < D; ++i) {
                check(reference.state[i][j], oracle[i][j], D, step, "reference", "state");
#if defined(__ARM_FEATURE_SME)
                check(actual.state[i][j], oracle[i][j], D, step, "SME", "state");
#endif
            }
        }
    }
}

int main() {
    run<1>(); run<3>(); run<16>(); run<17>();
    run<32>(); run<65>(); run<128>();
#if defined(__ARM_FEATURE_SME)
    std::puts("PASS: reference and SME against FP64 oracle");
#else
    std::puts("PASS: reference against FP64 oracle; SME not enabled");
#endif
}
#endif
