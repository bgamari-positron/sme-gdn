#include "gated_delta.h"

#if defined(GATED_DELTA_COMPILE_ONLY)
#if !defined(__ARM_FEATURE_SME)
#error "Compile this check for an SME target"
#endif
template struct gated_delta::gdn_head<1, gated_delta::sme_float>;
template struct gated_delta::gdn_head<17, gated_delta::sme_float>;
template struct gated_delta::gdn_head<128, gated_delta::sme_float>;
#else
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

template <class Scalar>
static void check(Scalar got, double expected, std::size_t dimension, unsigned step,
                  const char* implementation, const char* field) {
    constexpr double absolute = std::is_same_v<Scalar, float> ? 3e-5 : 2e-12;
    constexpr double relative = std::is_same_v<Scalar, float> ? 3e-4 : 2e-12;
    if (!std::isfinite(got) ||
        std::abs(got - expected) > absolute + relative * std::abs(expected)) {
        std::fprintf(stderr, "%s D=%zu step=%u %s: %.17g != %.17g\n",
                     implementation, dimension, step, field, double(got), expected);
        std::abort();
    }
}

template <std::size_t D, class Backend>
void run(const char* name) {
    using head_type = gated_delta::gdn_head<D, Backend>;
    using scalar_type = typename head_type::scalar_type;
    head_type head;
    typename head_type::vector_type q{}, k{}, v{}, output{};
    std::array<std::array<double, D>, D> oracle{};
    std::mt19937 random(42 + D);
    std::uniform_real_distribution<scalar_type> sample(-1, 1);
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j) {
            check(head.state[i][j], 0, D, 0, name, "initial state");
            oracle[i][j] = head.state[i][j] = sample(random);
        }

    for (unsigned step = 0; step < 128; ++step) {
        double norm = 0;
        for (std::size_t i = 0; i < D; ++i) {
            q[i] = sample(random) / std::sqrt(static_cast<scalar_type>(D));
            k[i] = sample(random);
            v[i] = sample(random);
            norm += static_cast<double>(k[i]) * k[i];
        }
        for (auto& x : k)
            x = static_cast<scalar_type>(x / std::sqrt(norm));
        // Include no decay, complete decay, no correction, and full correction.
        const scalar_type alpha = step % 11 == 0 ? 0 :
                                  step % 3 == 0 ? 1 : scalar_type(0.97);
        const scalar_type beta = step % 7 == 0 ? 0 :
                                 step % 5 == 0 ? 1 : scalar_type(0.6);

        // Independent FP64 oracle; do not call the generic head implementation.
        std::array<double, D> prediction{}, expected{};
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j) {
                oracle[i][j] *= alpha;
                prediction[j] += k[i] * oracle[i][j];
            }
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j) {
                oracle[i][j] += k[i] * (beta * (v[j] - prediction[j]));
                expected[j] += q[i] * oracle[i][j];
            }

        head.prefetch(); // Instruction smoke test, not a timing benchmark.
        head.update(q, k, v, alpha, beta, output);
        const head_type& const_head = head;
        const auto predicted = const_head.predict(q);
        for (std::size_t j = 0; j < D; ++j) {
            check(output[j], expected[j], D, step, name, "output");
            check(predicted[j], expected[j], D, step, name, "prediction");
            for (std::size_t i = 0; i < D; ++i)
                check(head.state[i][j], oracle[i][j], D, step, name, "state");
        }
    }
    head.decay(scalar_type(0.5));
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            check(head.state[i][j], 0.5 * oracle[i][j], D, 128, name, "decay");
}

template <class Backend>
void run_dimensions(const char* name) {
    run<1, Backend>(name); run<3, Backend>(name); run<16, Backend>(name);
    run<17, Backend>(name); run<32, Backend>(name); run<65, Backend>(name);
    run<128, Backend>(name);
    std::printf("PASS: %s update/predict/decay/prefetch against independent FP64 oracle\n", name);
}

int main() {
    run_dimensions<float>("float");
    run_dimensions<double>("double");
#if defined(__ARM_FEATURE_SME)
    run_dimensions<gated_delta::sme_float>("sme_float");
#else
    std::puts("SME not enabled");
#endif
}
#endif
