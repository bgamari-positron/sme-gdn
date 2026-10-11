#include "gated_delta.h"

#if defined(GATED_DELTA_COMPILE_ONLY)
#if !defined(__ARM_FEATURE_SME)
#error "Compile this check for an SME target"
#endif
template struct gated_delta::gdn_head<1, gated_delta::sme_matrix_float>;
template struct gated_delta::gdn_head<17, gated_delta::sme_matrix_float>;
template struct gated_delta::gdn_head<128, gated_delta::sme_matrix_float>;
template struct gated_delta::gdn_head<1, gated_delta::sme_vector_float>;
template struct gated_delta::gdn_head<17, gated_delta::sme_vector_float>;
template struct gated_delta::gdn_head<128, gated_delta::sme_vector_float>;
template struct gated_delta::gdn_head<128, gated_delta::sme_blocked_float<1>>;
template struct gated_delta::gdn_head<128, gated_delta::sme_blocked_float<2>>;
template struct gated_delta::gdn_head<128, gated_delta::sme_blocked_float<3>>;
template struct gated_delta::gdn_head<128, gated_delta::sme_blocked_float<4>>;
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

#if defined(__ARM_FEATURE_SME)
template <class Head>
__arm_locally_streaming __attribute__((noinline))
void prefetch_batch_probe(Head* heads, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        heads[i].prefetch_streaming();
}
#endif

template <std::size_t D, class Backend>
void run_batch(const char* name) {
    using head_type = gated_delta::gdn_head<D, Backend>;
    using scalar_type = typename head_type::scalar_type;
    constexpr std::size_t capacity = 3;
    std::array<head_type, capacity> batched{}, individual{};
    std::array<typename head_type::vector_type, capacity> q{}, k{}, v{}, got{}, expected{};
    std::array<scalar_type, capacity> alpha{}, beta{};
    std::mt19937 random(123 + D);
    std::uniform_real_distribution<float> sample(-1, 1);
    gated_delta::update_batch<head_type>(nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, 0);
    for (std::size_t h = 0; h < capacity; ++h) {
        got[h].fill(-99);
        expected[h].fill(-99);
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                batched[h].state[i][j] = individual[h].state[i][j] = sample(random);
    }
    for (unsigned step = 0; step < 12; ++step) {
        const std::size_t count = step % (capacity + 1);
        for (std::size_t h = 0; h < capacity; ++h) {
            alpha[h] = (step + h) % 3 == 0 ? 0 : scalar_type(0.97);
            beta[h] = (step + h) % 4 == 0 ? 0 : scalar_type(0.6);
            for (std::size_t i = 0; i < D; ++i) {
                q[h][i] = sample(random) / std::sqrt(scalar_type(D));
                k[h][i] = sample(random) / std::sqrt(scalar_type(D));
                v[h][i] = sample(random);
            }
        }
#if defined(__ARM_FEATURE_SME)
        if constexpr (head_type::uses_sme)
            prefetch_batch_probe(batched.data(), count);
#endif
        for (std::size_t h = 0; h < count; ++h)
            individual[h].update(q[h], k[h], v[h], alpha[h], beta[h], expected[h]);
        gated_delta::update_batch(batched.data(), q.data(), k.data(), v.data(),
                                 alpha.data(), beta.data(), got.data(), count);
        for (std::size_t h = 0; h < capacity; ++h) {
            for (std::size_t j = 0; j < D; ++j) {
                check(got[h][j], expected[h][j], D, step, name, "batch output");
                for (std::size_t i = 0; i < D; ++i)
                    check(batched[h].state[i][j], individual[h].state[i][j],
                          D, step, name, "batch state");
            }
        }
    }
}

template <class Backend>
void run_dimensions(const char* name) {
    run<1, Backend>(name); run<3, Backend>(name); run<16, Backend>(name);
    run<17, Backend>(name); run<32, Backend>(name); run<65, Backend>(name);
    run<128, Backend>(name);
    run_batch<17, Backend>(name);
    run_batch<128, Backend>(name);
    std::printf("PASS: %s FP64-oracle checks and batched-versus-individual updates\n", name);
}

int main() {
    run_dimensions<float>("float");
    run_dimensions<double>("double");
#if defined(__ARM_FEATURE_SME)
    run_dimensions<gated_delta::sme_matrix_float>("sme_matrix_float");
    run_dimensions<gated_delta::sme_vector_float>("sme_vector_float");
    run_dimensions<gated_delta::sme_blocked_float<1>>("sme_blocked_float<1>");
    run_dimensions<gated_delta::sme_blocked_float<2>>("sme_blocked_float<2>");
    run_dimensions<gated_delta::sme_blocked_float<3>>("sme_blocked_float<3>");
    run_dimensions<gated_delta::sme_blocked_float<4>>("sme_blocked_float<4>");
#else
    std::puts("SME not enabled");
#endif
}
#endif
