#pragma once

#include "prefill.h"
#include <cmath>

namespace gated_delta {

// Arm intrinsics use __fp16 storage. _Float16 supplies the host reference type.
#if defined(__aarch64__)
using float16 = __fp16;
#else
using float16 = _Float16;
#endif

// All FP16 backends store vectors, gates, outputs, and recurrent state in FP16.
// Vector operations round to FP16. Matrix products accumulate in FP32 and round
// on stores. Finite FP16 inputs can still overflow intermediates or lose accuracy.
struct sme_matrix_half {};
struct sme_vector_half {};
template <std::size_t B>
struct sme_blocked_half {
    static_assert(B >= 1 && B <= 4, "A block must contain 1 through 4 streaming vectors");
};
template <std::size_t C = 64> struct sme_rowwise_half_prefill {};
template <std::size_t C = 64, std::size_t B = 16> struct sme_blocked_half_prefill {};
template <std::size_t C = 64, std::size_t B = 16> struct sme_blocked_joint_half_prefill {};

// A scalar FP16 reference, including explicit rounding between operations.
template <std::size_t D>
struct gdn_head<D, float16> {
    static_assert(D > 0, "A head must have at least one dimension");
    using scalar_type = float16;
    using vector_type = std::array<float16, D>;
    using state_type = std::array<vector_type, D>;
    static constexpr bool uses_sme = false, uses_za = false;
    state_type state{};
    vector_type predict(const vector_type& key) const {
        vector_type result{};
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                result[j] = float16(std::fma(double(key[i]), double(state[i][j]), double(result[j])));
        return result;
    }
    void decay(float16 alpha) {
        for (auto& row : state)
            for (auto& value : row)
                value = float16(float(value) * float(alpha));
    }
    void prefetch() const noexcept {}
    void update(const vector_type& q, const vector_type& k, const vector_type& v,
                float16 alpha, float16 beta, vector_type& output) {
        decay(alpha);
        auto residual = predict(k);
        for (std::size_t j = 0; j < D; ++j) {
            const float16 error = float16(float(v[j]) - float(residual[j]));
            residual[j] = float16(float(beta) * float(error));
        }
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < D; ++j)
                state[i][j] = float16(std::fma(double(k[i]), double(residual[j]), double(state[i][j])));
        output = predict(q);
    }
};

template <std::size_t D> struct gdn_head<D, sme_matrix_half>;
template <std::size_t D> struct gdn_head<D, sme_vector_half>;
template <std::size_t D, std::size_t B> struct gdn_head<D, sme_blocked_half<B>>;

#if defined(__ARM_FEATURE_SME)
namespace detail {

// SVE conversions use one half in each 32-bit lane; storage uses packed halves.
__attribute__((always_inline))
inline svfloat32_t widen_half(svbool_t pg32, svfloat16_t h) __arm_streaming {
    return svcvt_f32_f16_x(pg32, svzip1_f16(h, svdup_f16(0)));
}
__attribute__((always_inline))
inline svfloat16_t narrow_half(svbool_t pg32, svfloat32_t s) __arm_streaming {
    const svfloat16_t h = svcvt_f16_f32_x(pg32, s);
    return svuzp1_f16(h, h);
}

template <std::size_t D>
struct half_sme_base : gdn_head<D, float16> {
    using gdn_head<D, float16>::state;
    static constexpr bool uses_sme = true;
    __arm_locally_streaming __attribute__((noinline))
    void prefetch() const noexcept { prefetch_streaming(); }
    __attribute__((always_inline))
    void prefetch_streaming() const noexcept __arm_streaming {
        const auto* base = reinterpret_cast<const unsigned char*>(&state);
        for (std::size_t offset = 0; offset < sizeof(state); offset += 1024) {
            const std::size_t remaining = sizeof(state) - offset;
            const uint64_t metadata = remaining < 1024 ? remaining : 1024;
            asm volatile("rprfm pstkeep, %x[metadata], [%[address]]" :
                         : [metadata] "r" (metadata), [address] "r" (base + offset) : "memory");
        }
    }
};

template <std::size_t D, std::size_t B>
struct half_vector : half_sme_base<D> {
    static_assert(B >= 1 && B <= 4, "A block must contain 1 through 4 streaming vectors");
    using typename half_sme_base<D>::vector_type;
    using half_sme_base<D>::state;
    static constexpr bool uses_za = false;

private:
    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat16x4_t load_tail(const vector_type& row, std::size_t col,
                                         std::size_t lanes, svfloat16x4_t values) __arm_streaming {
        const std::size_t column = col + I * lanes;
        const auto pg = svwhilelt_b16_u64(column, D);
        values = svset4_f16(values, I, svld1_f16(pg, row.data() + (column < D ? column : D)));
        if constexpr (I + 1 < B) return load_tail<I + 1>(row, col, lanes, values);
        return values;
    }
    __attribute__((always_inline))
    static inline svfloat16x4_t load(const vector_type& row, std::size_t col,
                                    std::size_t lanes) __arm_streaming {
        const auto zero = svdup_f16(0);
#if defined(__ARM_FEATURE_SME2)
        if (D - col >= B * lanes) {
            if constexpr (B == 4) return svld1_f16_x4(svptrue_c16(), row.data() + col);
            if constexpr (B == 2) {
                const auto pair = svld1_f16_x2(svptrue_c16(), row.data() + col);
                return svcreate4_f16(svget2_f16(pair, 0), svget2_f16(pair, 1), zero, zero);
            }
        }
#endif
        return load_tail(row, col, lanes, svcreate4_f16(zero, zero, zero, zero));
    }
    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline void store_tail(vector_type& row, std::size_t col, std::size_t lanes,
                                  svfloat16x4_t values) __arm_streaming {
        const std::size_t column = col + I * lanes;
        svst1_f16(svwhilelt_b16_u64(column, D), row.data() + (column < D ? column : D),
                  svget4_f16(values, I));
        if constexpr (I + 1 < B) store_tail<I + 1>(row, col, lanes, values);
    }
    __attribute__((always_inline))
    static inline void store(vector_type& row, std::size_t col, std::size_t lanes,
                             svfloat16x4_t values) __arm_streaming {
#if defined(__ARM_FEATURE_SME2)
        if (D - col >= B * lanes) {
            if constexpr (B == 4) {
                svst1_f16_x4(svptrue_c16(), row.data() + col, values);
                return;
            }
            if constexpr (B == 2) {
                svst1_f16_x2(svptrue_c16(), row.data() + col,
                            svcreate2_f16(svget4_f16(values, 0), svget4_f16(values, 1)));
                return;
            }
        }
#endif
        store_tail(row, col, lanes, values);
    }
    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat16x4_t accumulate(svfloat16x4_t acc, svfloat16x4_t values,
                                           std::size_t col, std::size_t lanes,
                                           float16 key, float16 alpha) __arm_streaming {
        const auto pg = svwhilelt_b16_u64(col + I * lanes, D);
        const auto s = svmul_n_f16_x(pg, svget4_f16(values, I), alpha);
        acc = svset4_f16(acc, I, svmla_n_f16_m(pg, svget4_f16(acc, I), s, key));
        if constexpr (I + 1 < B) return accumulate<I + 1>(acc, values, col, lanes, key, alpha);
        return acc;
    }
    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat16x4_t residual(svfloat16x4_t prediction, svfloat16x4_t value,
                                         std::size_t col, std::size_t lanes, float16 beta) __arm_streaming {
        const auto pg = svwhilelt_b16_u64(col + I * lanes, D);
        const auto error = svsub_f16_x(pg, svget4_f16(value, I), svget4_f16(prediction, I));
        prediction = svset4_f16(prediction, I, svmul_n_f16_x(pg, error, beta));
        if constexpr (I + 1 < B) return residual<I + 1>(prediction, value, col, lanes, beta);
        return prediction;
    }
    template <std::size_t I = 0>
    __attribute__((always_inline))
    static inline svfloat16x4_t correct(svfloat16x4_t values, svfloat16x4_t error,
                                        std::size_t col, std::size_t lanes,
                                        float16 key, float16 alpha) __arm_streaming {
        const auto pg = svwhilelt_b16_u64(col + I * lanes, D);
        const auto decayed = svmul_n_f16_x(pg, svget4_f16(values, I), alpha);
        values = svset4_f16(values, I, svmla_n_f16_m(pg, decayed, svget4_f16(error, I), key));
        if constexpr (I + 1 < B) return correct<I + 1>(values, error, col, lanes, key, alpha);
        return values;
    }
public:
    __arm_locally_streaming __attribute__((noinline))
    void update(const vector_type& q, const vector_type& k, const vector_type& v,
                float16 alpha, float16 beta, vector_type& output) {
        update_streaming(q, k, v, alpha, beta, output);
    }
    __attribute__((always_inline))
    void update_streaming(const vector_type& q, const vector_type& k, const vector_type& v,
                          float16 alpha, float16 beta, vector_type& output) __arm_streaming {
        const std::size_t lanes = svcnth();
        const auto zero = svdup_f16(0);
        for (std::size_t col = 0; col < D; col += B * lanes) {
            auto r = svcreate4_f16(zero, zero, zero, zero);
            for (std::size_t i = 0; i < D; ++i)
                r = accumulate(r, load(state[i], col, lanes), col, lanes, k[i], alpha);
            r = residual(r, load(v, col, lanes), col, lanes, beta);
            auto y = svcreate4_f16(zero, zero, zero, zero);
            for (std::size_t i = 0; i < D; ++i) {
                const auto s = correct(load(state[i], col, lanes), r, col, lanes, k[i], alpha);
                store(state[i], col, lanes, s);
                y = accumulate(y, s, col, lanes, q[i], float16(1));
            }
            store(output, col, lanes, y);
        }
    }
};
} // namespace detail

template <std::size_t D>
struct gdn_head<D, sme_vector_half> : detail::half_vector<D, 1> {};
template <std::size_t D, std::size_t B>
struct gdn_head<D, sme_blocked_half<B>> : detail::half_vector<D, B> {};

template <std::size_t D>
struct gdn_head<D, sme_matrix_half> : detail::half_sme_base<D> {
    using typename detail::half_sme_base<D>::vector_type;
    using detail::half_sme_base<D>::state;
    static constexpr bool uses_za = true;
    __arm_locally_streaming __arm_new("za") __attribute__((noinline))
    void update(const vector_type& q, const vector_type& k, const vector_type& v,
                float16 alpha, float16 beta, vector_type& output) {
        update_streaming(q, k, v, alpha, beta, output);
    }
    __attribute__((always_inline))
    void update_streaming(const vector_type& q, const vector_type& k, const vector_type& v,
                          float16 alpha, float16 beta, vector_type& output)
        __arm_streaming __arm_inout("za") {
        const std::size_t lanes = svcntw();
        const auto zero = svdup_f16(0);
        const auto zero32 = svdup_f32(0);
        for (std::size_t col = 0; col < D; col += lanes) {
            const std::size_t width = D - col < lanes ? D - col : lanes;
            const auto cp = svwhilelt_b16_u64(0, width);
            const auto cp32 = svwhilelt_b32_u64(0, width);
            auto prediction = zero;
            for (std::size_t i = 0; i < D; ++i) {
                const auto s = svmul_n_f16_x(cp, svld1_f16(cp, state[i].data() + col), alpha);
                prediction = svmla_n_f16_m(cp, prediction, s, k[i]);
            }
            const auto error = svsub_f16_x(cp, svld1_f16(cp, v.data() + col), prediction);
            const auto residual = svmul_n_f16_x(cp, error, beta);
            auto y = zero;
            for (std::size_t row = 0; row < D; row += lanes) {
                const std::size_t height = D - row < lanes ? D - row : lanes;
                for (std::size_t i = 0; i < height; ++i) {
                    const auto s = svmul_n_f16_x(cp, svld1_f16(cp, state[row + i].data() + col), alpha);
                    svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), cp32, detail::widen_half(cp32, s));
                }
                // Widening FMOPA sums pairs. Pad the rank-one update with zeros.
                const auto keys = svld1_f16(svwhilelt_b16_u64(0, height), k.data() + row);
                svmopa_za32_f16_m(0, svwhilelt_b16_u64(0, 2 * height),
                                 svwhilelt_b16_u64(0, 2 * width),
                                 svzip1_f16(keys, zero), svzip1_f16(residual, zero));
                for (std::size_t i = 0; i < height; ++i) {
                    const auto s = detail::narrow_half(cp32, svread_hor_za32_f32_m(
                        zero32, cp32, 0, static_cast<uint32_t>(i)));
                    svst1_f16(cp, state[row + i].data() + col, s);
                    y = svmla_n_f16_m(cp, y, s, q[row + i]);
                }
            }
            svst1_f16(cp, output.data() + col, y);
        }
    }
};

namespace detail {

template <std::size_t D, std::size_t C, std::size_t B, bool Joint = false>
struct half_prefill : gdn_head<D, sme_matrix_half> {
    static_assert(C > 0, "A chunk must contain at least one token");
    using base_type = gdn_head<D, sme_matrix_half>;
    using typename base_type::vector_type;
    using base_type::state;
    using base_type::update;

private:
    // Two adjacent reduction entries feed each widening FP16 FMOPA.
    // Pair=true shares coefficient loads across U and W in ZA0.S/ZA1.S.
    template <bool Pair = false, std::size_t M, std::size_t K, std::size_t N>
    __attribute__((always_inline))
    static inline void product(const std::array<std::array<float16, M>, K>& left,
                               const std::array<std::array<float16, N>, K>& right,
                               std::array<float16, N>* dst, std::size_t rows,
                               std::size_t columns, std::size_t inner, bool subtract,
                               bool accumulate, std::size_t offset = 0,
                               const std::array<std::array<float16, N>, K>* right2 = nullptr,
                               std::array<float16, N>* dst2 = nullptr)
        __arm_streaming __arm_inout("za") {
        const std::size_t lanes = svcntw();
        const auto zero = svdup_f16(0);
        const auto zero32 = svdup_f32(0);
        for (std::size_t row = 0; row < rows; row += lanes) {
            const std::size_t height = rows - row < lanes ? rows - row : lanes;
            const auto rp = svwhilelt_b16_u64(0, height);
            for (std::size_t col = 0; col < columns; col += lanes) {
                const std::size_t width = columns - col < lanes ? columns - col : lanes;
                const auto cp = svwhilelt_b16_u64(0, width);
                const auto cp32 = svwhilelt_b32_u64(0, width);
                // Full predicates avoid GCC 15 dropping odd FP16 predicate bits.
                // Predicated loads zero-pad inputs; clear inactive ZA elements too.
                svzero_za();
                for (std::size_t i = 0; i < height; ++i) {
                    const auto initial = accumulate ?
                        widen_half(cp32, svld1_f16(cp, dst[row + i].data() + col)) : zero32;
                    svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), cp32, initial);
                    if constexpr (Pair)
                        svwrite_hor_za32_f32_m(1, static_cast<uint32_t>(i), cp32,
                            widen_half(cp32, svld1_f16(cp, dst2[row + i].data() + col)));
                }
                for (std::size_t a = 0; a < inner; a += 2) {
                    const auto l0 = svld1_f16(rp, left[a].data() + offset + row);
                    const auto l1 = a + 1 < inner ?
                        svld1_f16(rp, left[a + 1].data() + offset + row) : zero;
                    auto l = svzip1_f16(l0, l1);
                    if (subtract) l = svneg_f16_x(svptrue_b16(), l);
                    const auto r0 = svld1_f16(cp, right[a].data() + col);
                    const auto r1 = a + 1 < inner ? svld1_f16(cp, right[a + 1].data() + col) : zero;
                    svmopa_za32_f16_m(0, svptrue_b16(), svptrue_b16(), l, svzip1_f16(r0, r1));
                    if constexpr (Pair) {
                        const auto s0 = svld1_f16(cp, (*right2)[a].data() + col);
                        const auto s1 = a + 1 < inner ?
                            svld1_f16(cp, (*right2)[a + 1].data() + col) : zero;
                        svmopa_za32_f16_m(1, svptrue_b16(), svptrue_b16(), l, svzip1_f16(s0, s1));
                    }
                }
                for (std::size_t i = 0; i < height; ++i) {
                    svst1_f16(cp, dst[row + i].data() + col, narrow_half(cp32,
                        svread_hor_za32_f32_m(zero32, cp32, 0, static_cast<uint32_t>(i))));
                    if constexpr (Pair)
                        svst1_f16(cp, dst2[row + i].data() + col, narrow_half(cp32,
                            svread_hor_za32_f32_m(zero32, cp32, 1, static_cast<uint32_t>(i))));
                }
            }
        }
    }

public:
    // Same token-array contract as reference_prefill. All scratch is FP16.
    // Matrix products widen to FP32; the triangular solve uses FP16 FMAs.
    __arm_locally_streaming __arm_new("za") __attribute__((noinline))
    void update(const vector_type* q, const vector_type* k, const vector_type* v,
                const float16* alpha, const float16* beta, vector_type* output, std::size_t count) {
        if (count == 0) return;
        std::array<vector_type, C> u, w;
        std::array<std::array<float16, C>, D> kt, xt;
        std::array<std::array<float16, C>, C> gram, attention;
        std::array<float16, C> gamma;
        const std::size_t lanes = svcnth();
        for (std::size_t start = 0; start < count;) {
            const std::size_t n = count - start < C ? count - start : C;
            float16 prefix = 1;
            for (std::size_t i = 0; i < n; ++i) {
                gamma[i] = prefix = float16(float(prefix) * float(alpha[start + i]));
                for (std::size_t d = 0; d < D; ++d) {
                    kt[d][i] = k[start + i][d];
                    xt[d][i] = q[start + i][d];
                }
            }
            product(kt, kt, gram.data(), n, n, D, false, false);
            product(xt, kt, attention.data(), n, n, D, false, false);
            // Store transpose(A) above the diagonal, leaving its source intact.
            for (std::size_t i = 0; i < n; ++i) {
                float16 decay = 1;
                for (std::size_t j = i; j > 0;) {
                    decay = float16(float(decay) * float(alpha[start + j]));
                    --j;
                    const float16 gated = float16(float(beta[start + i]) * float(decay));
                    gram[j][i] = float16(float(gated) * float(gram[i][j]));
                }
            }
            // B=0 uses one diagonal block: the row-wise strategy.
            constexpr std::size_t block = B == 0 ? C : B;
            for (std::size_t begin = 0; begin < n;) {
                const std::size_t height = n - begin < block ? n - begin : block;
                const std::size_t end = begin + height;
                for (std::size_t i = begin; i < end; ++i) {
                    const std::size_t t = start + i;
                    const float16 bg = float16(float(beta[t]) * float(gamma[i]));
                    for (std::size_t d = 0; d < D; d += lanes) {
                        const auto pg = svwhilelt_b16_u64(d, D);
                        svst1_f16(pg, u[i].data() + d, svmul_n_f16_x(pg, svld1_f16(pg, v[t].data() + d), beta[t]));
                        svst1_f16(pg, w[i].data() + d, svmul_n_f16_x(pg, svld1_f16(pg, k[t].data() + d), bg));
                    }
                }
                if (begin != 0) {
                    if constexpr (Joint) {
                        product<true>(gram, u, u.data() + begin, height, D, begin, true, true,
                                      begin, &w, w.data() + begin);
                    } else {
                        product(gram, u, u.data() + begin, height, D, begin, true, true, begin);
                        product(gram, w, w.data() + begin, height, D, begin, true, true, begin);
                    }
                }
                for (std::size_t i = begin; i < end; ++i)
                    for (std::size_t d = 0; d < D; d += lanes) {
                        const auto pg = svwhilelt_b16_u64(d, D);
                        auto ui = svld1_f16(pg, u[i].data() + d);
                        auto wi = svld1_f16(pg, w[i].data() + d);
                        for (std::size_t j = i; j > begin;) {
                            --j;
                            const float16 a = float16(-float(gram[j][i]));
                            ui = svmla_n_f16_m(pg, ui, svld1_f16(pg, u[j].data() + d), a);
                            wi = svmla_n_f16_m(pg, wi, svld1_f16(pg, w[j].data() + d), a);
                        }
                        svst1_f16(pg, u[i].data() + d, ui);
                        svst1_f16(pg, w[i].data() + d, wi);
                    }
                begin = end;
            }
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < n; ++i)
                    xt[d][i] = float16(float(xt[d][i]) * float(gamma[i]));
            product(xt, state, output + start, n, D, D, false, false);
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t i = 0; i < n; ++i) xt[d][i] = w[i][d];
            product(xt, state, u.data(), n, D, D, true, true);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i + 1; j < n; ++j) gram[j][i] = 0;
                float16 decay = 1;
                for (std::size_t j = i + 1; j > 0;) {
                    --j;
                    gram[j][i] = float16(float(decay) * float(attention[i][j]));
                    decay = float16(float(decay) * float(alpha[start + j]));
                }
            }
            product(gram, u, output + start, n, D, n, false, true);
            for (auto& row : state)
                for (std::size_t d = 0; d < D; d += lanes) {
                    const auto pg = svwhilelt_b16_u64(d, D);
                    svst1_f16(pg, row.data() + d, svmul_n_f16_x(
                        pg, svld1_f16(pg, row.data() + d), gamma[n - 1]));
                }
            float16 decay = 1;
            for (std::size_t j = n; j > 0;) {
                --j;
                for (std::size_t d = 0; d < D; d += lanes) {
                    const auto pg = svwhilelt_b16_u64(d, D);
                    svst1_f16(pg, w[j].data() + d, svmul_n_f16_x(
                        pg, svld1_f16(pg, k[start + j].data() + d), decay));
                }
                decay = float16(float(decay) * float(alpha[start + j]));
            }
            product(w, u, state.data(), D, D, n, false, true);
            start += n;
        }
    }
};
} // namespace detail

template <std::size_t D, std::size_t C>
struct gdn_prefill<D, sme_rowwise_half_prefill<C>> : detail::half_prefill<D, C, 0> {};
template <std::size_t D, std::size_t C, std::size_t B>
struct gdn_prefill<D, sme_blocked_half_prefill<C, B>> : detail::half_prefill<D, C, B> {
    static_assert(B > 0, "A block must contain at least one token");
};
template <std::size_t D, std::size_t C, std::size_t B>
struct gdn_prefill<D, sme_blocked_joint_half_prefill<C, B>> : detail::half_prefill<D, C, B, true> {
    static_assert(B > 0, "A block must contain at least one token");
};
#endif
} // namespace gated_delta
