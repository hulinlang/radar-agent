// B6a: AVX2 fused dequant-dot kernels (Q4_K / Q6_K) and AVX2 F16 matvec.
// Same math as the scalar reference (quants.cpp): exact dequant + F32
// activations; only the evaluation is fused into registers with FMA.
// Falls back to scalar signatures when compiled without __AVX2__.

#include "simd_avx2.h"

#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "f16.h"
#include "ops.h"
#include "quants.h"

namespace ri {

#if defined(__AVX2__)
namespace {

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline uint16_t load_f16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// 16 unsigned bytes (values 0..63) against f32 x[16]: out {Σ q·x, Σ x}.
inline void dot_u8x16(__m128i qv, const float* x, float& sum_qx, float& sum_x) {
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qv));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(qv, 8)));
    const __m256 x0 = _mm256_loadu_ps(x);
    const __m256 x1 = _mm256_loadu_ps(x + 8);
    const __m256 qx = _mm256_fmadd_ps(f1, x1, _mm256_mul_ps(f0, x0));
    sum_qx = hsum256(qx);
    sum_x = hsum256(_mm256_add_ps(x0, x1));
}

// 32-byte version for Q4_K (per-chunk scale is constant across the 32 values).
inline void dot_u8x32(__m256i qv, const float* x, float& sum_qx, float& sum_x) {
    const __m128i vl = _mm256_castsi256_si128(qv);
    const __m128i vh = _mm256_extracti128_si256(qv, 1);
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vl));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vl, 8)));
    const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vh));
    const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vh, 8)));
    const __m256 x0 = _mm256_loadu_ps(x + 0);
    const __m256 x1 = _mm256_loadu_ps(x + 8);
    const __m256 x2 = _mm256_loadu_ps(x + 16);
    const __m256 x3 = _mm256_loadu_ps(x + 24);
    __m256 qx = _mm256_mul_ps(f0, x0);
    qx = _mm256_fmadd_ps(f1, x1, qx);
    qx = _mm256_fmadd_ps(f2, x2, qx);
    qx = _mm256_fmadd_ps(f3, x3, qx);
    sum_qx = hsum256(qx);
    sum_x = hsum256(_mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)));
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

float row_dot_q4_k_avx2(const uint8_t* row, const float* x, int in) {
    float acc = 0.0f;
    const int nb = in / 256;
    for (int b = 0; b < nb; ++b) {
        const float d = f16_to_f32(load_f16(row));
        const float dmin = f16_to_f32(load_f16(row + 2));
        const uint8_t* scales = row + 4;
        const uint8_t* qs = row + 16;
        const __m256i nib = _mm256_set1_epi8(0x0F);
        int is = 0;
        for (int g = 0; g < 4; ++g) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
            const __m256i lo = _mm256_and_si256(v, nib);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nib);
            float qx, sx;
            dot_u8x32(lo, x, qx, sx);
            acc += d1 * qx - m1 * sx;
            dot_u8x32(hi, x + 32, qx, sx);
            acc += d2 * qx - m2 * sx;
            qs += 32;
            x += 64;
            is += 2;
        }
        row += 144;
    }
    return acc;
}

float row_dot_q6_k_avx2(const uint8_t* row, const float* x, int in) {
    float acc = 0.0f;
    const int nb = in / 256;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* ql = row;
        const uint8_t* qh = row + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(row + 192);
        const float d = f16_to_f32(load_f16(row + 208));
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i m15 = _mm256_set1_epi8(0x0F);
        for (int h = 0; h < 2; ++h) {  // two 128-value halves per super-block
            const __m256i qll = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i qlh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i qhh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            const __m256i qv[4] = {
                _mm256_or_si256(_mm256_and_si256(qll, m15),
                                _mm256_slli_epi16(_mm256_and_si256(qhh, m3), 4)),
                _mm256_or_si256(_mm256_and_si256(qlh, m15),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhh, 2), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qll, 4), m15),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhh, 4), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlh, 4), m15),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhh, 6), m3), 4)),
            };
            // per-16-value scales: chunk g uses sc[2g] (first 16) and sc[2g+1] (second 16)
            for (int g = 0; g < 4; ++g) {
                const float* xx = x + 32 * g;
                float qx, sx;
                dot_u8x16(_mm256_castsi256_si128(qv[g]), xx, qx, sx);
                acc += d * sc[2 * g] * (qx - 32.0f * sx);
                dot_u8x16(_mm256_extracti128_si256(qv[g], 1), xx + 16, qx, sx);
                acc += d * sc[2 * g + 1] * (qx - 32.0f * sx);
            }
            ql += 64;
            qh += 32;
            sc += 8;
            x += 128;
        }
        row += 210;
    }
    return acc;
}

}  // namespace
#endif  // __AVX2__

bool have_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

void matvec_f16_avx2(const uint16_t* w, const float* x, float* y, int in, int out) {
#if defined(__AVX2__)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const uint16_t* row = w + static_cast<size_t>(j) * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            const __m256 f = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i)));
            acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(x + i), acc);
        }
        float tail = 0.0f;
        for (; i < in; ++i) tail += f16_to_f32(row[i]) * x[i];
        y[j] = hsum256(acc) + tail;
    }
#else
    matvec_f16(w, x, y, in, out);
#endif
}

void matvec_q4_k_avx2(const uint8_t* w, const float* x, float* y, int in, int out) {
#if defined(__AVX2__)
    const size_t row_bytes = static_cast<size_t>(in) / 256 * 144;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        y[j] = row_dot_q4_k_avx2(w + static_cast<size_t>(j) * row_bytes, x, in);
    }
#else
    matvec_q4_k(w, x, y, in, out);
#endif
}

void matvec_q6_k_avx2(const uint8_t* w, const float* x, float* y, int in, int out) {
#if defined(__AVX2__)
    const size_t row_bytes = static_cast<size_t>(in) / 256 * 210;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        y[j] = row_dot_q6_k_avx2(w + static_cast<size_t>(j) * row_bytes, x, in);
    }
#else
    matvec_q6_k(w, x, y, in, out);
#endif
}

}  // namespace ri
