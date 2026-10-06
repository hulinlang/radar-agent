#include "quants.h"

#include <vector>

#include "f16.h"

namespace ri {
namespace {

constexpr int QK_K = 256;

uint16_t load_f16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// ggml get_scale_min_k4: 8 sub-block scales and mins, 6-bit each, split-packed
// into 12 bytes.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

float dot(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

}  // namespace

void dequantize_row_q4_K(const uint8_t* x, float* y, int64_t n) {
    const int64_t nb = n / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(load_f16(x));
        const float dmin = f16_to_f32(load_f16(x + 2));
        const uint8_t* scales = x + 4;
        const uint8_t* q = x + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
        x += 144;
    }
}

void dequantize_row_q6_K(const uint8_t* x, float* y, int64_t n) {
    const int64_t nb = n / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* ql = x;
        const uint8_t* qh = x + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(x + 192);
        const float d = f16_to_f32(load_f16(x + 208));
        for (int nn = 0; nn < QK_K; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        x += 210;
    }
}

void matvec_q4_k(const uint8_t* w, const float* x, float* y, int in, int out) {
    const size_t row_bytes = static_cast<size_t>(in) / QK_K * 144;
    #pragma omp parallel
    {
        std::vector<float> buf(in);
        #pragma omp for schedule(static)
        for (int j = 0; j < out; ++j) {
            dequantize_row_q4_K(w + static_cast<size_t>(j) * row_bytes, buf.data(), in);
            y[j] = dot(buf.data(), x, in);
        }
    }
}

void matvec_q6_k(const uint8_t* w, const float* x, float* y, int in, int out) {
    const size_t row_bytes = static_cast<size_t>(in) / QK_K * 210;
    #pragma omp parallel
    {
        std::vector<float> buf(in);
        #pragma omp for schedule(static)
        for (int j = 0; j < out; ++j) {
            dequantize_row_q6_K(w + static_cast<size_t>(j) * row_bytes, buf.data(), in);
            y[j] = dot(buf.data(), x, in);
        }
    }
}

}  // namespace ri
