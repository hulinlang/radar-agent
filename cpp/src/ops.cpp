#include "ops.h"

#include <cmath>

#include "f16.h"

namespace ri {

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

void matvec_f16(const uint16_t* w, const float* x, float* y, int in, int out) {
    // Row-partitioned: each row's dot product keeps its sequential accumulation
    // order, so the result is bit-identical to the single-threaded version.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out; ++j) {
        const uint16_t* row = w + static_cast<size_t>(j) * in;
        float acc = 0.0f;
        for (int i = 0; i < in; ++i) acc += f16_to_f32(row[i]) * x[i];
        y[j] = acc;
    }
}

void rope_neox(float* x, int n_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        float* v = x + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) /
                                                   static_cast<float>(head_dim));
            const float c = std::cos(pos * freq);
            const float s = std::sin(pos * freq);
            const float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace ri
