#pragma once

#include <cstdint>

namespace ri {

// B3 naive operators: single-threaded, F32 compute throughout, plain loops.
// These are the semantic reference; B5/B6 replace the hot ones (matvec)
// behind the same signatures.

// y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
void rmsnorm(const float* x, const float* w, float* y, int n, float eps);

// y[j] = sum_i f16_to_f32(w[j*in + i]) * x[i]   (GGUF layout: ne[0]=in contiguous)
void matvec_f16(const uint16_t* w, const float* x, float* y, int in, int out);

// In-place rotary (NeoX rotate-half layout) for n_heads heads of head_dim,
// all at the same absolute position: pairs (i, i + head_dim/2) rotate by
// pos * theta^(-2i/head_dim).
void rope_neox(float* x, int n_heads, int head_dim, int pos, float theta);

// In-place softmax with max subtraction (F32).
void softmax(float* x, int n);

float silu(float x);

}  // namespace ri
