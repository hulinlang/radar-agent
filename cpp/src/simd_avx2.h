#pragma once

#include <cstdint>

namespace ri {

bool have_avx2();

// AVX2 fused variants of the scalar matvec kernels (same math, register-fused).
// Each transparently falls back to the scalar implementation without AVX2.
void matvec_f16_avx2(const uint16_t* w, const float* x, float* y, int in, int out);
void matvec_q4_k_avx2(const uint8_t* w, const float* x, float* y, int in, int out);
void matvec_q6_k_avx2(const uint8_t* w, const float* x, float* y, int in, int out);

}  // namespace ri
