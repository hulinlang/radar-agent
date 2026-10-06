#pragma once

#include <cstdint>

namespace ri {

// Q4_K / Q6_K dequantization and fused matvec, following the ggml bit-level
// definitions (tools/llama.cpp/ggml/src/ggml-quants.c). Super-block = 256 values.
//
// Q4_K block layout (144 B): f16 d, f16 dmin, uint8 scales[12], uint8 qs[128].
//   8 sub-blocks of 32 values; per sub-block j: w = d*sc_j * q - dmin*min_j,
//   scales/mins are 6-bit, packed with the split layout of get_scale_min_k4.
// Q6_K block layout (210 B): uint8 ql[128], uint8 qh[64], int8 scales[16], f16 d.
//   q = (ql low/high nibble | qh 2 bits) - 32; w = d * scales[j] * q.

void dequantize_row_q4_K(const uint8_t* x, float* y, int64_t n);
void dequantize_row_q6_K(const uint8_t* x, float* y, int64_t n);

// y[j] = dot(dequant(row j of w), x); w holds `out` rows of `in` values each.
// OpenMP row-parallel with per-thread scratch.
void matvec_q4_k(const uint8_t* w, const float* x, float* y, int in, int out);
void matvec_q6_k(const uint8_t* w, const float* x, float* y, int in, int out);

}  // namespace ri
