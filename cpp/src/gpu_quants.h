#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// B7: Q4_K / Q6_K on CUDA. Raw quantized bytes live in VRAM; these launchers
// wrap the handwritten fused dequant-dot gemv kernels and the dequant-only
// kernels used by the unit test. Numerics: exact dequant + F32 dot (same
// expressions as the CPU reference in quants.cpp — no activation quantization).

namespace ri {
namespace gpuq {

// y[j] = dot(dequant(row j of w), x); w holds `out` rows of `in` values each.
void gemv_q4k(const uint8_t* w, const float* x, float* y, int in, int out,
              cudaStream_t stream = nullptr);
void gemv_q6k(const uint8_t* w, const float* x, float* y, int in, int out,
              cudaStream_t stream = nullptr);

// B8 multi-column variants: x is `ncol` columns of `in` floats each (stride
// in), y is `ncol` columns of `out` floats (stride out). Weights are read and
// dequantized ONCE per row for all columns. Each column's accumulation order
// is identical to the single-column kernel ⇒ per-column bitwise parity.
// ncol is clamped to the largest supported template (1, 2, 4, 8, 16); x/y
// must have that many columns allocated (padding columns computed, ignored).
void gemv_q4k_mc(const uint8_t* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream = nullptr);
void gemv_q6k_mc(const uint8_t* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream = nullptr);

// F16 multi-column gemv (defined in gpu_model.cu next to its single-column
// sibling gemv_f16_v2). Same column-parity guarantee.
void gemv_f16_mc(const __half* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream = nullptr);

// x[i] = dequant(row *token_p of the Q6_K embedding matrix), i in [0, hidden).
// token is read from device memory so the decode step can be graph-captured.
void embed_row_q6k(const uint8_t* embd, float* x, const int* token_p, int hidden,
                   cudaStream_t stream = nullptr);

// Whole-tensor dequantize (unit-test path only; the forward pass never
// materializes F32 weights).
void dequant_q4k(const uint8_t* w, float* y, int64_t n);
void dequant_q6k(const uint8_t* w, float* y, int64_t n);

}  // namespace gpuq
}  // namespace ri
