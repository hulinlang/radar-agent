#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "int8_act.h"

// B7: Q4_K / Q6_K on CUDA. Raw quantized bytes live in VRAM; these launchers
// wrap the handwritten fused dequant-dot gemv kernels and the dequant-only
// kernels used by the unit test. Numerics: exact dequant + F32 dot (same
// expressions as the CPU reference in quants.cpp — no activation quantization).
//
// B11 adds the int8-activation route (docs/34): same Q4_K weights, but the
// activations are quantized to 32-value symmetric int8 groups and multiplied
// with __dp4a. Only the prefill multi-column path uses it; the decode kernel
// above is deliberately untouched.

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

// ---------------- B11: int8 activations x Q4_K (prefill only) ----------------
//
// x[ncol][k] fp32 -> qs[ncol][k] int8 + ds[ncol][k/32]; mirrors llama.cpp CUDA
// quantize_q8_1 (fp32 scale to quantize, fp16 scale to multiply).
void quantize_q8_1(const float* x, int8_t* qs, Q8_1Ds* ds, int k, int ncol,
                   cudaStream_t stream = nullptr);

// y[c][out] = W[out][in] . dequant_int8(qs[c], ds[c]).
// ncol must be <= 16; every column up to 16 must be quantized, because the
// kernel is a fixed 2-rows x 16-columns warp tile (padding columns are computed
// and ignored). Bit-exact with ri::matvec_q4_k_q8_1_host, which is bit-exact
// with cpp/tests/ref_q4_int8.py.
void gemv_q4k_q8_1(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                   int in, int out, int ncol, cudaStream_t stream = nullptr);

}  // namespace gpuq
}  // namespace ri
