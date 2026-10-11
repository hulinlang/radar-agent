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
// ncol must be <= 16; the activation layout always carries 16 columns (the
// prefill chunk width), and columns >= ncol are computed and dropped. The tile
// shape does not change the arithmetic, so every variant is bit-exact with
// ri::matvec_q4_k_q8_1_host, which is bit-exact with cpp/tests/ref_q4_int8.py.
//
// B11 prefill tile (rows x columns per warp). Chosen by an end-to-end paired
// A/B in the engine, NOT by cpp/tests/gemv_bench.cpp (whose 200 back-to-back
// calls on one 7 MB matrix run L2-resident and prefer a different tile).
// RADAR_B11_TILE=<rw>x<nc> overrides it for that measurement.
inline constexpr int B11_TILE_RW = 2;
inline constexpr int B11_TILE_NC = 16;

void gemv_q4k_q8_1(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                   int in, int out, int ncol, cudaStream_t stream = nullptr);
// Explicit tile, for the sweep and for regression-testing a specific shape.
void gemv_q4k_q8_1_tile(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                        int in, int out, int ncol, int rw, int nc,
                        cudaStream_t stream = nullptr);

// ---------------- B12: int8 MMA (tensor core), prefill only ----------------
//
// Same inputs and same output as gemv_q4k_q8_1, but the inner loop is
// mma.sync.aligned.m16n8k32 instead of __dp4a. One MMA's K=32 is exactly one
// Q4_K sub-block, so it is bit-exact with ri::matvec_q4_k_q8_1_host (verified in
// int8-ref-test) and with the dp4a route.
//
// N is a RUNTIME parameter on purpose (docs/36 7.5): how many tokens share one
// weight load is the only real lever on weight traffic, and the same kernel
// should serve the current 16-wide chunk through a full-prompt width.
//
// `stride` is the allocated column count of the activation buffers (B11 always
// passes 16); the B-fragment load touches a whole column group, so stride must
// cover ceil(ncol / 16) * 16 columns. `out` must be a multiple of 16 and `in` a
// multiple of 256 -- checked, and a no-op (leaving y alone) otherwise.
void gemv_q4k_mma(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                  int in, int out, int ncol, int stride,
                  cudaStream_t stream = nullptr);

// ---------------- B13: the same treatment for Q6_K (m16n8k16) ----------------
//
// Q6_K's scale granularity is 16 values, so this uses `m16n8k16` (K=16 = exactly
// one half-block); the measured tensor-core ceilings are identical to K=32's
// (1023.7 vs 1023.8 MAC/SM/clk), so the smaller K costs nothing. Bit-exact with
// ri::matvec_q6_k_q8_1_host.
//
// ⭐ The A operand is built SIGNED (q6 in [-32,31]), which removes the offset term
//   `-32*sum(q8)` and with it the need for a half-block sum of q8 -- which
//   Q8_1Ds does not store. docs/37 §3.3 flagged that as the project's one real
//   design decision; this makes it moot. `int8(v-32) == (v + 0x60) ^ 0x80`,
//   byte-wise, no carry (valid because v <= 63).
//
// Same `stride` contract as gemv_q4k_mma.
void gemv_q6k_mma(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                  int in, int out, int ncol, int stride,
                  cudaStream_t stream = nullptr);

}  // namespace gpuq
}  // namespace ri
