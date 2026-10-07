// B7 CUDA quantized gemv kernels. Q4_K / Q6_K super-block layouts are the
// ggml bit-level definitions (see quants.h); the dequant expressions are the
// same scalar ops as the CPU reference so the dequant-only kernels bit-match.
//
// gemv shape: warp-per-row (4 warps per 128-thread block, as gemv_f16_v2),
// warp-shuffle reduction, no shared memory. Both kernels assign lanes to
// STRIDED values (value s*32+lane in Q4_K, lane-strided groups in Q6_K) so
// the activation reads stay fully coalesced — contiguous per-lane chunks
// scatter x across 128B strides and collapse L1 throughput (B7 v1 lesson).

#include "gpu_quants.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ri {
namespace gpuq {
namespace {

constexpr int QK_K = 256;
constexpr int Q4K_BLOCK_BYTES = 144;
constexpr int Q6K_BLOCK_BYTES = 210;

// ggml get_scale_min_k4: 8 sub-block scales and mins, 6-bit each, split-packed
// into 12 bytes.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q,
                                                 uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantized value i (0..255) of one Q4_K super-block. Same scalar op order
// as dequantize_row_q4_K.
__device__ __forceinline__ float q4k_at(const uint8_t* sb, int i) {
    const float d = __half2float(*reinterpret_cast<const __half*>(sb));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
    const int j = i >> 5;           // sub-block
    const int l = i & 31;
    uint8_t sc, m;
    get_scale_min_k4(j, sb + 4, &sc, &m);
    const uint8_t byte = sb[16 + (j >> 1) * 32 + l];
    const int nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
    return d * sc * nib - dmin * m;
}

// Dequantized value i (0..255) of one Q6_K super-block. Same scalar op order
// as dequantize_row_q6_K.
__device__ __forceinline__ float q6k_at(const uint8_t* sb, int i) {
    const int nn = i >> 7;          // 128-value half
    const int r = i & 127;
    const int qq = r >> 5;          // strided group within the half
    const int l = r & 31;
    const uint8_t* ql = sb + nn * 64;
    const uint8_t* qh = sb + 128 + nn * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
    const float d = __half2float(*reinterpret_cast<const __half*>(sb + 208));
    const uint8_t b = ql[l + (qq & 1) * 32];
    const int nib = (qq < 2) ? (b & 0xF) : (b >> 4);
    const int q = (nib | (((qh[l] >> (2 * qq)) & 3) << 4)) - 32;
    return d * sc[l / 16 + 2 * qq] * q;
}

// ---------------- whole-tensor dequant (unit test) ----------------

__global__ void dequant_q4k_kernel(const uint8_t* w, float* y, int64_t n) {
    const int64_t sb = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (sb >= n / QK_K) return;
    const uint8_t* b = w + sb * Q4K_BLOCK_BYTES;
    float* out = y + sb * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = q4k_at(b, i);
}

__global__ void dequant_q6k_kernel(const uint8_t* w, float* y, int64_t n) {
    const int64_t sb = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (sb >= n / QK_K) return;
    const uint8_t* b = w + sb * Q6K_BLOCK_BYTES;
    float* out = y + sb * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = q6k_at(b, i);
}

// ---------------- fused dequant-dot gemv ----------------

// Q4_K v7: warp per row. Per-row prologue: the warp cooperatively decodes ALL
// (d*sc, dmin*m) pairs into shared memory (nb*8 pairs, ≤ 24 sb x 8 = 192 per
// warp; each lane decodes a few pairs ONCE — v4 re-decoded all 8 per lane per
// super-block, ~64 issue slots/sb, which made the kernel issue-bound; the
// shuffle-broadcast variant lost because SM89 shuffle throughput is ~1/4 of
// FMA). Main loop per super-block: 4 coalesced qs byte loads, 8 smem
// broadcast loads, all activation reads fully coalesced 128B.
// (History: v1 lane-per-sub-block scattered x across 128B strides; v2 32B/step
// too little MLP — 49-66 GB/s; v3/v4 1-2 sb/step with per-lane decode —
// 105-166 GB/s. Q6_K's 96B/step shape reaches 168-232 GB/s on the same grids.)
__global__ void gemv_q4k_kernel(const uint8_t* w, const float* x, float* y,
                                int in, int out) {
    constexpr int MAX_NB = 24;  // in <= 6144
    // interleaved (dsc, dm) so the main loop reads one float2 per sub-block.
    __shared__ float2 sdmsc[4][MAX_NB * 8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 4 + warp;
    if (row >= out) return;
    const uint8_t* wr = w + static_cast<size_t>(row) * (in / QK_K) * Q4K_BLOCK_BYTES;
    const int nb = in / QK_K;
    float2* pm = sdmsc[warp];

    for (int idx = lane; idx < nb * 8; idx += 32) {
        const uint8_t* sb = wr + (idx >> 3) * Q4K_BLOCK_BYTES;
        const int j = idx & 7;
        const float d = __half2float(*reinterpret_cast<const __half*>(sb));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
        const uint32_t s0 = *reinterpret_cast<const uint32_t*>(sb + 4);
        const uint32_t s1 = *reinterpret_cast<const uint32_t*>(sb + 8);
        const uint32_t s2 = *reinterpret_cast<const uint32_t*>(sb + 12);
        uint32_t sc, mn;
        if (j < 4) {
            sc = (s0 >> (8 * j)) & 63;
            mn = (s1 >> (8 * j)) & 63;
        } else {
            const uint32_t lo = (s0 >> (8 * (j - 4))) & 0xFF;   // scales[j-4]
            const uint32_t mid = (s1 >> (8 * (j - 4))) & 0xFF;  // scales[j]
            const uint32_t hi = (s2 >> (8 * (j - 4))) & 0xFF;   // scales[j+4]
            sc = (hi & 0xF) | ((lo >> 6) << 4);
            mn = (hi >> 4) | ((mid >> 6) << 4);
        }
        pm[idx] = make_float2(d * sc, dmin * mn);
    }
    __syncwarp();

    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* sb = wr + b * Q4K_BLOCK_BYTES;
        const uint8_t qs[4] = {sb[16 + lane], sb[48 + lane], sb[80 + lane], sb[112 + lane]};
        const float* xr = x + b * QK_K + lane;
        const float2* pm_b = pm + b * 8;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            // sub-block j=m; its packed byte is qs[m>>1], nibble parity m&1.
            const uint32_t nib = (qs[m >> 1] >> ((m & 1) * 4)) & 0xF;
            const float2 dmsc = pm_b[m];
            acc += (dmsc.x * nib - dmsc.y) * xr[32 * m];
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// Q6_K: warp per row; per 128-value half, lane handles the 4 values at
// l, l+32, l+64, l+96 (qq = 0..3).
__global__ void gemv_q6k_kernel(const uint8_t* w, const float* x, float* y,
                                int in, int out) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out) return;
    const uint8_t* wr = w + static_cast<size_t>(row) * (in / QK_K) * Q6K_BLOCK_BYTES;
    const int nb = in / QK_K;

    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* sb = wr + b * Q6K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(sb + 208));
        for (int nn = 0; nn < 2; ++nn) {
            const uint8_t* ql = sb + nn * 64;
            const uint8_t* qh = sb + 128 + nn * 32;
            const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
            const float* xr = x + b * QK_K + nn * 128;
            const uint32_t qlo = ql[lane], qhi = ql[lane + 32], h = qh[lane];
            const int is = lane / 16;
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
                const uint32_t b8 = (qq & 1) ? qhi : qlo;
                const int nib = (qq < 2) ? (b8 & 0xF) : (b8 >> 4);
                const int q = (nib | (((h >> (2 * qq)) & 3) << 4)) - 32;
                acc += (d * sc[is + 2 * qq] * q) * xr[qq * 32 + lane];
            }
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

__global__ void embed_row_q6k_kernel(const uint8_t* embd, float* x,
                                     const int* token_p, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const int token = *token_p;
    const uint8_t* row = embd + static_cast<size_t>(token) * (hidden / QK_K) * Q6K_BLOCK_BYTES;
    x[i] = q6k_at(row + (i / QK_K) * Q6K_BLOCK_BYTES, i % QK_K);
}

}  // namespace

void gemv_q4k(const uint8_t* w, const float* x, float* y, int in, int out,
              cudaStream_t stream) {
    gemv_q4k_kernel<<<(out + 3) / 4, 128, 0, stream>>>(w, x, y, in, out);
}

void gemv_q6k(const uint8_t* w, const float* x, float* y, int in, int out,
              cudaStream_t stream) {
    gemv_q6k_kernel<<<(out + 3) / 4, 128, 0, stream>>>(w, x, y, in, out);
}

void embed_row_q6k(const uint8_t* embd, float* x, const int* token_p, int hidden,
                   cudaStream_t stream) {
    embed_row_q6k_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(embd, x, token_p, hidden);
}

void dequant_q4k(const uint8_t* w, float* y, int64_t n) {
    const int64_t nb = n / QK_K;
    dequant_q4k_kernel<<<static_cast<unsigned>((nb + 255) / 256), 256>>>(w, y, n);
}

void dequant_q6k(const uint8_t* w, float* y, int64_t n) {
    const int64_t nb = n / QK_K;
    dequant_q6k_kernel<<<static_cast<unsigned>((nb + 255) / 256), 256>>>(w, y, n);
}

}  // namespace gpuq
}  // namespace ri
