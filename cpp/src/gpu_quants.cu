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

#include "gpu_quants.cuh"

namespace ri {
namespace gpuq {
namespace {

constexpr int QK_K = 256;
constexpr int Q4K_BLOCK_BYTES = 144;
constexpr int Q6K_BLOCK_BYTES = 210;

using detail::q4k_at;
using detail::q6k_at;

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

// defined below in the multi-column section (shared by single- and
// multi-column Q4_K kernels)
__device__ __forceinline__ void q4k_row_scales(const uint8_t* wr, int nb, int lane,
                                               float2* pm);

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
    q4k_row_scales(wr, nb, lane, pm);

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

// ---------------- B8 multi-column gemv (weights read once for NC columns) ----------------

// shared prologue: decode the row's (d*sc, dmin*m) pairs into smem pm[nb*8].
__device__ __forceinline__ void q4k_row_scales(const uint8_t* wr, int nb, int lane,
                                               float2* pm) {
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
}

// Multi-column Q4_K: warp per R consecutive rows, NC activation columns.
// Per value the dequantized wv is computed once (same expression and rounding
// as the single kernel) and FMA'd into acc[r][c]; x is loaded once per
// (column, m) and reused across the R rows — that reuse is what keeps the
// kernel DRAM-bound: R=1 collapses L1 under NC x 1KB of x traffic per 144B
// super-block (B8 bench: ncol=16 R=1 -> 19 GB/s; the x side, not the weights).
// Each column/row's accumulation order is unchanged -> bitwise parity.
template <int R, int NC>
__global__ void gemv_q4k_mc_kernel(const uint8_t* w, const float* x, float* y,
                                   int in, int out) {
    constexpr int MAX_NB = 24;  // in <= 6144
    __shared__ float2 sdmsc[4][R][MAX_NB * 8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= out) return;
    const size_t row_bytes = static_cast<size_t>(in / QK_K) * Q4K_BLOCK_BYTES;
    const int nb = in / QK_K;
    float2* pm = &sdmsc[warp][0][0];
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (row0 + r < out) q4k_row_scales(w + (row0 + r) * row_bytes, nb, lane, pm + r * MAX_NB * 8);

    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;

    for (int b = 0; b < nb; ++b) {
        uint8_t qs[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= out) break;
            const uint8_t* sb = w + (row0 + r) * row_bytes + b * Q4K_BLOCK_BYTES;
            qs[r][0] = sb[16 + lane];
            qs[r][1] = sb[48 + lane];
            qs[r][2] = sb[80 + lane];
            qs[r][3] = sb[112 + lane];
        }
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            float wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float2 dmsc = pm[r * MAX_NB * 8 + b * 8 + m];
                const uint32_t nib = (qs[r][m >> 1] >> ((m & 1) * 4)) & 0xF;
                wv[r] = dmsc.x * nib - dmsc.y;
            }
            const int off = b * QK_K + 32 * m + lane;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float xv = x[static_cast<size_t>(c) * in + off];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][c] += wv[r] * xv;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= out) break;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float a = acc[r][c];
            for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) y[static_cast<size_t>(c) * out + row0 + r] = a;
        }
    }
}

// Multi-column Q6_K: same R-rows-per-warp x-reuse structure as Q4_K mc.
template <int R, int NC>
__global__ void gemv_q6k_mc_kernel(const uint8_t* w, const float* x, float* y,
                                   int in, int out) {
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    if (row0 >= out) return;
    const size_t row_bytes = static_cast<size_t>(in / QK_K) * Q6K_BLOCK_BYTES;
    const int nb = in / QK_K;

    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
                float wv[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (row0 + r >= out) { wv[r] = 0.0f; continue; }
                    const uint8_t* sb = w + (row0 + r) * row_bytes + b * Q6K_BLOCK_BYTES;
                    const float d = __half2float(*reinterpret_cast<const __half*>(sb + 208));
                    const uint8_t* ql = sb + nn * 64;
                    const uint8_t* qh = sb + 128 + nn * 32;
                    const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
                    const uint32_t b8 = (qq & 1) ? ql[lane + 32] : ql[lane];
                    const int nib = (qq < 2) ? (b8 & 0xF) : (b8 >> 4);
                    const int q = (nib | (((qh[lane] >> (2 * qq)) & 3) << 4)) - 32;
                    wv[r] = d * sc[lane / 16 + 2 * qq] * q;
                }
                const int off = b * QK_K + nn * 128 + qq * 32 + lane;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const float xv = x[static_cast<size_t>(c) * in + off];
#pragma unroll
                    for (int r = 0; r < R; ++r) acc[r][c] += wv[r] * xv;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= out) break;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float a = acc[r][c];
            for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) y[static_cast<size_t>(c) * out + row0 + r] = a;
        }
    }
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

void gemv_q4k_mc(const uint8_t* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream) {
    if (ncol <= 1) { gemv_q4k(w, x, y, in, out, stream); return; }
    // 4 warps per block, R rows per warp
    auto grid_for = [out](int r) { return ((out + r - 1) / r + 3) / 4; };
    if (ncol <= 2) gemv_q4k_mc_kernel<4, 2><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 4) gemv_q4k_mc_kernel<4, 4><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 8) gemv_q4k_mc_kernel<4, 8><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else gemv_q4k_mc_kernel<2, 16><<<grid_for(2), 128, 0, stream>>>(w, x, y, in, out);
}

void gemv_q6k_mc(const uint8_t* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream) {
    if (ncol <= 1) { gemv_q6k(w, x, y, in, out, stream); return; }
    auto grid_for = [out](int r) { return ((out + r - 1) / r + 3) / 4; };
    if (ncol <= 2) gemv_q6k_mc_kernel<4, 2><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 4) gemv_q6k_mc_kernel<4, 4><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 8) gemv_q6k_mc_kernel<4, 8><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else gemv_q6k_mc_kernel<2, 16><<<grid_for(2), 128, 0, stream>>>(w, x, y, in, out);
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
