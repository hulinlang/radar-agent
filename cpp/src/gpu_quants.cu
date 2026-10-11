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

// ---------------- B11: int8 activations x Q4_K (prefill only) ----------------
//
// Why this shape (docs/34 section 3.3): the fp32 route is *instruction-issue*
// bound -- 256 nibbles per super-block each need a shift/and/scale/multiply --
// and P11 measured it flat at ~16% of the compute roof on three prompt lengths
// (length-independent => not a memory problem). dp4a retires 4 int8 MACs per
// instruction, and the int32 dot products let the fp32 part collapse to two
// terms per sub-block instead of eight per value.
//
// Warp tile: lane = (row, column) with column fastest, so 32 lanes cover
// ROWS_PER_WARP rows x 16 columns of a prefill chunk. Each lane owns ONE output
// element and walks the whole k dimension by itself. That has three properties
// this stage needs:
//   * no shared memory and no cross-lane reduction at all (SM89 shuffles run at
//     ~1/4 of FMA throughput, so a per-column reduction would eat the win);
//   * the weight loads inside a warp are per-row distinct addresses, so 16 lanes
//     broadcast one 128-byte line -- DRAM traffic stays 1x the weight size;
//   * the fp32 accumulation order is exactly the reference's (ascending
//     super-block, ascending sub-block), which is what makes layer 1 of the
//     cross-check bit-exact instead of "close".
__global__ void quantize_q8_1_kernel(const float* __restrict__ x,
                                     int8_t* __restrict__ qs,
                                     Q8_1Ds* __restrict__ ds,
                                     int k, int ncols, int total_groups) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= total_groups) return;
    const int groups_per_col = k >> 5;
    const int c = warp / groups_per_col;
    const int g = warp - c * groups_per_col;

    const float xv = x[static_cast<size_t>(c) * k + static_cast<size_t>(g) * 32 + lane];
    float amax = fabsf(xv);
#pragma unroll
    for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    // Same as llama.cpp CUDA quantize_q8_1: quantize with the fp32 scale,
    // store/use the fp16 one.
    const float d32 = amax / 127.0f;
    int q = 0;
    if (d32 != 0.0f) q = static_cast<int>(roundf(xv / d32));   // roundf = half away from zero
    // ⚠️ __reduce_add_sync is a *collective*: every lane in the mask must execute
    // it. Calling it inside `if (lane == 0)` is undefined behaviour and silently
    // returned garbage sums -- which is exactly the Q4_K offset term, so the
    // whole prefill decoded to nonsense (B11 D2: teacher-agreement 0/50).
    // The layer-1 test missed it because it fed the kernel host-quantized
    // activations; it now checks the CUDA quantizer output byte-for-byte too.
    const int sum = __reduce_add_sync(0xffffffffu, q);
    // Block-major layout [k/32][ncol][32]: all columns' data for one 32-value
    // block is contiguous. With a column-major [ncol][k] layout the gemv below
    // would touch one 128-byte line per column per load -> 16 lines per warp
    // load and a 2.4x net SLOWDOWN vs the fp32 route (measured, see docs/34 §7).
    qs[static_cast<size_t>(g) * ncols * 32 + static_cast<size_t>(c) * 32 + lane] =
        static_cast<int8_t>(q);
    if (lane == 0) {
        ds[static_cast<size_t>(g) * ncols + c].sum_q = sum;
        ds[static_cast<size_t>(g) * ncols + c].d = __half2float(__float2half_rn(d32));
    }
}

// Tile = RW rows x NC columns per warp, looping over the LC = 16 allocated
// columns in groups of NC. The choice is purely a memory-shape tradeoff:
//
//   L1 bytes per output per super-block  ~  4/(NC*256) [weights re-read per
//   column group]  +  1/RW  [activation reuse across rows]
//
// and the MAC cost per lane is identical for every tile (each lane owns one
// (row, column) output and walks the whole k). So the tile does NOT change the
// arithmetic -- only which addresses a warp touches -- and bit-exactness of the
// accumulation order is preserved for all of them. B11 measured the sweep
// (see cpp/tests/gemv_bench.cpp int8 mode): see docs/34 §7.
template <int RW, int NC, int LC>
__global__ void gemv_q4k_q8_1_kernel(const uint8_t* __restrict__ w,
                                     const int8_t* __restrict__ qs,
                                     const Q8_1Ds* __restrict__ ds,
                                     float* __restrict__ y, int in, int out, int ncol) {
    const int lane = threadIdx.x & 31;
    const int r_local = lane / NC;
    const int c_in = lane - r_local * NC;
    const int row = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RW + r_local;
    if (row >= out) return;

    const int nb = in / QK_K;
    const uint8_t* wr = w + static_cast<size_t>(row) * nb * Q4K_BLOCK_BYTES;

#pragma unroll
    for (int c0 = 0; c0 < LC; c0 += NC) {
        const int c = c0 + c_in;
        // ⚠️ LC is the ALLOCATED column count (the activation stride) and is now
        // larger than the chunk width, so columns >= ncol are padding: skip them or
        // the store y[c*out+row] would run past the caller's buffer. This was a
        // latent out-of-bounds write while LC == 16 == the chunk width.
        if (c >= ncol) continue;
        // Activation layout is [k/32][LC][32]: one block's columns are contiguous.
        const int8_t* qc = qs + static_cast<size_t>(c) * 32;
        const Q8_1Ds* dc = ds + c;
        float acc = 0.0f;

        for (int b = 0; b < nb; ++b) {
            const uint8_t* sb = wr + static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
            const float d4 = __half2float(*reinterpret_cast<const __half*>(sb));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
            float sd = 0.0f, sm = 0.0f;
            // One 32-byte qs group serves sub-blocks 2g (low nibbles) and 2g+1
            // (high nibbles): 2 x uint4 weight loads for 64 values.
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint4 w0 = *reinterpret_cast<const uint4*>(sb + 16 + g * 32);
                const uint4 w1 = *reinterpret_cast<const uint4*>(sb + 16 + g * 32 + 16);
                const int8_t* a0 = qc + static_cast<size_t>(b * 8 + 2 * g) * LC * 32;
                const int8_t* a1 = a0 + static_cast<size_t>(LC) * 32;
                const uint4 x0 = *reinterpret_cast<const uint4*>(a0);
                const uint4 x1 = *reinterpret_cast<const uint4*>(a0 + 16);
                const uint4 x2 = *reinterpret_cast<const uint4*>(a1);
                const uint4 x3 = *reinterpret_cast<const uint4*>(a1 + 16);
                const uint32_t wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                const uint32_t av0[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                const uint32_t av1[8] = {x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
                int dot_lo = 0, dot_hi = 0;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    dot_lo = __dp4a(static_cast<int>(wv[q] & 0x0F0F0F0Fu), static_cast<int>(av0[q]), dot_lo);
                    dot_hi = __dp4a(static_cast<int>((wv[q] >> 4) & 0x0F0F0F0Fu), static_cast<int>(av1[q]), dot_hi);
                }
                uint8_t sc, m;
                detail::get_scale_min_k4(2 * g, sb + 4, &sc, &m);
                const float dlo = dc[static_cast<size_t>(b * 8 + 2 * g) * LC].d;
                sd = __fadd_rn(sd, __fmul_rn(dlo, static_cast<float>(dot_lo * static_cast<int>(sc))));
                sm = __fadd_rn(sm, __fmul_rn(dlo, static_cast<float>(
                    dc[static_cast<size_t>(b * 8 + 2 * g) * LC].sum_q * static_cast<int>(m))));
                detail::get_scale_min_k4(2 * g + 1, sb + 4, &sc, &m);
                const float dhi = dc[static_cast<size_t>(b * 8 + 2 * g + 1) * LC].d;
                sd = __fadd_rn(sd, __fmul_rn(dhi, static_cast<float>(dot_hi * static_cast<int>(sc))));
                sm = __fadd_rn(sm, __fmul_rn(dhi, static_cast<float>(
                    dc[static_cast<size_t>(b * 8 + 2 * g + 1) * LC].sum_q * static_cast<int>(m))));
            }
            // __fmul_rn/__fadd_rn everywhere: -fmad=true is the nvcc default and
            // FMA contraction would change the last bit, breaking the bit-exact
            // criterion of the layer-1 cross-check.
            const float part = __fsub_rn(__fmul_rn(d4, sd), __fmul_rn(dmin, sm));
            acc = __fadd_rn(acc, part);
        }
        y[static_cast<size_t>(c) * out + row] = acc;
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
    // ⚠️ The kernel maps the 32 lanes to R*NC (row, column) pairs, so NC tops out
    // at 16 here. The dispatch USED to end at NC=16 with no fallback, which was
    // invisible only because the prefill width was itself 16. Widening W (docs/37
    // §7.6) exposed it: columns 16.. came back silently UNWRITTEN. Serve wider
    // chunks by iterating 16-column groups with offset pointers; the arithmetic per
    // column is unchanged, so this stays bitwise-equal to the single-group launch.
    for (int done = 0; done < ncol; done += 16) {
        const int nc = (ncol - done >= 16) ? 16 : (ncol - done);
        const float* xc = x + static_cast<size_t>(done) * in;
        float* yc = y + static_cast<size_t>(done) * out;
        if (nc <= 2) gemv_q4k_mc_kernel<4, 2><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else if (nc <= 4) gemv_q4k_mc_kernel<4, 4><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else if (nc <= 8) gemv_q4k_mc_kernel<4, 8><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else gemv_q4k_mc_kernel<2, 16><<<grid_for(2), 128, 0, stream>>>(w, xc, yc, in, out);
    }
}

void gemv_q6k_mc(const uint8_t* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream) {
    if (ncol <= 1) { gemv_q6k(w, x, y, in, out, stream); return; }
    auto grid_for = [out](int r) { return ((out + r - 1) / r + 3) / 4; };
    // Same 16-column ceiling and the same fix as gemv_q4k_mc above.
    for (int done = 0; done < ncol; done += 16) {
        const int nc = (ncol - done >= 16) ? 16 : (ncol - done);
        const float* xc = x + static_cast<size_t>(done) * in;
        float* yc = y + static_cast<size_t>(done) * out;
        if (nc <= 2) gemv_q6k_mc_kernel<4, 2><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else if (nc <= 4) gemv_q6k_mc_kernel<4, 4><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else if (nc <= 8) gemv_q6k_mc_kernel<4, 8><<<grid_for(4), 128, 0, stream>>>(w, xc, yc, in, out);
        else gemv_q6k_mc_kernel<2, 16><<<grid_for(2), 128, 0, stream>>>(w, xc, yc, in, out);
    }
}

void embed_row_q6k(const uint8_t* embd, float* x, const int* token_p, int hidden,
                   cudaStream_t stream) {
    embed_row_q6k_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(embd, x, token_p, hidden);
}

void quantize_q8_1(const float* x, int8_t* qs, Q8_1Ds* ds, int k, int ncol,
                   cudaStream_t stream) {
    const int groups = k >> 5;
    const int total = ncol * groups;
    constexpr int WARPS_PER_BLOCK = 4;
    const int blocks = (total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    quantize_q8_1_kernel<<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(x, qs, ds, k, ncol, total);
}

void gemv_q4k_q8_1(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                   int in, int out, int ncol, int lc, cudaStream_t stream) {
    // ⚠️ The tile is chosen from the END-TO-END paired measurement, not from the
    // isolated bench: gemv-bench calls the same 7 MB matrix 200 times in a row,
    // so it runs L2-resident, while the engine streams ~900 MB per chunk with no
    // reuse. The two disagree (bench prefers 4x8 for the wide shapes; the engine
    // measures 8x4 faster), and only the engine number is real.
    gemv_q4k_q8_1_tile(w, qs, ds, y, in, out, ncol, B11_TILE_RW, B11_TILE_NC, lc, stream);
}

namespace {

// LC is the ALLOCATED activation stride. It MUST match the stride the qs/ds data
// was built with: the engine allocates B11_LC columns, while int8-ref-test's
// golden payloads use stride = ncol. Getting this wrong reads out of bounds --
// which is exactly how the first version of the W widening failed layer 1 on CUDA
// (it hard-coded LC = B11_LC while the test payload was 16 columns wide).
// `ncol` (valid columns) is passed down so the kernel can skip padding columns.
template <int LC>
void gemv_q4k_q8_1_tile_lc(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                           int in, int out, int ncol, int rw, int nc, cudaStream_t stream) {
    constexpr int WPB = 4;   // warps per block
    // rows_per_block = RW * WPB, so the grid is ceil(out / (RW * 4)).
    switch (rw * 100 + nc) {
        case 2 * 100 + 16:
            gemv_q4k_q8_1_kernel<2, 16, LC>
                <<<(out + 2 * WPB - 1) / (2 * WPB), 128, 0, stream>>>(w, qs, ds, y, in, out,
                                                                     ncol);
            break;
        case 4 * 100 + 8:
            gemv_q4k_q8_1_kernel<4, 8, LC>
                <<<(out + 4 * WPB - 1) / (4 * WPB), 128, 0, stream>>>(w, qs, ds, y, in, out,
                                                                     ncol);
            break;
        case 8 * 100 + 4:
            gemv_q4k_q8_1_kernel<8, 4, LC>
                <<<(out + 8 * WPB - 1) / (8 * WPB), 128, 0, stream>>>(w, qs, ds, y, in, out,
                                                                     ncol);
            break;
        case 16 * 100 + 2:
            gemv_q4k_q8_1_kernel<16, 2, LC>
                <<<(out + 16 * WPB - 1) / (16 * WPB), 128, 0, stream>>>(w, qs, ds, y, in, out,
                                                                       ncol);
            break;
        case 32 * 100 + 1:
            gemv_q4k_q8_1_kernel<32, 1, LC>
                <<<(out + 32 * WPB - 1) / (32 * WPB), 128, 0, stream>>>(w, qs, ds, y, in, out,
                                                                       ncol);
            break;
        default:
            gemv_q4k_q8_1_kernel<B11_TILE_RW, B11_TILE_NC, LC>
                <<<(out + B11_TILE_RW * WPB - 1) / (B11_TILE_RW * WPB), 128, 0, stream>>>(
                    w, qs, ds, y, in, out, ncol);
            break;
    }
}

}  // namespace

void gemv_q4k_q8_1_tile(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                        int in, int out, int ncol, int rw, int nc, int lc,
                        cudaStream_t stream) {
    if (lc > 16) {
        gemv_q4k_q8_1_tile_lc<B11_LC>(w, qs, ds, y, in, out, ncol, rw, nc, stream);
    } else {
        gemv_q4k_q8_1_tile_lc<16>(w, qs, ds, y, in, out, ncol, rw, nc, stream);
    }
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
