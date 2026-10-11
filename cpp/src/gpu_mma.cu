// B12: int8 tensor-core (IMMA) Q4_K prefill GEMM.
//
// Replaces the dp4a inner loop of cpp/src/gpu_quants.cu's int8 gemv with
// mma.sync.aligned.m16n8k32 (sm_89 accepts it; see cpp/tests/b12_roof.cu).
// Everything else -- the Q8_1 activation layout, the Q4_K super-block structure,
// the fp32 epilogue and its parenthesisation -- is IDENTICAL to B11, so the
// result is bit-exact with ri::matvec_q4_k_q8_1_host and with B11's dp4a kernel.
//
// Why the epilogue cannot be folded into the accumulator: the reference scales
// EACH sub-block's dot by d8_j * sc_j in fp32 and sums those terms
//     sd = sum_j d8_j * f32(dot1_j * sc_j)
// so every j must be read out of the tensor core separately. One MMA's K=32 is
// exactly one Q4_K sub-block, which is what makes that natural.
//
// N (the number of token columns) is a RUNTIME parameter, not a compile-time
// chunk width: docs/36 section 7.5 showed the only real lever on weight re-read
// traffic is how many tokens share one weight load, and the same kernel should
// serve W=16 (today's chunk) through W=full-prompt. NTILE only controls how many
// columns one warp chews per A-fragment; gridDim.y covers the rest.
//
// ⚠️ Q4_K's nibble layout is why the "unpack" here is nearly free, and it is
// also where an earlier cost model of ours went wrong: inside a sub-block the 32
// values live in 32 CONSECUTIVE BYTES, all at the same nibble position
// (gpu_quants.cuh q4k_at). So 4 consecutive k values cost one u32 load plus
// `p & 0x0F0F0F0F` (even sub-block) or `(p >> 4) & 0x0F0F0F0F` (odd) -- 1-2 ALU
// ops per 4 values, i.e. ~0.375 ops/value. cpp/tests/b12_roof.cu originally
// measured a GENERAL nibble expansion (4 nibbles packed in 2 bytes -> 4 bytes,
// 1.77 ops/value), which is NOT this layout and overstated the cost ~4.7x.

#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>

#include "gpu_quants.cuh"
#include "gpu_quants.h"
#include "int8_act.h"

namespace ri {
namespace gpuq {
namespace {

constexpr int QK_K = detail::QK_K_D;
constexpr int Q4K_BLOCK_BYTES = detail::Q4K_BLOCK_BYTES_D;

// m16n8k32, int8 in / int32 out. `.satfinite` is safe and unreachable here: the
// largest possible |dot1| is 15 * 127 * 32 = 60960, far below INT32_MAX, so
// saturation can never engage and this is exactly the reference's int32 sum.
__device__ __forceinline__ void imma_s8(int32_t c[4], uint32_t a0, uint32_t a1,
                                       uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One warp owns a 16-row x (NTILE*8)-column tile. Grid:
//   x = row tiles / WARPS,  y = column groups.
template <int WARPS, int NTILE>
__global__ void __launch_bounds__(WARPS * 32) gemv_q4k_mma_kernel(
    const uint8_t* __restrict__ w, const int8_t* __restrict__ qs,
    const Q8_1Ds* __restrict__ ds, float* __restrict__ y, int in, int out,
    int ncol, int stride) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;   // 0..7
    const int tig = lane & 3;    // 0..3
    const int r0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 16;
    if (r0 >= out) return;

    const int nb = in / QK_K;
    const uint8_t* wrA = w + static_cast<size_t>(r0 + gid) * nb * Q4K_BLOCK_BYTES;
    const uint8_t* wrB = w + static_cast<size_t>(r0 + gid + 8) * nb * Q4K_BLOCK_BYTES;
    const int cb0 = blockIdx.y * (NTILE * 8);

    // C/D fragment: row = gid (+8 for regs 2,3), column = tig*2 (+1 for odd regs).
    const int n_local = tig * 2;

    float acc[NTILE][4];
    float sd[NTILE][4], sm[NTILE][4];
#pragma unroll
    for (int t = 0; t < NTILE; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[t][i] = 0.0f;
            sd[t][i] = 0.0f;
            sm[t][i] = 0.0f;
        }

    for (int b = 0; b < nb; ++b) {
        const uint8_t* sbA = wrA + static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
        const uint8_t* sbB = wrB + static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
        const float d4A = __half2float(*reinterpret_cast<const __half*>(sbA));
        const float dminA = __half2float(*reinterpret_cast<const __half*>(sbA + 2));
        const float d4B = __half2float(*reinterpret_cast<const __half*>(sbB));
        const float dminB = __half2float(*reinterpret_cast<const __half*>(sbB + 2));

#pragma unroll
        for (int t = 0; t < NTILE; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sd[t][i] = 0.0f;
                sm[t][i] = 0.0f;
            }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // ---- A fragments: 4 registers, 4 consecutive k each ----
            // regs 0/1 cover k = tig*4..+3 (byte offset tig*4), regs 2/3 cover
            // k = 16+tig*4..+3; regs 0/2 are row gid, regs 1/3 are row gid+8.
            const uint8_t* qA = sbA + 16 + (j >> 1) * 32;
            const uint8_t* qB = sbB + 16 + (j >> 1) * 32;
            const bool hi = (j & 1) != 0;
            const uint32_t p0 = *reinterpret_cast<const uint32_t*>(qA + tig * 4);
            const uint32_t p1 = *reinterpret_cast<const uint32_t*>(qB + tig * 4);
            const uint32_t p2 = *reinterpret_cast<const uint32_t*>(qA + 16 + tig * 4);
            const uint32_t p3 = *reinterpret_cast<const uint32_t*>(qB + 16 + tig * 4);
            const uint32_t a0 = hi ? ((p0 >> 4) & 0x0F0F0F0Fu) : (p0 & 0x0F0F0F0Fu);
            const uint32_t a1 = hi ? ((p1 >> 4) & 0x0F0F0F0Fu) : (p1 & 0x0F0F0F0Fu);
            const uint32_t a2 = hi ? ((p2 >> 4) & 0x0F0F0F0Fu) : (p2 & 0x0F0F0F0Fu);
            const uint32_t a3 = hi ? ((p3 >> 4) & 0x0F0F0F0Fu) : (p3 & 0x0F0F0F0Fu);

            uint8_t scA, mA, scB, mB;
            detail::get_scale_min_k4(j, sbA + 4, &scA, &mA);
            detail::get_scale_min_k4(j, sbB + 4, &scB, &mB);

#pragma unroll
            for (int t = 0; t < NTILE; ++t) {
                const int cb = cb0 + t * 8;
                // ---- B fragment: column = gid, k = tig*4..+3 / 16+tig*4..+3 ----
                const size_t blk = static_cast<size_t>(b * 8 + j);
                const int8_t* qc = qs + blk * stride * 32 + (cb + gid) * 32;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(qc + tig * 4);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(qc + 16 + tig * 4);
                int32_t c[4] = {0, 0, 0, 0};
                imma_s8(c, a0, a1, a2, a3, b0, b1);

                // ---- read out and apply the fp32 epilogue, per sub-block ----
                // d8/dot2 depend on the COLUMN only; sc/m on the ROW only.
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = n_local + (i & 1);
                    if (cb + col >= ncol) continue;
                    const size_t di = blk * stride + cb + col;
                    const float d8 = ds[di].d;
                    const int32_t dot2 = ds[di].sum_q;
                    const bool rowB = (i >= 2);
                    const uint8_t sc = rowB ? scB : scA;
                    const uint8_t m = rowB ? mB : mA;
                    sd[t][i] = __fadd_rn(sd[t][i], __fmul_rn(
                        d8, static_cast<float>(c[i] * static_cast<int32_t>(sc))));
                    sm[t][i] = __fadd_rn(sm[t][i], __fmul_rn(
                        d8, static_cast<float>(dot2 * static_cast<int32_t>(m))));
                }
            }
        }

#pragma unroll
        for (int t = 0; t < NTILE; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float d4 = (i >= 2) ? d4B : d4A;
                const float dmin = (i >= 2) ? dminB : dminA;
                acc[t][i] = __fadd_rn(acc[t][i],
                                      __fsub_rn(__fmul_rn(d4, sd[t][i]),
                                                __fmul_rn(dmin, sm[t][i])));
            }
    }

#pragma unroll
    for (int t = 0; t < NTILE; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = cb0 + t * 8 + n_local + (i & 1);
            if (col >= ncol) continue;
            const int row = r0 + gid + ((i >= 2) ? 8 : 0);
            y[static_cast<size_t>(col) * out + row] = acc[t][i];
        }
}

}  // namespace

void gemv_q4k_mma(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                  int in, int out, int ncol, int stride, cudaStream_t stream) {
    // RADAR_MMA_NTILE: how many 8-column tiles one warp chews per A-fragment load.
    // This is the whole point of widening W (docs/37 §7.4): the Q6_K kernel is
    // latency-bound at W=16 because ~5.3 warps/SM is all the parallelism out/16
    // allows, so the fix is to make each A build feed more MMAs.
    static const int ntile = [] {
        const char* e = std::getenv("RADAR_MMA_NTILE");
        return e ? std::atoi(e) : 2;
    }();
    constexpr int WARPS = 4;
    const dim3 g2((out / 16 + WARPS - 1) / WARPS, (ncol + 15) / 16);
    const dim3 g4((out / 16 + WARPS - 1) / WARPS, (ncol + 31) / 32);
    const dim3 g8((out / 16 + WARPS - 1) / WARPS, (ncol + 63) / 64);
    // The B-fragment load touches columns [cb0, cb0+7] of the LAST group, so the
    // activation buffer must be allocated to a whole number of column groups.
    if (ntile >= 8) {
        if (stride < ((ncol + 63) / 64) * 64) return;
        gemv_q4k_mma_kernel<WARPS, 8>
            <<<g8, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride);
    } else if (ntile >= 4) {
        if (stride < ((ncol + 31) / 32) * 32) return;
        gemv_q4k_mma_kernel<WARPS, 4>
            <<<g4, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride);
    } else {
        if (stride < ((ncol + 15) / 16) * 16) return;
        gemv_q4k_mma_kernel<WARPS, 2>
            <<<g2, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride);
    }
}

}  // namespace gpuq
}  // namespace ri
