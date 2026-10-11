// B13: int8 tensor-core (IMMA) Q6_K prefill GEMM.
//
// Completes the coverage B11/B12 left open: B11 put int8 activations on Q4_K only
// and kept Q6_K on the fp32 route (docs/34 section 6, a scope-control decision);
// B12 measured that the int8 route then stops being the limiter at all -- Q6_K is
// 30% of the parameters and about HALF the gemv time (docs/36 section 7.8,
// docs/37 section 7.1). This file moves that half onto int8.
//
// Two design points, both settled by measurement before coding (docs/37 section 7):
//
//  1. `m16n8k16`, not m16n8k32. Q6_K's scale granularity is 16 VALUES
//     (sc[nn*8 + l/16 + 2*qq]), so K=16 is exactly one half-block. Measured
//     tensor-core ceilings are identical (1023.7 vs 1023.8 MAC/SM/clk), so the
//     smaller K costs nothing. K=32 simply could not express per-half scaling.
//
//  2. The A operand is built SIGNED (q6 in [-32,31]), which removes the offset
//     term entirely. Q6_K stores `v` in [0,63] and the reference computes
//     `q6 = v - 32`, so a naive build needs a correction `-32*sum(q8)` over the
//     half -- and the half-block sum of q8 is NOT stored in Q8_1Ds (only the
//     32-value sum is). Rather than recompute it (dp4a against ones, per column)
//     or widen the Q8_1 record, note that q6 FITS in int8, so the MMA can be fed
//     q6 directly:
//
//         int8(v - 32) == (v + 0x60) ^ 0x80     (byte-wise, no carry: v <= 63)
//
//     Two packed ops per 4 values, and the offset term disappears. This is why
//     docs/37 section 3.3's "how to obtain the half-block sum of q8" -- flagged
//     there as the project's one real design decision -- is moot.
//
// Exactness: A in [-32,31], B in [-127,127], K=16 -> |dot| <= 32*127*16 = 65024,
// so `.satfinite` is unreachable and the int32 result is exactly the reference's
// integer dot. The fp32 epilogue keeps the reference's parenthesisation and its
// (nn, qq, half) accumulation order, which is part of the spec.

#include <cstdint>

#include <cuda_runtime.h>

#include "gpu_quants.cuh"
#include "gpu_quants.h"
#include "int8_act.h"

namespace ri {
namespace gpuq {
namespace {

constexpr int QK_K = detail::QK_K_D;
constexpr int Q6K_BLOCK_BYTES = detail::Q6K_BLOCK_BYTES_D;

// m16n8k16 int8: A = 16x16 (2 regs), B = 16x8 (1 reg), C/D = 16x8 int32 (4 regs).
__device__ __forceinline__ void imma16_s8(int32_t c[4], uint32_t a0, uint32_t a1,
                                         uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// ⚠️ Q6_K's super-block is 210 bytes, and 210 % 4 == 2, so for ODD b the block
// start is only 2-byte aligned and a u32 load is illegal (it faults; this was a
// real misaligned-address crash). Q4_K never hits this because 144 % 4 == 0, and
// gpu_quants.cu's Q6_K gemv uses byte loads. Assemble the 4 bytes from two u16
// loads instead -- everything inside a block is 4-byte offset, so 2-byte
// alignment is guaranteed.
__device__ __forceinline__ uint32_t load_q6k_u32(const uint8_t* p) {
    const uint32_t lo = *reinterpret_cast<const uint16_t*>(p);
    const uint32_t hi = *reinterpret_cast<const uint16_t*>(p + 2);
    return lo | (hi << 16);
}

// MEASUREMENT-ONLY ablation knob (docs/37 §7.4). RADAR_Q6K_ABLATE=1 drops the
// packed sign conversion, =2 drops the whole A-fragment extraction *including its
// loads*, feeding a constant instead. The results are numerically WRONG by design;
// the point is to bound how much of the Q6_K time the extraction actually costs,
// so that an alignment repack is not built on speculation. (B12's lesson: isolated
// probes bound things from above, so measure the real kernel.)
//
// Read on the HOST and passed in: a function-scope static __device__ variable with
// dynamic initialisation is a compile error, and a per-call getenv would be silly.
int q6k_ablate_level() {
    static const int lvl = [] {
        const char* e = std::getenv("RADAR_Q6K_ABLATE");
        return e ? std::atoi(e) : 0;
    }();
    return lvl;
}

// One warp owns a 16-row x (NTILE*8)-column tile. Grid:
//   x = row tiles / WARPS,  y = column groups.
template <int WARPS, int NTILE>
__global__ void __launch_bounds__(WARPS * 32) gemv_q6k_mma_kernel(
    const uint8_t* __restrict__ w, const int8_t* __restrict__ qs,
    const Q8_1Ds* __restrict__ ds, float* __restrict__ y, int in, int out,
    int ncol, int stride, int ablate) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;   // 0..7
    const int tig = lane & 3;    // 0..3
    const int r0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 16;
    if (r0 >= out) return;

    const int nb = in / QK_K;
    const uint8_t* wrA = w + static_cast<size_t>(r0 + gid) * nb * Q6K_BLOCK_BYTES;
    const uint8_t* wrB = w + static_cast<size_t>(r0 + gid + 8) * nb * Q6K_BLOCK_BYTES;
    const int cb0 = blockIdx.y * (NTILE * 8);
    const int n_local = tig * 2;

    float acc[NTILE][4];
    float sd[NTILE][4];
#pragma unroll
    for (int t = 0; t < NTILE; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[t][i] = 0.0f;
            sd[t][i] = 0.0f;
        }

    for (int b = 0; b < nb; ++b) {
        const uint8_t* sbA = wrA + static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
        const uint8_t* sbB = wrB + static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
        const float d6A = __half2float(*reinterpret_cast<const __half*>(sbA + 208));
        const float d6B = __half2float(*reinterpret_cast<const __half*>(sbB + 208));

#pragma unroll
        for (int t = 0; t < NTILE; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) sd[t][i] = 0.0f;

        // Reference order: nn, then qq, then half (part of the spec).
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
                const int g = nn * 4 + qq;
                const int nib_shift = (qq < 2) ? 0 : 4;   // low nibble for qq<2
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    // ---- A fragment: 2 registers, K=16 = one half-block ----
                    // l = half*16 + k; ql index = nn*64 + (qq&1)*32 + l,
                    // qh index = 128 + nn*32 + l. 4 consecutive l = one u32 each.
                    const int ab = ablate;
                    uint32_t a0, a1;
                    if (ab >= 2) {
                        a0 = 0x01010101u;   // ablate: no loads, no extraction
                        a1 = 0x01010101u;
                    } else {
                        const int l0 = half * 16 + tig * 4;
                        const uint32_t qlA = load_q6k_u32(sbA + nn * 64 + (qq & 1) * 32 + l0);
                        const uint32_t qhA = load_q6k_u32(sbA + 128 + nn * 32 + l0);
                        const uint32_t qlB = load_q6k_u32(sbB + nn * 64 + (qq & 1) * 32 + l0);
                        const uint32_t qhB = load_q6k_u32(sbB + 128 + nn * 32 + l0);
                        const uint32_t mA = 0x0F0F0F0Fu, m2 = 0x03030303u;
                        // v = nib | (hi2 << 4), then the packed signed conversion.
                        const uint32_t vA = ((qlA >> nib_shift) & mA) |
                                            ((((qhA >> (2 * qq)) & m2)) << 4);
                        const uint32_t vB = ((qlB >> nib_shift) & mA) |
                                            ((((qhB >> (2 * qq)) & m2)) << 4);
                        if (ab == 1) {          // ablate: keep v (unsigned), skip the sign step
                            a0 = vA;
                            a1 = vB;
                        } else {
                            a0 = (vA + 0x60606060u) ^ 0x80808080u;
                            a1 = (vB + 0x60606060u) ^ 0x80808080u;
                        }
                    }

                    // sc is int8 at sb+192, index nn*8 + half + 2*qq (per ROW).
                    const int sc_idx = nn * 8 + half + 2 * qq;
                    const int32_t scgA = static_cast<int32_t>(
                        reinterpret_cast<const int8_t*>(sbA + 192)[sc_idx]);
                    const int32_t scgB = static_cast<int32_t>(
                        reinterpret_cast<const int8_t*>(sbB + 192)[sc_idx]);

#pragma unroll
                    for (int t = 0; t < NTILE; ++t) {
                        const int cb = cb0 + t * 8;
                        // ---- B fragment: 1 register, column = gid ----
                        const size_t blk = static_cast<size_t>(b * 8 + g);
                        const int8_t* qc = qs + blk * stride * 32 + (cb + gid) * 32;
                        const uint32_t b0 =
                            *reinterpret_cast<const uint32_t*>(qc + half * 16 + tig * 4);
                        int32_t c[4] = {0, 0, 0, 0};
                        imma16_s8(c, a0, a1, b0);

                        // ---- read out per half and scale (the spec's epilogue) ----
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int col = n_local + (i & 1);
                            if (cb + col >= ncol) continue;
                            const size_t di = blk * stride + cb + col;
                            const float d8 = ds[di].d;
                            const int32_t scg = (i >= 2) ? scgB : scgA;
                            sd[t][i] = __fadd_rn(sd[t][i], __fmul_rn(
                                d8, static_cast<float>(c[i] * scg)));
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int t = 0; t < NTILE; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float d6 = (i >= 2) ? d6B : d6A;
                acc[t][i] = __fadd_rn(acc[t][i], __fmul_rn(d6, sd[t][i]));
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

void gemv_q6k_mma(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds, float* y,
                  int in, int out, int ncol, int stride, cudaStream_t stream) {
    if (ncol <= 0 || in <= 0 || out <= 0) return;
    if (in % QK_K != 0 || out % 16 != 0) return;
    // RADAR_MMA_NTILE: N-tiles per A-fragment load. See gemv_q4k_mma; for Q6_K this
    // is the lever the ablation pointed at (docs/37 §7.4).
    static const int ntile = [] {
        const char* e = std::getenv("RADAR_MMA_NTILE");
        return e ? std::atoi(e) : 2;
    }();
    constexpr int WARPS = 4;
    const int ablate = q6k_ablate_level();
    const dim3 g2((out / 16 + WARPS - 1) / WARPS, (ncol + 15) / 16);
    const dim3 g4((out / 16 + WARPS - 1) / WARPS, (ncol + 31) / 32);
    const dim3 g8((out / 16 + WARPS - 1) / WARPS, (ncol + 63) / 64);
    if (ntile >= 8) {
        if (stride < ((ncol + 63) / 64) * 64) return;
        gemv_q6k_mma_kernel<WARPS, 8>
            <<<g8, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride, ablate);
    } else if (ntile >= 4) {
        if (stride < ((ncol + 31) / 32) * 32) return;
        gemv_q6k_mma_kernel<WARPS, 4>
            <<<g4, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride, ablate);
    } else {
        if (stride < ((ncol + 15) / 16) * 16) return;
        gemv_q6k_mma_kernel<WARPS, 2>
            <<<g2, WARPS * 32, 0, stream>>>(w, qs, ds, y, in, out, ncol, stride, ablate);
    }
}

}  // namespace gpuq
}  // namespace ri
