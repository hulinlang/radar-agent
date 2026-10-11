// B14: FA-style causal prefill attention (see gpu_attn.h for the why).
//
// This is the engine port of the prototype validated in cpp/tests/b14_attn.cu.
// Two things differ from the prototype and both are the reason the prototype's
// numbers are an upper bound rather than a result (docs/38 §7.9):
//
//   1. K/V are PAGINATED (kv_page = 16 keys per page) and reached through
//      block_table, so the tile loader walks the page table instead of doing a
//      flat contiguous copy.
//   2. The chunk may be narrower than BR (tail chunk, or a smaller
//      RADAR_PREFILL_W), so rows >= nr are padding: they are computed but their
//      output is discarded. The tile is chosen from nr to keep that waste small.
//
// Numerics: online softmax with fp32 accumulation, `scale` folded into Q. The
// per-row arithmetic is NOT identical to attention_prefill's (different reduction
// order, as that kernel already differs from the decode path), so the DoD is
// docs/35's Tier B, not Tier A.
#include "gpu_attn.h"

#include <cstdio>
#include <cstdlib>

namespace ri {
namespace gpua {
namespace {

constexpr int DH = FA_HEAD_DIM;

// One block: BR = 8*TQ query rows of ONE q head, sweeping keys in BC = CQG*TC
// tiles. threads = 8*CQG, DPT = DH/CQG output dims per thread.
//
//   thread (rq, cq) owns  S: TQ rows x TC keys   and   O: TQ rows x DPT dims
//
// PAD is not decoration: with a smem row stride of DH=128 (a multiple of 32
// words) every lane of the K-tile float4 load hits the same bank group, i.e. a
// CQG-way conflict. Stride DH+4 = 132 spreads them and is worth 1.2-1.7x
// (docs/38 §7.3).
template <int TQ, int TC, int CQG, int PAD>
__global__ void attn_fa_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                               float* __restrict__ att,
                               size_t layer_k_stride, size_t layer_v_stride,
                               size_t block_stride, size_t head_stride,
                               int n_head, int n_kv_head, int kv_page,
                               const int* __restrict__ block_table,
                               int pos_base, int nr, float scale) {
    constexpr int BR = 8 * TQ;
    constexpr int BC = CQG * TC;
    constexpr int DPT = DH / CQG;   // multiple of 4: the PV step loads float4
    constexpr int SPD = DH + PAD;   // smem row stride
    constexpr int NV4 = DH / 4;     // float4 per row
    extern __shared__ float sm[];
    float* qs = sm;                 // BR * SPD
    float* ks = qs + BR * SPD;      // BC * SPD
    float* vs = ks + BC * SPD;      // BC * SPD
    float* ps = vs + BC * SPD;      // BR * BC

    const int tid = threadIdx.x;
    const int rq = tid / CQG;
    const int cq = tid % CQG;
    const int qh = blockIdx.x % n_head;
    const int row0 = (blockIdx.x / n_head) * BR;
    const int kvh = qh / (n_head / n_kv_head);
    const int row_end = min(row0 + BR, nr);      // exclusive
    const int kmax = pos_base + row_end;         // keys 0..kmax-1 are in play

    // ---- Q tile (scale folded in here: free, and what the prototype validated) ----
    for (int i = tid; i < BR * NV4; i += blockDim.x) {
        const int rl = i / NV4;
        const int v4 = i % NV4;
        const int rr = row0 + rl;
        float4 qq = make_float4(0.f, 0.f, 0.f, 0.f);
        if (rr < nr) {
            qq = *reinterpret_cast<const float4*>(q + ((size_t)rr * n_head + qh) * DH +
                                                  (size_t)v4 * 4);
            qq.x *= scale; qq.y *= scale; qq.z *= scale; qq.w *= scale;
        }
        *reinterpret_cast<float4*>(&qs[rl * SPD + v4 * 4]) = qq;
    }
    __syncthreads();

    const float* kb = kv + layer_k_stride;
    const float* vb = kv + layer_v_stride;

    float m[TQ], l[TQ], o[TQ][DPT];
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        m[i] = -1e30f;
        l[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d) o[i][d] = 0.0f;
    }

    const int rbase = rq * TQ, cbase = cq * TC, dbase = cq * DPT;

    for (int t0 = 0; t0 < kmax; t0 += BC) {
        __syncthreads();   // previous iteration's ps/ks/vs reads must be done
        // K/V tile through the page table. BC=32 keys span 2 pages; within a page
        // one kv head's DH floats per key are contiguous (16*DH floats), and the
        // 32 lanes of a warp cover exactly one key row (512 B) -> fully coalesced.
        for (int i = tid; i < BC * NV4; i += blockDim.x) {
            const int j = i / NV4;
            const int v4 = i % NV4;
            const int t = t0 + j;
            float4 kk = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv = kk;
            if (t < kmax) {
                const size_t off = (size_t)block_table[t / kv_page] * block_stride +
                                   (size_t)kvh * head_stride +
                                   (size_t)(t % kv_page) * DH + (size_t)v4 * 4;
                kk = *reinterpret_cast<const float4*>(kb + off);
                vv = *reinterpret_cast<const float4*>(vb + off);
            }
            *reinterpret_cast<float4*>(&ks[j * SPD + v4 * 4]) = kk;
            *reinterpret_cast<float4*>(&vs[j * SPD + v4 * 4]) = vv;
        }
        __syncthreads();

        // ---- S = Q K^T, register blocked over d ----
        float s[TQ][TC];
#pragma unroll
        for (int i = 0; i < TQ; ++i)
#pragma unroll
            for (int j = 0; j < TC; ++j) s[i][j] = 0.0f;
#pragma unroll 4
        for (int d = 0; d < DH; d += 4) {
            float4 qv[TQ], kv4[TC];
#pragma unroll
            for (int i = 0; i < TQ; ++i)
                qv[i] = *reinterpret_cast<const float4*>(&qs[(rbase + i) * SPD + d]);
#pragma unroll
            for (int j = 0; j < TC; ++j)
                kv4[j] = *reinterpret_cast<const float4*>(&ks[(cbase + j) * SPD + d]);
#pragma unroll
            for (int i = 0; i < TQ; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j)
                    s[i][j] += qv[i].x * kv4[j].x + qv[i].y * kv4[j].y +
                               qv[i].z * kv4[j].z + qv[i].w * kv4[j].w;
        }

        // ---- causal mask + running row max (CQG-lane segments = the rq group) ----
        float mt[TQ];
#pragma unroll
        for (int i = 0; i < TQ; ++i) {
            const int qpos = pos_base + row0 + rbase + i;
            mt[i] = -1e30f;
#pragma unroll
            for (int j = 0; j < TC; ++j) {
                if (t0 + cbase + j > qpos) s[i][j] = -1e30f;
                mt[i] = fmaxf(mt[i], s[i][j]);
            }
#pragma unroll
            for (int off = CQG / 2; off; off >>= 1)
                mt[i] = fmaxf(mt[i], __shfl_xor_sync(0xffffffffu, mt[i], off, CQG));
        }
        // ---- rescale O and l onto the new max, then exponentiate ----
        float lt[TQ];
#pragma unroll
        for (int i = 0; i < TQ; ++i) {
            const float mn = fmaxf(m[i], mt[i]);
            const float alpha = expf(m[i] - mn);
            l[i] *= alpha;
#pragma unroll
            for (int d = 0; d < DPT; ++d) o[i][d] *= alpha;
            m[i] = mn;
            lt[i] = 0.0f;
#pragma unroll
            for (int j = 0; j < TC; ++j) {
                s[i][j] = expf(s[i][j] - mn);
                lt[i] += s[i][j];
            }
#pragma unroll
            for (int off = CQG / 2; off; off >>= 1)
                lt[i] += __shfl_xor_sync(0xffffffffu, lt[i], off, CQG);
            l[i] += lt[i];
        }

        for (int i = 0; i < TQ; ++i)
#pragma unroll
            for (int j = 0; j < TC; ++j) ps[(rbase + i) * BC + cbase + j] = s[i][j];
        __syncthreads();

        // ---- O += P V ----
        for (int j = 0; j < BC; ++j) {
            float pv[TQ];
#pragma unroll
            for (int i = 0; i < TQ; ++i) pv[i] = ps[(rbase + i) * BC + j];
#pragma unroll
            for (int dv = 0; dv < DPT; dv += 4) {
                const float4 vv = *reinterpret_cast<const float4*>(&vs[j * SPD + dbase + dv]);
#pragma unroll
                for (int i = 0; i < TQ; ++i) {
                    o[i][dv + 0] += pv[i] * vv.x;
                    o[i][dv + 1] += pv[i] * vv.y;
                    o[i][dv + 2] += pv[i] * vv.z;
                    o[i][dv + 3] += pv[i] * vv.w;
                }
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        const int rr = row0 + rbase + i;
        if (rr >= nr) continue;   // padding row: computed, deliberately not written
        const float inv = (l[i] > 0.0f) ? 1.0f / l[i] : 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d)
            att[((size_t)rr * n_head + qh) * DH + dbase + d] = o[i][d] * inv;
    }
}

constexpr int kTC = 1, kCQG = 32, kPAD = 4;

template <int TQ>
FaStatus launch_tq(const float* q, const float* kv, float* att, size_t layer_k_stride,
                   size_t layer_v_stride, size_t block_stride, size_t head_stride,
                   int n_head, int n_kv_head, int kv_page, const int* block_table,
                   int pos_base, int nr, float scale, cudaStream_t stream) {
    constexpr int BR = 8 * TQ;
    constexpr int BC = kCQG * kTC;
    constexpr int SPD = DH + kPAD;
    constexpr size_t SH = (size_t)(BR * SPD + 2 * BC * SPD + BR * BC) * 4;
    // >48 KB needs the opt-in; if the device refuses, say so instead of launching
    // a kernel that would fail (or worse, silently do nothing).
    static int ready = -1;
    if (ready < 0) {
        ready = (cudaFuncSetAttribute(attn_fa_kernel<TQ, kTC, kCQG, kPAD>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)SH) == cudaSuccess) ? 1 : 0;
    }
    if (!ready) return FA_NO_SMEM;
    const int rt = (nr + BR - 1) / BR;
    attn_fa_kernel<TQ, kTC, kCQG, kPAD><<<rt * n_head, 8 * kCQG, SH, stream>>>(
        q, kv, att, layer_k_stride, layer_v_stride, block_stride, head_stride, n_head,
        n_kv_head, kv_page, block_table, pos_base, nr, scale);
    return FA_LAUNCHED;
}

}  // namespace

FaStatus attention_prefill_fa(const float* q, const float* kv, float* att,
                              size_t layer_k_stride, size_t layer_v_stride,
                              size_t block_stride, size_t head_stride, int n_head,
                              int n_kv_head, int head_dim, int kv_page,
                              const int* block_table, int pos_base, int nr, float scale,
                              int cfg, cudaStream_t stream) {
    if (head_dim != DH || nr <= 0 || n_head <= 0 || n_kv_head <= 0) return FA_UNSUPPORTED;
    if (cfg == 2) {
        return launch_tq<8>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                            head_stride, n_head, n_kv_head, kv_page, block_table, pos_base,
                            nr, scale, stream);
    }
    if (cfg == 3) {
        return launch_tq<4>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                            head_stride, n_head, n_kv_head, kv_page, block_table, pos_base,
                            nr, scale, stream);
    }
    // cfg 1: smallest BR that still covers nr. Padding rows are computed by the QK
    // and PV steps, so a BR far wider than nr wastes real time on a full KV sweep
    // -- which matters most for the tail chunk (its KV range is the longest).
    if (nr > 32) {
        return launch_tq<8>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                            head_stride, n_head, n_kv_head, kv_page, block_table, pos_base,
                            nr, scale, stream);
    }
    if (nr > 16) {
        return launch_tq<4>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                            head_stride, n_head, n_kv_head, kv_page, block_table, pos_base,
                            nr, scale, stream);
    }
    if (nr > 8) {
        return launch_tq<2>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                            head_stride, n_head, n_kv_head, kv_page, block_table, pos_base,
                            nr, scale, stream);
    }
    return launch_tq<1>(q, kv, att, layer_k_stride, layer_v_stride, block_stride,
                        head_stride, n_head, n_kv_head, kv_page, block_table, pos_base, nr,
                        scale, stream);
}

const char* attention_prefill_fa_tile_name(int cfg, int nr) {
    if (cfg == 2) return "TQ=8 (BR=64, forced)";
    if (cfg == 3) return "TQ=4 (BR=32, forced)";
    if (nr > 32) return "TQ=8 (BR=64)";
    if (nr > 16) return "TQ=4 (BR=32)";
    if (nr > 8) return "TQ=2 (BR=16)";
    return "TQ=1 (BR=8)";
}

}  // namespace gpua
}  // namespace ri
