// B14 attention A/B -- the pre-implementation instrument (docs/38 §5 step 2).
//
// Why this exists: docs/38 §0 shows the shipped prefill attention runs at ~2.3% of
// the same-session FFMA peak and ~36% of the L2 read ceiling. Two explanations fit
// that, and they call for different kernels:
//
//   (a) redundant traffic -- the block is one query row x one head, so every row
//       re-reads all of K and V. Fix = tile reuse (FA).
//   (b) issue/latency -- the inner loop is 4 FMA + 5 shuffles per key (a
//       cross-lane reduction *per key*), so the shuffle port and the dependent
//       chain dominate regardless of traffic. Fix = register blocking.
//
// They are not exclusive; FA fixes both. But pricing them apart in one binary, in
// one session, is what stops me writing a kernel against the wrong story -- the
// mistake B12 made twice (docs/36 §7.5, §7.8).
//
// Sections:
//   [2a] FA vs a host double reference, incl. a case that FORCES an online-softmax
//        rescale (the error mode that looks plausible everywhere else)
//   [1]  the shipped kernel, copied verbatim, with per-phase clock64 accounting
//   [3]  A/B at production shape across T, plus a (TQ, TC, CQG, PAD) sweep
//
// Template knobs on the FA kernel:
//   TQ  rows per thread        BR = 8*TQ          (query rows per block)
//   TC  keys per thread        BC = CQG*TC        (keys per KV tile)
//   CQG column-groups          threads = 8*CQG, DPT = DH/CQG output dims per thread
//   PAD smem row padding       stride = DH + PAD  (bank-conflict knob, see below)
//
// Two findings from the first (buggy) run are already baked in:
//   * softmax is 0.3-1.6% of the shipped kernel's block-cycles -- the 14
//     __syncthreads there are not the problem, so online softmax costs nothing.
//   * with stride = DH = 128 the K-tile float4 load has all CQG lanes hitting the
//     same bank group (row stride is a multiple of 32 words), i.e. a CQG-way
//     conflict. Hence the PAD knob: stride 132 (= 4 mod 32, still 16 B aligned)
//     spreads the lanes over distinct bank groups.
//
// Simplification vs the engine, stated so the numbers are not over-read: K and V
// live in a contiguous per-head buffer here, not the paginated KV of the engine.
// Total traffic and coalescing are the same; only the page-index arithmetic is
// dropped, and both sides of the A/B use the same layout.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int NH = 16, NKV = 8, DH = 128;   // qwen3vl-2b: head_count / kv / head_dim

// ============================ [1] shipped kernel ============================
// Copied from cpp/src/gpu_model.cu attention_prefill, with clock64() around each
// phase and an added __syncthreads() at the phase-2/3 boundary so the clocks are
// read after all threads finish. The added syncs cannot change the result (each
// phase's inputs are complete at those points).
__global__ void attn_shipped(const float* q, const float* kbase, const float* vbase,
                             int n_head, int n_kv_head, int head_dim, int Tmax,
                             int pos_base, float scale, float* att,
                             unsigned long long* phase) {
    extern __shared__ float sm[];
    float* scores = sm;
    float* red = sm + Tmax;                 // engine: sm + (pos_base + nr)
    const int r = blockIdx.x / n_head;
    const int qh = blockIdx.x % n_head;
    const int kvh = qh / (n_head / n_kv_head);
    const float* q_h = q + ((size_t)r * n_head + qh) * head_dim;
    const int T = pos_base + r + 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;

    unsigned long long c0 = clock64();
    float qpart[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qpart[i] = q_h[32 * i + lane];
    const float* khead = kbase + (size_t)kvh * Tmax * head_dim;
    for (int t = warp; t < T; t += nwarp) {
        const float* kk = khead + (size_t)t * head_dim;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc += qpart[i] * kk[32 * i + lane];
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncthreads();
    unsigned long long c1 = clock64();

    // phase 2: softmax (block tree reductions, exactly as the engine)
    float mx = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) mx = fmaxf(mx, scores[t]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        scores[t] = expf(scores[t] - mx);
        sum += scores[t];
    }
    __syncthreads();
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    __syncthreads();
    unsigned long long c2 = clock64();

    // phase 3: output -- warp per 32-dim slice, lanes over d, sequential over t
    const int d = warp * 32 + lane;
    if (d < head_dim) {
        const float* vhead = vbase + (size_t)kvh * Tmax * head_dim;
        float acc = 0.0f;
        for (int t = 0; t < T; ++t) acc += scores[t] * inv * vhead[(size_t)t * head_dim + d];
        att[((size_t)r * n_head + qh) * head_dim + d] = acc;
    }
    unsigned long long c3 = clock64();

    if (threadIdx.x == 0) {
        atomicAdd(&phase[0], c1 - c0);
        atomicAdd(&phase[1], c2 - c1);
        atomicAdd(&phase[2], c3 - c2);
        atomicAdd(&phase[3], 1);
    }
}

// ============================== [2] FA prototype ==============================
template <int TQ, int TC, int CQG, int PAD>
__global__ void attn_fa(const float* q, const float* kbase, const float* vbase,
                        int n_head, int n_kv_head, int head_dim, int Tmax, int nr,
                        int pos_base, float scale, float* att) {
    constexpr int BR = 8 * TQ;
    constexpr int BC = CQG * TC;
    constexpr int DPT = DH / CQG;      // output dims per thread (multiple of 4)
    constexpr int SPD = DH + PAD;      // smem row stride
    extern __shared__ float sm[];
    float* qs = sm;                    // BR * SPD
    float* ks = qs + BR * SPD;         // BC * SPD
    float* vs = ks + BC * SPD;         // BC * SPD
    float* ps = vs + BC * SPD;         // BR * BC

    const int tid = threadIdx.x;
    const int rq = tid / CQG;
    const int cq = tid % CQG;
    const int qh = blockIdx.x % n_head;
    const int row0 = (blockIdx.x / n_head) * BR;
    const int kvh = qh / (n_head / n_kv_head);
    const int row_end = min(row0 + BR, nr);        // exclusive
    const int kmax = pos_base + row_end;           // keys 0..kmax-1 are in play

    // `scale` is folded into Q here. The reference applies it to the score;
    // folding it into Q is free (one multiply at load) and is what FA does.
    for (int i = tid; i < BR * SPD; i += blockDim.x) {
        const int rr = row0 + i / SPD;
        const int dd = i % SPD;
        qs[i] = (rr < nr && dd < DH) ? q[((size_t)rr * n_head + qh) * DH + dd] * scale : 0.0f;
    }
    __syncthreads();

    const float* khead = kbase + (size_t)kvh * Tmax * DH;
    const float* vhead = vbase + (size_t)kvh * Tmax * DH;

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
        __syncthreads();                       // previous iteration's ps/ks/vs reads done
        for (int i = tid; i < BC * SPD; i += blockDim.x) {
            const int tt = t0 + i / SPD;
            const int dd = i % SPD;
            const bool ok = (tt < kmax && dd < DH);
            ks[i] = ok ? khead[(size_t)tt * DH + dd] : 0.0f;
            vs[i] = ok ? vhead[(size_t)tt * DH + dd] : 0.0f;
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
            float4 qv[TQ], kv[TC];
#pragma unroll
            for (int i = 0; i < TQ; ++i)
                qv[i] = *reinterpret_cast<const float4*>(&qs[(rbase + i) * SPD + d]);
#pragma unroll
            for (int j = 0; j < TC; ++j)
                kv[j] = *reinterpret_cast<const float4*>(&ks[(cbase + j) * SPD + d]);
#pragma unroll
            for (int i = 0; i < TQ; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j)
                    s[i][j] += qv[i].x * kv[j].x + qv[i].y * kv[j].y +
                               qv[i].z * kv[j].z + qv[i].w * kv[j].w;
        }

        // ---- causal mask + running row max (width-CQG segments = the rq group) ----
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
        // ---- rescale O and l to the new max, then exponentiate ----
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
        if (rr >= nr) continue;
        const float inv = (l[i] > 0.0f) ? 1.0f / l[i] : 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d)
            att[((size_t)rr * n_head + qh) * DH + dbase + d] = o[i][d] * inv;
    }
}

#define FA_SMEM(TQv, TCv, CQGv, PADv) \
    ((size_t)(8 * (TQv) * (DH + (PADv)) + 2 * (CQGv) * (TCv) * (DH + (PADv)) + 8 * (TQv) * (CQGv) * (TCv)) * 4)
#define FA_GRID(nr, TQv) ((((nr) + 8 * (TQv) - 1) / (8 * (TQv))) * NH)
#define FA_THREADS(CQGv) (8 * (CQGv))

// ============================ host double reference ============================
static void ref_attn(const float* q, const float* k, const float* v, int nr, int n_head,
                     int n_kv_head, int Tmax, int pos_base, float scale, double* out) {
    const int gqa = n_head / n_kv_head;
    double* sc = new double[Tmax];
    for (int r = 0; r < nr; ++r) {
        const int T = pos_base + r + 1;
        for (int qh = 0; qh < n_head; ++qh) {
            const int kvh = qh / gqa;
            double mx = -1e30;
            for (int t = 0; t < T; ++t) {
                double a = 0.0;
                for (int d = 0; d < DH; ++d)
                    a += (double)q[((size_t)r * n_head + qh) * DH + d] *
                         (double)k[((size_t)kvh * Tmax + t) * DH + d];
                sc[t] = a * scale;
                if (sc[t] > mx) mx = sc[t];
            }
            double sum = 0.0;
            for (int t = 0; t < T; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            for (int d = 0; d < DH; ++d) {
                double a = 0.0;
                for (int t = 0; t < T; ++t)
                    a += sc[t] * (double)v[((size_t)kvh * Tmax + t) * DH + d];
                out[((size_t)r * n_head + qh) * DH + d] = a / sum;
            }
        }
    }
    delete[] sc;
}

static float frand(unsigned& st) {
    st = st * 1664525u + 1013904223u;
    return ((st >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s  SMs=%d  L2=%d MiB\n\n", p.name, p.multiProcessorCount,
                p.l2CacheSize >> 20);

    const int Tmax = 8192;
    const size_t nq = (size_t)Tmax * NH * DH, nkv = (size_t)NKV * Tmax * DH;
    float* hq = new float[nq];
    float* hk = new float[nkv];
    float* hv = new float[nkv];
    unsigned st = 12345u;
    for (size_t i = 0; i < nq; ++i) hq[i] = frand(st) * 0.5f;
    for (size_t i = 0; i < nkv; ++i) { hk[i] = frand(st); hv[i] = frand(st); }

    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *datt = nullptr;
    unsigned long long* dphase = nullptr;
    cudaMalloc(&dq, nq * 4);
    cudaMalloc(&dk, nkv * 4);
    cudaMalloc(&dv, nkv * 4);
    cudaMalloc(&datt, nq * 4);
    cudaMalloc(&dphase, 4 * sizeof(unsigned long long));
    cudaMemcpy(dq, hq, nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, nkv * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, nkv * 4, cudaMemcpyHostToDevice);
    const float scale = 1.0f / std::sqrt((float)DH);

    bool corr_ok = true;
    // NOTE: this must be float, not double. The first version declared it `double*`
    // and cudaMemcpy'd raw floats into it, then compared as doubles -- reading a
    // float's bits as a double yields a denormal, so the error came out as
    // |ref| regardless of what the kernel did. The tell was that the reported
    // error was *identically* 9.99e-01 for every config and every case: an error
    // that does not move with the configuration is a harness bug, not a kernel bug.
    float* hatt = new float[nq];
    double* ref = new double[nq];

    std::printf("[2a] FA vs host double reference\n");
#define RUN_FA(TQv, TCv, CQGv, PADv)                                                        \
    do {                                                                                    \
        const int BR = 8 * (TQv);                                                           \
        const size_t sh = FA_SMEM(TQv, TCv, CQGv, PADv);                                    \
        cudaMemset(datt, 0, nq * 4);                                                        \
        cudaFuncSetAttribute(attn_fa<TQv, TCv, CQGv, PADv>,                                 \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);         \
        attn_fa<TQv, TCv, CQGv, PADv><<<FA_GRID(nr, TQv), FA_THREADS(CQGv), sh>>>(          \
            dq, dk, dv, NH, NKV, DH, Tmax, nr, pos_base, scale, datt);                      \
        const cudaError_t e = cudaDeviceSynchronize();                                      \
        if (e != cudaSuccess) {                                                             \
            std::printf("      %-22s TQ=%d TC=%d CQG=%d PAD=%d CUDA: %s\n", tag, (TQv),     \
                        (TCv), (CQGv), (PADv), cudaGetErrorString(e));                      \
            corr_ok = false;                                                                \
        } else {                                                                            \
            cudaMemcpy(hatt, datt, nq * 4, cudaMemcpyDeviceToHost);                         \
            double worst = 0.0;                                                             \
            for (int r = 0; r < nr; ++r)                                                    \
                for (int qh = 0; qh < NH; ++qh)                                             \
                    for (int d = 0; d < DH; ++d) {                                          \
                        const size_t i = ((size_t)r * NH + qh) * DH + d;                    \
                        const double err = std::fabs(hatt[i] - ref[i]);                     \
                        if (err > worst) worst = err;                                       \
                    }                                                                       \
            std::printf("      %-22s TQ=%d TC=%d CQG=%d PAD=%d BR=%2d BC=%2d : %.2e  %s\n", \
                        tag, (TQv), (TCv), (CQGv), (PADv), BR, (CQGv) * (TCv), worst,       \
                        worst < 1e-5 ? "OK" : "FAIL");                                      \
            if (!(worst < 1e-5)) corr_ok = false;                                           \
        }                                                                                   \
    } while (0)

    // Case A: one KV tile.
    {
        const int nr = 8, pos_base = 0;
        ref_attn(hq, hk, hv, nr, NH, NKV, Tmax, pos_base, scale, ref);
        const char* tag = "A: 1 tile";
        RUN_FA(1, 1, 16, 0);
        RUN_FA(2, 2, 16, 0);
        RUN_FA(2, 1, 32, 0);
        RUN_FA(1, 1, 16, 4);
        RUN_FA(2, 2, 16, 4);
        RUN_FA(2, 1, 32, 4);
    }
    // Case B: TWO KV tiles with a forced rescale. k rows 32..63 are scaled x20, so
    // the second tile's running max blows past the first tile's and
    // exp(m_old - m_new) is far from 1 -- the only case here that exercises the
    // rescale of O and l, i.e. the FA bug that looks plausible everywhere else.
    {
        for (int h = 0; h < NKV; ++h)
            for (int t = 32; t < 64; ++t)
                for (int d = 0; d < DH; ++d) hk[((size_t)h * Tmax + t) * DH + d] *= 20.0f;
        cudaMemcpy(dk, hk, nkv * 4, cudaMemcpyHostToDevice);
        const int nr = 64, pos_base = 0;
        ref_attn(hq, hk, hv, nr, NH, NKV, Tmax, pos_base, scale, ref);
        const char* tag = "B: 2 tiles + rescale";
        RUN_FA(1, 1, 16, 0);
        RUN_FA(2, 1, 16, 0);
        RUN_FA(2, 2, 16, 0);
        RUN_FA(4, 1, 16, 0);
        RUN_FA(1, 1, 16, 4);
        RUN_FA(2, 1, 16, 4);
        RUN_FA(2, 2, 16, 4);
        RUN_FA(4, 1, 16, 4);
        for (int h = 0; h < NKV; ++h)
            for (int t = 32; t < 64; ++t)
                for (int d = 0; d < DH; ++d) hk[((size_t)h * Tmax + t) * DH + d] /= 20.0f;
        cudaMemcpy(dk, hk, nkv * 4, cudaMemcpyHostToDevice);
    }
#undef RUN_FA

    // ---------------- [1] phase split of the shipped kernel ----------------
    std::printf("\n[1] shipped kernel: per-phase share (nr=64, block-cycles summed)\n");
    std::printf("      %-8s %9s %9s %9s   %s\n", "T", "scores", "softmax", "output", "wall ms");
    for (int T : {1024, 2048, 4096, 7525}) {
        const int nr = 64, pos_base = T - nr;
        cudaMemset(dphase, 0, 4 * sizeof(unsigned long long));
        const size_t sh = (size_t)(T + 128) * 4;
        attn_shipped<<<nr * NH, 128, sh>>>(dq, dk, dv, NH, NKV, DH, T, pos_base, scale, datt, dphase);
        cudaDeviceSynchronize();
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        attn_shipped<<<nr * NH, 128, sh>>>(dq, dk, dv, NH, NKV, DH, T, pos_base, scale, datt, dphase);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        unsigned long long ph[4];
        cudaMemcpy(ph, dphase, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        const double tot = (double)(ph[0] + ph[1] + ph[2]);
        std::printf("      %-8d %8.1f%% %8.1f%% %8.1f%%   %8.3f\n", T,
                    100.0 * ph[0] / tot, 100.0 * ph[1] / tot, 100.0 * ph[2] / tot, ms);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }

    // ---------------- [3] A/B across T ----------------
    auto time_shipped = [&](int T) {
        const int nr = 64, pos_base = T - nr;
        const size_t sh = (size_t)(T + 128) * 4;
        attn_shipped<<<nr * NH, 128, sh>>>(dq, dk, dv, NH, NKV, DH, T, pos_base, scale, datt, dphase);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int rep = 0; rep < 3; ++rep) {
            cudaEvent_t a, b;
            cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            attn_shipped<<<nr * NH, 128, sh>>>(dq, dk, dv, NH, NKV, DH, T, pos_base, scale, datt, dphase);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            if (ms < best) best = ms;
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        return best;
    };

    std::printf("\n[3] A/B: shipped vs a few FA configs across T (nr=64 chunk)\n");
    std::printf("      %-8s %-18s %10s %11s\n", "T", "kernel", "ms", "vs shipped");
    for (int T : {1024, 2048, 4096, 7525}) {
        const int nr = 64, pos_base = T - nr;
        const float shipped_ms = time_shipped(T);
        std::printf("      %-8d %-18s %10.3f %11s\n", T, "shipped", shipped_ms, "1.00x");
#define AB(TQv, TCv, CQGv, PADv)                                                            \
        do {                                                                                \
            const size_t sh = FA_SMEM(TQv, TCv, CQGv, PADv);                                \
            cudaFuncSetAttribute(attn_fa<TQv, TCv, CQGv, PADv>,                             \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);     \
            attn_fa<TQv, TCv, CQGv, PADv><<<FA_GRID(nr, TQv), FA_THREADS(CQGv), sh>>>(      \
                dq, dk, dv, NH, NKV, DH, T, nr, pos_base, scale, datt);                     \
            cudaDeviceSynchronize();                                                        \
            const cudaError_t e = cudaGetLastError();                                       \
            if (e != cudaSuccess) {                                                         \
                std::printf("      %-8d FA<%d,%d,%d,%d> %10s\n", T, (TQv), (TCv), (CQGv),   \
                            (PADv), cudaGetErrorString(e));                                 \
            } else {                                                                        \
                float best = 1e30f;                                                         \
                for (int rep = 0; rep < 3; ++rep) {                                         \
                    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);             \
                    cudaEventRecord(a);                                                     \
                    attn_fa<TQv, TCv, CQGv, PADv><<<FA_GRID(nr, TQv), FA_THREADS(CQGv),     \
                                                    sh>>>(dq, dk, dv, NH, NKV, DH, T, nr,   \
                                                          pos_base, scale, datt);           \
                    cudaEventRecord(b); cudaEventSynchronize(b);                            \
                    float ms = 0; cudaEventElapsedTime(&ms, a, b);                          \
                    if (ms < best) best = ms;                                               \
                    cudaEventDestroy(a); cudaEventDestroy(b);                               \
                }                                                                           \
                std::printf("      %-8d FA<%d,%d,%d,%d>   %10.3f %10.2fx\n", T, (TQv),       \
                            (TCv), (CQGv), (PADv), best, shipped_ms / best);                \
            }                                                                               \
        } while (0)
        AB(2, 1, 16, 4);
        AB(8, 2, 16, 4);
        AB(2, 1, 32, 4);
        AB(4, 2, 32, 4);
#undef AB
    }

    // ---------------- [3b] full sweep at T=7525 ----------------
    std::printf("\n[3b] sweep at T=7525 (nr=64). BR=8*TQ  BC=CQG*TC  threads=8*CQG\n");
    {
        const int T = 7525, nr = 64, pos_base = T - nr;
        const float shipped_ms = time_shipped(T);
        std::printf("      shipped = %.3f ms\n", shipped_ms);
        std::printf("      %-28s %6s %8s %8s %9s %8s\n", "config", "smemKB", "BR", "BC",
                    "thr", "ms");
#define SWEEP(TQv, TCv, CQGv, PADv)                                                         \
        do {                                                                                \
            const size_t sh = FA_SMEM(TQv, TCv, CQGv, PADv);                                \
            const cudaError_t se = cudaFuncSetAttribute(                                   \
                attn_fa<TQv, TCv, CQGv, PADv>,                                              \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);                      \
            attn_fa<TQv, TCv, CQGv, PADv><<<FA_GRID(nr, TQv), FA_THREADS(CQGv), sh>>>(      \
                dq, dk, dv, NH, NKV, DH, T, nr, pos_base, scale, datt);                     \
            cudaDeviceSynchronize();                                                        \
            const cudaError_t e = cudaGetLastError();                                       \
            if (se != cudaSuccess || e != cudaSuccess) {                                    \
                std::printf("      %-28s %6zu %8d %8d %9d   unavailable(%s)\n",             \
                            "TQ=" #TQv " TC=" #TCv " CQG=" #CQGv " PAD=" #PADv, sh / 1024,  \
                            8 * (TQv), (CQGv) * (TCv), 8 * (CQGv), cudaGetErrorString(e));  \
            } else {                                                                        \
                float best = 1e30f;                                                         \
                for (int rep = 0; rep < 3; ++rep) {                                         \
                    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);             \
                    cudaEventRecord(a);                                                     \
                    attn_fa<TQv, TCv, CQGv, PADv><<<FA_GRID(nr, TQv), FA_THREADS(CQGv),     \
                                                    sh>>>(dq, dk, dv, NH, NKV, DH, T, nr,   \
                                                          pos_base, scale, datt);           \
                    cudaEventRecord(b); cudaEventSynchronize(b);                            \
                    float ms = 0; cudaEventElapsedTime(&ms, a, b);                          \
                    if (ms < best) best = ms;                                               \
                    cudaEventDestroy(a); cudaEventDestroy(b);                               \
                }                                                                           \
                std::printf("      %-28s %6zu %8d %8d %9d %7.3f ms %6.2fx\n",               \
                            "TQ=" #TQv " TC=" #TCv " CQG=" #CQGv " PAD=" #PADv, sh / 1024,  \
                            8 * (TQv), (CQGv) * (TCv), 8 * (CQGv), best, shipped_ms / best); \
            }                                                                               \
        } while (0)
        SWEEP(1, 1, 16, 0);
        SWEEP(1, 1, 16, 4);
        SWEEP(2, 1, 16, 0);
        SWEEP(2, 1, 16, 4);
        SWEEP(4, 1, 16, 4);
        SWEEP(2, 2, 16, 4);
        SWEEP(4, 2, 16, 4);
        SWEEP(8, 2, 16, 4);
        SWEEP(2, 4, 16, 4);
        SWEEP(4, 4, 16, 4);
        SWEEP(1, 1, 32, 0);
        SWEEP(1, 1, 32, 4);
        SWEEP(2, 1, 32, 4);
        SWEEP(4, 1, 32, 4);
        SWEEP(2, 2, 32, 4);
        SWEEP(4, 2, 32, 4);
        SWEEP(8, 1, 32, 4);
        SWEEP(8, 2, 32, 4);
#undef SWEEP
    }

    std::printf("\ncorrectness: %s\n", corr_ok ? "PASS" : "FAIL");
    delete[] hq; delete[] hk; delete[] hv; delete[] hatt; delete[] ref;
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(datt); cudaFree(dphase);
    return corr_ok ? 0 : 1;
}
