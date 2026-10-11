// B12 roof probe -- "先量再写 kernel" (docs/36 §4 step 1).
//
// The question this answers is NOT "how fast is int8 MMA" -- mma_probe already
// measured 1023.9 MAC/SM/clk. The question is:
//
//   ★ Q4_K weights are 4-bit but the MMA needs int8. How much does the on-the-fly
//     nibble expansion cost, and therefore how wide must the prefill chunk be?
//
// Model (docs/36 §2.2b): one weight value costs `c` INT32 ops to expand, and that
// value is then reused for T MACs (once per token). So
//
//     unpack-op overhead  =  (1024 * c / T) / 64  =  16c / T
//
// At T=16 (today's chunk) even c=1 means 100% overhead on the INT32 pipe. The
// required T is directly proportional to c -- which is exactly why c must be
// measured rather than assumed.
//
// Methodology: ONE resident block per SM (no wave quantisation -- the flaw that
// made an earlier version of mma_probe report dp4a at 191 instead of 256), and
// the MIN of 3 round-robin passes. This is a THROUGHPUT probe: the nibble-to-byte
// mapping here is deliberately not the real fragment layout, because we are
// measuring cost, not correctness (correctness is doc 36 step 2/3's job).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("CUDA ERROR %s: %s\n", what, cudaGetErrorString(e));
}

__device__ __forceinline__ uint32_t expand4(uint32_t w) {
    // 1 word (4 nibbles) -> 4 int8 packed in one 32-bit register.
    return (w & 0x0000000Fu)
         | ((w & 0x000000F0u) << 4)
         | ((w & 0x00000F00u) << 8)
         | ((w & 0x0000F000u) << 12);
}

__device__ __forceinline__ void mma_m16n8k32(int32_t c[4], uint32_t a0, uint32_t a1,
                                            uint32_t a2, uint32_t a3, uint32_t b0,
                                            uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------- 1. pure MMA pipe
template <int CHAINS>
__global__ void probe_mma(int8_t* dummy, long long* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t a0 = 0x01010101u * (uint32_t)(lane + 1);
    uint32_t a1 = 0x01010101u * (uint32_t)(lane + 2);
    uint32_t a2 = 0x01010101u * (uint32_t)(lane + 3);
    uint32_t a3 = 0x01010101u * (uint32_t)(lane + 4);
    uint32_t b0 = 0x01010101u * (uint32_t)(lane + 5);
    uint32_t b1 = 0x01010101u * (uint32_t)(lane + 6);
    int32_t c[CHAINS][4];
    #pragma unroll
    for (int k = 0; k < CHAINS; k++)
        #pragma unroll
        for (int i = 0; i < 4; i++) c[k][i] = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++)
        #pragma unroll
        for (int k = 0; k < CHAINS; k++) mma_m16n8k32(c[k], a0, a1, a2, a3, b0, b1);
    long long t1 = clock64();
    int32_t s = 0;
    #pragma unroll
    for (int k = 0; k < CHAINS; k++)
        #pragma unroll
        for (int i = 0; i < 4; i++) s += c[k][i];
    if (s == 123456789) dummy[0] = 1;
    if (lane == 0) out[blockIdx.x] = t1 - t0;
}

// ------------------------------------------------------ 2. unpack only (INT32)
// 16 words -> 64 int8 per lane per iteration. 4 independent accumulators so the
// DCE barrier does not serialise the chain (it adds 1 op per 4 values, which is
// called out in the printout).
__global__ void probe_unpack(uint32_t* dummy, long long* out, int iters) {
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = 0x01234567u * (uint32_t)(i + 1) + threadIdx.x;
    uint32_t acc[4] = {0, 0, 0, 0};
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            w[i] ^= (uint32_t)(it + i);          // defeat loop-invariant hoisting
            acc[i & 3] ^= expand4(w[i]);
        }
    }
    long long t1 = clock64();
    uint32_t s = acc[0] ^ acc[1] ^ acc[2] ^ acc[3];
    if (s == 0xDEADBEEFu) dummy[0] = 1;
    if ((threadIdx.x & 31) == 0) out[blockIdx.x] = t1 - t0;
}

// ------------------------------------------- 3. MMA fed by on-the-fly unpack
// Per iteration a lane expands 4 words -> a0..a3 (16 int8), then issues R MMAs
// that REUSE that A tile. R is the number of N-tiles = ceil(T/8), so R is exactly
// the weight-reuse factor of a prefill chunk of T tokens.
template <int R>
__global__ void probe_mma_unpack(int8_t* dummy, long long* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) w[i] = 0x01234567u * (uint32_t)(i + 1) + lane;
    uint32_t b0 = 0x01010101u * (uint32_t)(lane + 5);
    uint32_t b1 = 0x01010101u * (uint32_t)(lane + 6);
    int32_t c[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) c[r][i] = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) w[i] ^= (uint32_t)(it + i);  // defeat hoisting
        const uint32_t a0 = expand4(w[0]);
        const uint32_t a1 = expand4(w[1]);
        const uint32_t a2 = expand4(w[2]);
        const uint32_t a3 = expand4(w[3]);
        #pragma unroll
        for (int r = 0; r < R; r++) mma_m16n8k32(c[r], a0, a1, a2, a3, b0, b1);
    }
    long long t1 = clock64();
    int32_t s = 0;
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) s += c[r][i];
    if (s == 123456789) dummy[0] = 1;
    if (lane == 0) out[blockIdx.x] = t1 - t0;
}

// ------------------------------------------- 3b. REALISTIC Q4_K extraction
// The probe above models a GENERAL nibble unpack (4 nibbles packed in one 16-bit
// word -> 4 bytes), which costs ~1.77 ops/value. That is NOT the Q4_K layout.
// In Q4_K a sub-block's 32 values are 32 CONSECUTIVE BYTES all at the same nibble
// position (see gpu_quants.cuh q4k_at), so 4 consecutive k values are the low (or
// high) nibble of 4 consecutive bytes: `p & 0x0F0F0F0F` or `(p>>4) & 0x0F0F0F0F`.
// 1-2 ops per 4 values, i.e. ~0.375 ops/value -- ~4.7x cheaper.
__device__ __forceinline__ uint32_t q4k_lo4(uint32_t p) { return p & 0x0F0F0F0Fu; }
__device__ __forceinline__ uint32_t q4k_hi4(uint32_t p) { return (p >> 4) & 0x0F0F0F0Fu; }

__global__ void probe_q4k_extract(uint32_t* dummy, long long* out, int iters) {
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = 0x01234567u * (uint32_t)(i + 1) + threadIdx.x;
    uint32_t acc[4] = {0, 0, 0, 0};
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            w[i] ^= (uint32_t)(it + i);
            acc[i & 3] ^= ((i & 1) ? q4k_hi4(w[i]) : q4k_lo4(w[i]));
        }
    }
    long long t1 = clock64();
    uint32_t s = acc[0] ^ acc[1] ^ acc[2] ^ acc[3];
    if (s == 0xDEADBEEFu) dummy[0] = 1;
    if ((threadIdx.x & 31) == 0) out[blockIdx.x] = t1 - t0;
}

template <int R>
__global__ void probe_mma_q4k(int8_t* dummy, long long* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) w[i] = 0x01234567u * (uint32_t)(i + 1) + lane;
    uint32_t b0 = 0x01010101u * (uint32_t)(lane + 5);
    uint32_t b1 = 0x01010101u * (uint32_t)(lane + 6);
    int32_t c[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) c[r][i] = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) w[i] ^= (uint32_t)(it + i);
        // two sub-blocks' worth: even sub-block = lo nibbles, odd = hi nibbles
        const uint32_t a0 = q4k_lo4(w[0]);
        const uint32_t a1 = q4k_hi4(w[1]);
        const uint32_t a2 = q4k_lo4(w[2]);
        const uint32_t a3 = q4k_hi4(w[3]);
        #pragma unroll
        for (int r = 0; r < R; r++) mma_m16n8k32(c[r], a0, a1, a2, a3, b0, b1);
    }
    long long t1 = clock64();
    int32_t s = 0;
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) s += c[r][i];
    if (s == 123456789) dummy[0] = 1;
    if (lane == 0) out[blockIdx.x] = t1 - t0;
}

// ===================== B13: Q6_K extraction (layout verified first) ==========
//
// ⚠️ Lesson from B12: the probe measures whatever you ASK it to measure, so the
// data layout must be read out of the source BEFORE writing the model. From
// gpu_quants.cuh q6k_at (the dequant the engine already uses):
//
//   nn = i>>7, r = i&127, qq = r>>5, l = r&31
//   byte = ql[nn*64 + l + (qq&1)*32]     nibble = (qq<2) ? low : high
//   high = qh[128 + nn*32 + l]           bits [2*qq, 2*qq+1]
//   q6   = (nibble | (high << 4)) - 32
//   scale= sc[nn*8 + l/16 + 2*qq]        <-- 16-VALUE granularity, not 32
//
// So 4 consecutive values of a group are ONE u32 from ql plus ONE u32 from qh --
// again not a general nibble unpack. But there is a real fork in how to use them,
// and which side wins decides the kernel shape:
//
//   3-term : dot = sum nib*q8 + 16*sum hi2*q8 - 32*sum q8   (2 dp4a, no combine)
//   combine: v = nib | (hi2<<4)                             (1 dp4a, +2 ALU)
//
// An MMA A-operand must be the final int8 value, so MMA needs `combine`; the dp4a
// route can use either. Both are measured below.
__device__ __forceinline__ uint32_t q6k_nib4(uint32_t p, bool hi) {
    return hi ? ((p >> 4) & 0x0F0F0F0Fu) : (p & 0x0F0F0F0Fu);
}
__device__ __forceinline__ uint32_t q6k_hi2(uint32_t p, int qq) {
    return (p >> (2 * qq)) & 0x03030303u;
}

// Extraction only, 3-term form: per word a nibble mask + a 2-bit extract.
__global__ void probe_q6k_extract_3term(uint32_t* dummy, long long* out, int iters) {
    uint32_t ql[8], qh[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ql[i] = 0x01234567u * (uint32_t)(i + 1) + threadIdx.x;
        qh[i] = 0x89ABCDEFu * (uint32_t)(i + 1) + threadIdx.x;
    }
    uint32_t a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ql[i] ^= (uint32_t)(it + i);
            qh[i] ^= (uint32_t)(it + i + 7);
            a0[i & 3] ^= q6k_nib4(ql[i], (i & 1) != 0);
            a1[i & 3] ^= q6k_hi2(qh[i], i & 3);
        }
    }
    long long t1 = clock64();
    uint32_t s = a0[0] ^ a0[1] ^ a0[2] ^ a0[3] ^ a1[0] ^ a1[1] ^ a1[2] ^ a1[3];
    if (s == 0xDEADBEEFu) dummy[0] = 1;
    if ((threadIdx.x & 31) == 0) out[blockIdx.x] = t1 - t0;
}

// Extraction only, combine form: the above plus the shift+or that forms v.
__global__ void probe_q6k_extract_combine(uint32_t* dummy, long long* out, int iters) {
    uint32_t ql[8], qh[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ql[i] = 0x01234567u * (uint32_t)(i + 1) + threadIdx.x;
        qh[i] = 0x89ABCDEFu * (uint32_t)(i + 1) + threadIdx.x;
    }
    uint32_t a[4] = {0, 0, 0, 0};
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ql[i] ^= (uint32_t)(it + i);
            qh[i] ^= (uint32_t)(it + i + 7);
            const uint32_t nib = q6k_nib4(ql[i], (i & 1) != 0);
            const uint32_t h2 = q6k_hi2(qh[i], i & 3);
            a[i & 3] ^= (nib | (h2 << 4));
        }
    }
    long long t1 = clock64();
    uint32_t s = a[0] ^ a[1] ^ a[2] ^ a[3];
    if (s == 0xDEADBEEFu) dummy[0] = 1;
    if ((threadIdx.x & 31) == 0) out[blockIdx.x] = t1 - t0;
}

// m16n8k16 (int8): A = 16x16 -> 2 regs/thread, B = 16x8 -> 1 reg, C/D = 16x8 int32.
// 2048 MACs per warp-instruction. K=16 is exactly Q6_K's scale granularity.
__device__ __forceinline__ void mma_m16n8k16(int32_t c[4], uint32_t a0, uint32_t a1,
                                            uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// m16n8k16 pipe ceiling, A already in registers (reference point).
template <int R>
__global__ void probe_mma16_reg(int8_t* dummy, long long* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t a0 = 0x01010101u * (uint32_t)(lane + 1);
    uint32_t a1 = 0x01010101u * (uint32_t)(lane + 2);
    uint32_t b0 = 0x01010101u * (uint32_t)(lane + 5);
    int32_t c[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) c[r][i] = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++)
        #pragma unroll
        for (int r = 0; r < R; r++) mma_m16n8k16(c[r], a0, a1, b0);
    long long t1 = clock64();
    int32_t s = 0;
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) s += c[r][i];
    if (s == 123456789) dummy[0] = 1;
    if (lane == 0) out[blockIdx.x] = t1 - t0;
}

// MMA fed by on-the-fly Q6_K extraction. Per iteration a lane builds ONE A tile
// (4 register loads: ql/qh x rows gid / gid+8) and reuses it for R MMAs, where R
// is the number of N-tiles = the weight-reuse factor, exactly as for Q4_K.
template <int R>
__global__ void probe_mma16_q6k(int8_t* dummy, long long* out, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t qlR[2], qhR[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        qlR[i] = 0x01234567u * (uint32_t)(i + 1) + lane;
        qhR[i] = 0x89ABCDEFu * (uint32_t)(i + 1) + lane;
    }
    uint32_t b0 = 0x01010101u * (uint32_t)(lane + 5);
    int32_t c[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) c[r][i] = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            qlR[i] ^= (uint32_t)(it + i);
            qhR[i] ^= (uint32_t)(it + i + 3);
        }
        // rows 0-7 (reg a0) and rows 8-15 (reg a1); both cover k = 0..15 here
        const bool hi = (it & 1) != 0;
        const int qq = it & 3;
        const uint32_t a0 = q6k_nib4(qlR[0], hi) | (q6k_hi2(qhR[0], qq) << 4);
        const uint32_t a1 = q6k_nib4(qlR[1], hi) | (q6k_hi2(qhR[1], qq) << 4);
        #pragma unroll
        for (int r = 0; r < R; r++) mma_m16n8k16(c[r], a0, a1, b0);
    }
    long long t1 = clock64();
    int32_t s = 0;
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int i = 0; i < 4; i++) s += c[r][i];
    if (s == 123456789) dummy[0] = 1;
    if (lane == 0) out[blockIdx.x] = t1 - t0;
}

static long long max_of(const long long* h, int n) {
    long long m = 0;
    for (int i = 0; i < n; i++) if (h[i] > m) m = h[i];
    return m;
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 60000;
    int dev = 0, sms = 0;
    ck(cudaGetDevice(&dev), "getDevice");
    ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "sms");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sms);

    const int threads = 256, warps = threads / 32, blocks = sms;  // one wave
    long long* dT = nullptr;
    ck(cudaMalloc(&dT, sizeof(long long) * blocks), "mallocT");
    long long* hT = (long long*)malloc(sizeof(long long) * blocks);
    int8_t* dc = nullptr; uint32_t* du = nullptr;
    ck(cudaMalloc(&dc, 4), "mallocC"); ck(cudaMalloc(&du, 4), "mallocU");

    // ---------- pass 1: pure MMA pipe ----------
    double c_mma = 1e30;
    for (int p = 0; p < 3; p++) {
        probe_mma<4><<<blocks, threads>>>(dc, dT, iters); ck(cudaDeviceSynchronize(), "mma");
        probe_mma<4><<<blocks, threads>>>(dc, dT, iters); ck(cudaDeviceSynchronize(), "mma2");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c1");
        c_mma = fmin(c_mma, (double)max_of(hT, blocks));
    }
    const double macs_mma = (double)warps * iters * 4 * 4096;
    const double R_mma = macs_mma / c_mma;                       // MAC/SM/clk

    // ---------- pass 2: unpack only ----------
    double c_unp = 1e30;
    for (int p = 0; p < 3; p++) {
        probe_unpack<<<blocks, threads>>>(du, dT, iters); ck(cudaDeviceSynchronize(), "unp");
        probe_unpack<<<blocks, threads>>>(du, dT, iters); ck(cudaDeviceSynchronize(), "unp2");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c2");
        c_unp = fmin(c_unp, (double)max_of(hT, blocks));
    }
    const double vals_unp = (double)warps * 32.0 * iters * 64.0;   // 64 int8/lane/iter
    const double V_unp = vals_unp / c_unp;                        // values/SM/clk
    const double c_coef = 64.0 / V_unp;                           // ops per value

    std::printf("\none resident block per SM (%d x %d threads, %d iters, min of 3)\n\n",
                blocks, threads, iters);
    std::printf("A) pure MMA (A in registers)        %8.1f MAC/SM/clk   <- ceiling\n", R_mma);
    std::printf("B) nibble expansion only            %8.1f values/SM/clk\n", V_unp);
    std::printf("   => implied unpack cost c         %8.2f INT32 ops per weight value\n", c_coef);
    std::printf("      (includes ~0.25 op/value from the DCE barrier)\n");
    std::printf("   => predicted required chunk  T >= 16c = %.0f tokens\n", 16.0 * c_coef);
    std::printf("      (for <=100%% INT32-pipe overhead; T >= 80c for <=20%%)\n\n");

    // ---------- pass 3: MMA fed by unpack, at several reuse factors ----------
    std::printf("C) MMA reusing an on-the-fly unpacked A tile\n");
    std::printf("   %-6s %-8s %14s %14s %10s\n", "R", "T=8R", "MAC/SM/clk", "vs ceiling", "overhead");
    #define RUN_R(RV) do {                                                       \
        double best = 1e30;                                                      \
        for (int p = 0; p < 3; p++) {                                            \
            probe_mma_unpack<RV><<<blocks, threads>>>(dc, dT, iters);            \
            ck(cudaDeviceSynchronize(), "mu");                                   \
            probe_mma_unpack<RV><<<blocks, threads>>>(dc, dT, iters);            \
            ck(cudaDeviceSynchronize(), "mu2");                                  \
            ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks,                    \
                          cudaMemcpyDeviceToHost), "c3");                        \
            best = fmin(best, (double)max_of(hT, blocks));                       \
        }                                                                        \
        const double macs = (double)warps * iters * (RV) * 4096;                 \
        const double got = macs / best;                                          \
        std::printf("   %-6d %-8d %14.1f %13.1f%% %9.0f%%\n", (RV), 8 * (RV),   \
                    got, 100.0 * got / R_mma, 100.0 * (R_mma / got - 1.0));      \
    } while (0)
    RUN_R(1); RUN_R(2); RUN_R(4); RUN_R(8); RUN_R(16);
    #undef RUN_R

    std::printf("\n   model check: overhead should track 16c/T = %0.1f/T\n", 16.0 * c_coef);
    std::printf("   reading: if R=2 (T=16, today's chunk) shows ~100%%+ overhead, the\n");
    std::printf("   chunk MUST grow before the MMA is worth writing (docs/36 2.2b).\n");

    // ---------- pass 4: the REALISTIC Q4_K extraction (see 3b above) ----------
    double c_q4 = 1e30;
    for (int p = 0; p < 3; p++) {
        probe_q4k_extract<<<blocks, threads>>>(du, dT, iters); ck(cudaDeviceSynchronize(), "q4");
        probe_q4k_extract<<<blocks, threads>>>(du, dT, iters); ck(cudaDeviceSynchronize(), "q4b");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c4");
        c_q4 = fmin(c_q4, (double)max_of(hT, blocks));
    }
    const double V_q4 = vals_unp / c_q4;
    const double c_q4k = 64.0 / V_q4;

    std::printf("\nD) realistic Q4_K extraction (same nibble of 4 consecutive bytes)\n");
    std::printf("   extraction only                    %8.1f values/SM/clk\n", V_q4);
    std::printf("   => cost c                          %8.2f ops/value  (was %.2f for the\n",
                c_q4k, c_coef);
    std::printf("                                          GENERAL nibble unpack)\n");
    std::printf("   => required T >= 16c               %8.1f tokens (<=100%% overhead)\n", 16.0 * c_q4k);
    std::printf("   => required T for <=20%% overhead   %8.1f tokens\n", 80.0 * c_q4k);

    std::printf("\nE) MMA reusing a realistically-extracted Q4_K A tile\n");
    std::printf("   %-6s %-8s %14s %14s %10s\n", "R", "T=8R", "MAC/SM/clk", "vs ceiling", "overhead");
    #define RUN_Q(RV) do {                                                       \
        double best = 1e30;                                                      \
        for (int p = 0; p < 3; p++) {                                            \
            probe_mma_q4k<RV><<<blocks, threads>>>(dc, dT, iters);               \
            ck(cudaDeviceSynchronize(), "mq");                                   \
            probe_mma_q4k<RV><<<blocks, threads>>>(dc, dT, iters);               \
            ck(cudaDeviceSynchronize(), "mq2");                                  \
            ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks,                    \
                          cudaMemcpyDeviceToHost), "c5");                        \
            best = fmin(best, (double)max_of(hT, blocks));                       \
        }                                                                        \
        const double macs = (double)warps * iters * (RV) * 4096;                 \
        const double got = macs / best;                                          \
        std::printf("   %-6d %-8d %14.1f %13.1f%% %9.0f%%\n", (RV), 8 * (RV),   \
                    got, 100.0 * got / R_mma, 100.0 * (R_mma / got - 1.0));      \
    } while (0)
    RUN_Q(1); RUN_Q(2); RUN_Q(4); RUN_Q(8);
    #undef RUN_Q

    std::printf("\n   => with the real layout, W=16 (R=2) should no longer be the\n");
    std::printf("      bottleneck; the remaining limit is weight re-read traffic.\n");

    // ================= B13: Q6_K extraction + m16n8k16 =================
    // Layout verified against gpu_quants.cuh q6k_at BEFORE modelling this time.

    // ---------- F: Q6_K extraction, both forms ----------
    double c3 = 1e30, cc = 1e30, c16reg = 1e30;
    for (int p = 0; p < 3; p++) {
        probe_q6k_extract_3term<<<blocks, threads>>>(du, dT, iters);
        ck(cudaDeviceSynchronize(), "q63");
        probe_q6k_extract_3term<<<blocks, threads>>>(du, dT, iters);
        ck(cudaDeviceSynchronize(), "q63b");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c6");
        c3 = fmin(c3, (double)max_of(hT, blocks));

        probe_q6k_extract_combine<<<blocks, threads>>>(du, dT, iters);
        ck(cudaDeviceSynchronize(), "q6c");
        probe_q6k_extract_combine<<<blocks, threads>>>(du, dT, iters);
        ck(cudaDeviceSynchronize(), "q6cb");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c7");
        cc = fmin(cc, (double)max_of(hT, blocks));

        probe_mma16_reg<4><<<blocks, threads>>>(dc, dT, iters);
        ck(cudaDeviceSynchronize(), "k16");
        probe_mma16_reg<4><<<blocks, threads>>>(dc, dT, iters);
        ck(cudaDeviceSynchronize(), "k16b");
        ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks, cudaMemcpyDeviceToHost), "c8");
        c16reg = fmin(c16reg, (double)max_of(hT, blocks));
    }
    // each probe's loop handles 8 words/iteration = 32 values per lane
    const double vals32 = (double)warps * 32.0 * iters * 32.0;
    const double V3 = vals32 / c3, VC = vals32 / cc;
    const double R16 = ((double)warps * iters * 4 * 2048) / c16reg;

    std::printf("\nF) Q6_K extraction (layout verified against q6k_at FIRST)\n");
    std::printf("   m16n8k16 ceiling (A in registers)   %8.1f MAC/SM/clk\n", R16);
    std::printf("   3-term form   (nib + hi2 apart)     %8.1f values/SM/clk => c = %.2f\n",
                V3, 64.0 / V3);
    std::printf("   combine form  (v = nib|hi2<<4)      %8.1f values/SM/clk => c = %.2f\n",
                VC, 64.0 / VC);
    std::printf("   vs Q4_K's c = %.2f (both forms are >= that, as expected)\n", c_q4k);
    std::printf("   => required W >= 16c: 3-term %.1f, combine %.1f\n",
                16.0 * (64.0 / V3), 16.0 * (64.0 / VC));

    // ---------- G: m16n8k16 fed by Q6_K extraction ----------
    std::printf("\nG) m16n8k16 reusing an on-the-fly Q6_K A tile\n");
    std::printf("   %-6s %-8s %14s %14s %10s\n", "R", "W=8R", "MAC/SM/clk", "vs k16 ceil", "overhead");
    #define RUN_Q6(RV) do {                                                      \
        double best = 1e30;                                                      \
        for (int p = 0; p < 3; p++) {                                            \
            probe_mma16_q6k<RV><<<blocks, threads>>>(dc, dT, iters);             \
            ck(cudaDeviceSynchronize(), "m6");                                   \
            probe_mma16_q6k<RV><<<blocks, threads>>>(dc, dT, iters);             \
            ck(cudaDeviceSynchronize(), "m6b");                                  \
            ck(cudaMemcpy(hT, dT, sizeof(long long) * blocks,                    \
                          cudaMemcpyDeviceToHost), "c9");                        \
            best = fmin(best, (double)max_of(hT, blocks));                       \
        }                                                                        \
        const double macs = (double)warps * iters * (RV) * 2048;                 \
        const double got = macs / best;                                          \
        std::printf("   %-6d %-8d %14.1f %13.1f%% %9.0f%%\n", (RV), 8 * (RV),   \
                    got, 100.0 * got / R16, 100.0 * (R16 / got - 1.0));          \
    } while (0)
    RUN_Q6(1); RUN_Q6(2); RUN_Q6(4); RUN_Q6(8);
    #undef RUN_Q6

    std::printf("\n   B13 reading: compare G's W=16 row against E's 906.8 (Q4_K).\n");
    std::printf("   Q6_K also needs ~1.5-2x the dp4a count (3-term) or 2 loads/value\n");
    std::printf("   (combine), so a lower number here is expected -- it must still be\n");
    std::printf("   far above the dp4a pipe (256) for B13 to pay off.\n");
    return 0;
}
