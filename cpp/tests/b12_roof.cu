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
    return 0;
}
