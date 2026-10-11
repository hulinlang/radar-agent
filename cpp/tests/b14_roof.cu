// B14 roofline probe -- "先量再写 kernel" (docs/36 §4 step 1, docs/37 §7.1).
//
// B14 rewrites prefill attention for K/V tile reuse. Before writing that kernel,
// this probe answers two questions that decide whether tiling is even the right
// fix, and gives the denominators for the utilization claim in docs/38 §0.2:
//
//   1. FP32 FFMA peak **in this same session**. Absolute numbers on this box
//      drift ~30% between sessions (docs/27), so a peak measured last week cannot
//      support a "attention runs at X% of peak" statement made today.
//   2. L2-resident read bandwidth at the attention working set (~12 MB/layer).
//      The hypothesis is that prefill attention streams K/V far faster than DRAM
//      could serve (so it is L2-resident) -- and the question is whether it is
//      against the L2 *ceiling* (fix = tiling) or merely latency-bound (fix = ILP).
//
// ⚠️ The first version of the L2 half was wrong and reported >10 TB/s. The strided
//    kernel gated on `d >= cols` while `d` was a *global* thread index, so with
//    cols=128 only block 0 did any work and the other 383 returned immediately;
//    the sequential loops were also 0.4-6 ms, i.e. mostly launch overhead.
//    Fixed below: per-block row slicing, power-of-two footprint with a rotating
//    window (so L1 cannot serve the hits), and loops long enough to dwarf launch.
//    Lesson (3rd time in B12-B14): a probe that reports something too good must be
//    audited for block/thread indexing and timing length before it is believed.
//
//   bash: cmake --build $BUILD --target b14-roof && $BUILD/b14-roof
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- 1. FFMA peak
template <int UNROLL>
__global__ void ffma_bench(float* out, int iters) {
    float a0 = threadIdx.x * 1e-9f + 1.0f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
    const float b = 0.9999999f, c = 1e-12f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
            a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
        }
    }
    const float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s == 1234.5678f) *out = s;   // never true; keeps the FMAs from being dead
}

// --------------------------------------------------- 2. L2-resident read peak
__global__ void l2read(const float4* __restrict__ src, long n4, long iters, float* sink) {
    float acc = 0.f;
    const long stride = (long)gridDim.x * blockDim.x;
    const long mask = n4 - 1;               // n4 is a power of two
    for (long it = 0; it < iters; ++it) {
        const long rot = (it * 8191) & mask;    // rotate the window every pass
        for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
            const float4 v = src[(i + rot) & mask];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 1234.5678f) *sink = acc;
}

// The shipped kernel's output phase: 128 threads, thread-per-d, walking keys.
// This is the access shape the current attention_prefill actually issues.
__global__ void l2strided(const float* __restrict__ src, int rows, int cols, int iters,
                          float* sink) {
    float acc = 0.f;
    const int d = threadIdx.x;              // 128 threads -> the shipped block shape
    if (d >= cols) return;
    for (int it = 0; it < iters; ++it)
        for (int r = blockIdx.x; r < rows; r += gridDim.x) acc += src[(long)r * cols + d];
    if (acc == 1234.5678f) *sink = acc;
}

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s  SMs=%d  L2=%d MiB  nominal clock=%.2f GHz\n", p.name,
                p.multiProcessorCount, p.l2CacheSize >> 20, p.clockRate / 1e6);

    float* sink = nullptr;
    cudaMalloc(&sink, sizeof(float));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    // ---- 1. FFMA ----
    {
        const int iters = 20000, unroll = 8, nthr = 256;
        const int blocks = p.multiProcessorCount * 8;    // 2048 threads/SM
        ffma_bench<unroll><<<blocks, nthr>>>(sink, 100);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        ffma_bench<unroll><<<blocks, nthr>>>(sink, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double fma = (double)blocks * nthr * unroll * 8.0 * iters;
        const double tflops = 2.0 * fma / (ms / 1e3) / 1e12;
        std::printf("\n[1] FP32 FFMA peak (same session)\n");
        std::printf("    %.2f ms -> %.1f GFMA/s = %.2f TFLOP/s   (%.1f FMA/SM/clk @ %.2f GHz)\n",
                    ms, fma / (ms / 1e3) / 1e9, tflops, fma / (ms / 1e3) /
                    p.multiProcessorCount / (p.clockRate * 1e3), p.clockRate / 1e6);
        std::printf("    docs/38 §0.2 uses this as the denominator for attention utilization.\n");
    }

    // ---- 2. L2 read bandwidth ----
    {
        const long bytes = 8L << 20;   // L2-resident, well past L1
        const long n4 = bytes / 16;    // 2^19
        float4* src = nullptr;
        cudaMalloc(&src, bytes);
        cudaMemset(src, 1, bytes);
        std::printf("\n[2a] L2-resident sequential float4 reads, 8 MiB footprint\n");
        double best = 0;
        for (int nthr = 256; nthr <= 1024; nthr *= 2) {
            for (int bpsm = 2; bpsm <= 16; bpsm *= 2) {
                const int blocks = p.multiProcessorCount * bpsm;
                const long iters = 4000;
                l2read<<<blocks, nthr>>>(src, n4, 1, sink);      // warm L2
                cudaDeviceSynchronize();
                cudaEventRecord(a);
                l2read<<<blocks, nthr>>>(src, n4, iters, sink);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                const double gbs = (double)bytes * iters / 1e9 / (ms / 1000.0);
                if (gbs > best) best = gbs;
                std::printf("    thr=%4d blk=%3d : %8.2f ms  ->  %7.1f GB/s\n", nthr, blocks, ms, gbs);
            }
        }
        std::printf("    best = %.1f GB/s\n", best);

        // The shape the shipped attention_prefill actually issues in its output
        // phase: 128 threads, one column per thread, sequential over keys.
        const int cols = 128, rows = 24576;    // 24576*128*4 = 12 MiB
        float* s2 = nullptr;
        cudaMalloc(&s2, (long)rows * cols * 4);
        cudaMemset(s2, 1, (long)rows * cols * 4);
        std::printf("\n[2b] the shipped phase-3 access shape (128 thr, 512 B stride)\n");
        for (int bpsm = 4; bpsm <= 32; bpsm *= 2) {
            const int blocks = p.multiProcessorCount * bpsm;
            const int it2 = 200;
            l2strided<<<blocks, 128>>>(s2, rows, cols, 1, sink);
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            l2strided<<<blocks, 128>>>(s2, rows, cols, it2, sink);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            const double gbs = (double)rows * cols * 4.0 * it2 / 1e9 / (ms / 1000.0);
            std::printf("    blk=%3d : %8.2f ms  ->  %7.1f GB/s (logical)\n", blocks, ms, gbs);
        }
        cudaFree(s2);
        cudaFree(src);
    }
    cudaFree(sink);
    return 0;
}
