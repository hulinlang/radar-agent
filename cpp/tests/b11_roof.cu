// b11_roof.cu: what does a *pure read* of the Q4_K weights cost, under the
// access patterns B11 has to choose between? No math, just reads -- so any gap
// to the memcpy ceiling is the pattern's fault, not the arithmetic's.
//
//   b11-roof <q4km.gguf> [iters]
//
// Patterns, all reading exactly the same total bytes (out * in/256 * 144):
//   1. sequential         : linear vectorized sweep          (the memcpy shape)
//   2. warp-per-row        : one warp per row, 32B per lane   (the fp32 mc shape)
//   3. lane-per-(row,col)  : 8 rows x 4 cols per warp, 16B per lane, 8 lines
//                            per instruction                 (the int8 tile shape)

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include <cuda_runtime.h>

namespace {

#define CU_CHECK(call)                                                       \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(e_), __FILE__, __LINE__);        \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void probe_seq(const uint4* __restrict__ w, float* __restrict__ sink, size_t n16) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t acc = 0;
    for (; i < n16; i += stride) {
        const uint4 v = w[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xdeadbeefu) *sink = static_cast<float>(acc);
}

// one warp per row, 128B coalesced per super-block: lane reads 1 byte per 32B group
__global__ void probe_row(const uint8_t* __restrict__ w, float* __restrict__ sink,
                          int row_bytes, int out) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out) return;
    const uint8_t* wr = w + static_cast<size_t>(row) * row_bytes;
    uint32_t acc = 0;
    for (int off = 16 + lane; off < row_bytes; off += 32) acc += wr[off];
    if (acc == 0xdeadbeefu) *sink = static_cast<float>(acc);
}

// the B11 int8 tile shape: warp = 8 rows x 4 columns, each lane loads 16B per
// (sub-block, column-group) from a different row -> 8 lines per instruction
template <int RW, int NC>
__global__ void probe_tile(const uint8_t* __restrict__ w, float* __restrict__ sink,
                           int row_bytes, int out) {
    const int lane = threadIdx.x & 31;
    const int r_local = lane / NC;
    const int row = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RW + r_local;
    if (row >= out) return;
    const uint8_t* wr = w + static_cast<size_t>(row) * row_bytes;
    uint32_t acc = 0;
    for (int off = 0; off + 16 <= row_bytes; off += 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(wr + off);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xdeadbeefu) *sink = static_cast<float>(acc);
}

// Compute roof probe: 8 independent chains so the loop measures THROUGHPUT,
// not latency. If dp4a retires at half the FMA rate (Ada: 64 INT32 lanes vs 128
// FP32 lanes per SM), then 4 int8-MACs per dp4a gives exactly 2x the MAC rate of
// one FMA -- which is the ceiling any dp4a gemv can reach, no matter how the
// memory side is tuned.
__global__ void probe_dp4a(float* __restrict__ sink, int iters) {
    const int a = static_cast<int>(threadIdx.x) | 1;
    const int b = 0x01010101;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (int i = 0; i < iters; ++i) {
        c0 = __dp4a(a, b, c0); c1 = __dp4a(a, b, c1);
        c2 = __dp4a(a, b, c2); c3 = __dp4a(a, b, c3);
        c4 = __dp4a(a, b, c4); c5 = __dp4a(a, b, c5);
        c6 = __dp4a(a, b, c6); c7 = __dp4a(a, b, c7);
    }
    const int s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    if (s == 123456789) *sink = static_cast<float>(s);
}

__global__ void probe_fma(float* __restrict__ sink, int iters) {
    const float a = static_cast<float>(threadIdx.x) * 1e-8f + 1.0f;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (int i = 0; i < iters; ++i) {
        c0 = fmaf(a, c0, 1.0f); c1 = fmaf(a, c1, 1.0f);
        c2 = fmaf(a, c2, 1.0f); c3 = fmaf(a, c3, 1.0f);
        c4 = fmaf(a, c4, 1.0f); c5 = fmaf(a, c5, 1.0f);
        c6 = fmaf(a, c6, 1.0f); c7 = fmaf(a, c7, 1.0f);
    }
    const float s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    if (s == 123456789.0f) *sink = s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: b11-roof <q4km.gguf> [iters]\n");
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    try {
        // ---- compute roof: dp4a vs FMA instruction throughput ----
        {
            float* csink = nullptr;
            CU_CHECK(cudaMalloc(&csink, sizeof(float)));
            const int blocks = 24 * 8, threads = 256, loops = 40000;
            cudaEvent_t ea, eb;
            CU_CHECK(cudaEventCreate(&ea));
            CU_CHECK(cudaEventCreate(&eb));

            // ⚠️ Measurement methodology, fixed in B14 (docs/38 §7.11).
            //
            // This used to take a SINGLE timed launch of each probe after a single
            // warm-up launch. The fma kernel runs ~2.5 ms, which on a laptop GPU with
            // aggressive clock scaling is too short to reach a steady boosted clock,
            // while the dp4a kernel runs ~3.3x longer and therefore averages a
            // different (higher) clock. The quotient was consequently dominated by
            // clock ramp rather than pipe throughput: five consecutive runs of the
            // SAME binary gave 3.58x / 2.07x / 1.31x / 1.91x / 3.19x, with the dp4a
            // rate steady (63-79 G warp-instr/s) and the fma rate swinging 76-196.
            // test_b11.sh gated on that quotient, so the gate was a coin flip.
            //
            // Fix: interleave the two probes so they see the same clock trajectory,
            // and take the BEST of REPS -- the least throttled sample, which is what
            // a pipe-ceiling claim actually means. The spread is printed so a reader
            // can see how much clock noise is in play.
            const int REPS = 5;
            float best_fma = 1e30f, best_dp4a = 1e30f, worst_fma = 0.0f, worst_dp4a = 0.0f;
            for (int rep = 0; rep < REPS; ++rep) {
                float ms = 0.0f;
                probe_fma<<<blocks, threads>>>(csink, loops);   // warm, untimed
                CU_CHECK(cudaDeviceSynchronize());
                CU_CHECK(cudaEventRecord(ea));
                probe_fma<<<blocks, threads>>>(csink, loops);
                CU_CHECK(cudaEventRecord(eb));
                CU_CHECK(cudaEventSynchronize(eb));
                CU_CHECK(cudaEventElapsedTime(&ms, ea, eb));
                if (ms < best_fma) best_fma = ms;
                if (ms > worst_fma) worst_fma = ms;

                probe_dp4a<<<blocks, threads>>>(csink, loops);  // warm, untimed
                CU_CHECK(cudaDeviceSynchronize());
                CU_CHECK(cudaEventRecord(ea));
                probe_dp4a<<<blocks, threads>>>(csink, loops);
                CU_CHECK(cudaEventRecord(eb));
                CU_CHECK(cudaEventSynchronize(eb));
                CU_CHECK(cudaEventElapsedTime(&ms, ea, eb));
                if (ms < best_dp4a) best_dp4a = ms;
                if (ms > worst_dp4a) worst_dp4a = ms;
            }
            const float ms_fma = best_fma, ms_dp4a = best_dp4a;

            const double warp_instr = static_cast<double>(blocks) * (threads / 32) * loops * 8.0;
            const double fma_mac = warp_instr * 32.0 / (ms_fma * 1e-3);        // 1 MAC/lane
            const double dp4a_mac = warp_instr * 32.0 * 4.0 / (ms_dp4a * 1e-3);  // 4 MAC/lane
            std::printf("%-34s %8.2f ms  %7.1f G warp-instr/s  %7.3f T MAC/s   (best of %d; spread %.2f-%.2f ms)\n",
                        "fma  (8 indep chains)", ms_fma, warp_instr / (ms_fma * 1e-3) / 1e9,
                        fma_mac / 1e12, REPS, best_fma, worst_fma);
            std::printf("%-34s %8.2f ms  %7.1f G warp-instr/s  %7.3f T MAC/s   (best of %d; spread %.2f-%.2f ms)\n",
                        "dp4a (8 indep chains)", ms_dp4a, warp_instr / (ms_dp4a * 1e-3) / 1e9,
                        dp4a_mac / 1e12, REPS, best_dp4a, worst_dp4a);
            // Three different numbers, and the distinction is the whole point:
            //   * the PIPE ceiling is fixed by silicon: 64 INT32 lanes vs 128 FP32
            //     lanes per SM, times 4 int8 MACs per dp4a => exactly 2.00x. This is
            //     defined against the NOMINAL 128 FP32 MAC/SM/clk.
            //   * the MEASURED fma rate undershoots that nominal (this probe's FFMA
            //     mix carries register-bank pressure and loop overhead: 104-114 of
            //     128 MAC/SM/clk depending on the session).
            //   * the ACHIEVED ratio below divides dp4a by the MEASURED fma, NOT by
            //     the nominal 128. So it can legitimately read slightly ABOVE 2.00x
            //     (observed 2.00x and 2.17x in two sessions with identical code).
            //     Never gate an acceptance check on this quotient -- the hardware
            //     claim is the 2.00x pipe ceiling, and that is measured absolutely.
            // The >=1000 tok/s DoD needs 2.85x, which is above the pipe ceiling --
            // that is why no dp4a tuning can reach it.
            std::printf("%-34s achieved dp4a/fma MAC ratio = %.2fx\n", "implication",
                        dp4a_mac / fma_mac);
            std::printf("%-34s   (denominator = MEASURED fma, which undershoots the nominal"
                        " 128 MAC/SM/clk;\n", "");
            std::printf("%-34s    hence this quotient may exceed 2.00x without violating silicon)\n", "");
            std::printf("%-34s pipe ceiling = 2.00x vs the NOMINAL 128 FP32 MAC/SM/clk"
                        " (64 INT32 lanes x 4 MAC/dp4a);\n", "");
            std::printf("%-34s   the >=1000 tok/s DoD needs more than 2x (docs/34 7.1)\n", "");
            CU_CHECK(cudaEventDestroy(ea));
            CU_CHECK(cudaEventDestroy(eb));
            CU_CHECK(cudaFree(csink));
        }
        std::printf("\n");

        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        const ri::TensorInfo* t = f.tensor("blk.0.ffn_gate.weight");   // 2048 x 6144 Q4_K
        const int in = static_cast<int>(t->dims[0]);
        const int out = static_cast<int>(t->dims[1]);
        const size_t bytes = t->n_bytes;
        const int row_bytes = in / 256 * 144;
        std::printf("shape %dx%d  %zu bytes  row_bytes=%d\n", in, out, bytes, row_bytes);

        uint8_t* dw = nullptr;
        float* dsink = nullptr;
        CU_CHECK(cudaMalloc(&dw, bytes));
        CU_CHECK(cudaMalloc(&dsink, sizeof(float)));
        CU_CHECK(cudaMemcpy(dw, f.tensor_data(*t), bytes, cudaMemcpyHostToDevice));

        auto report = [&](const char* label, double ms) {
            std::printf("%-34s %9.3f us/call  %7.1f GB/s\n", label, ms * 1000.0 / iters,
                        bytes * iters / (ms * 1e-3) / 1e9);
        };

        cudaEvent_t a, b;
        CU_CHECK(cudaEventCreate(&a));
        CU_CHECK(cudaEventCreate(&b));
        auto bench = [&](auto fn) -> double {
            fn();
            CU_CHECK(cudaDeviceSynchronize());
            CU_CHECK(cudaEventRecord(a));
            for (int i = 0; i < iters; ++i) fn();
            CU_CHECK(cudaEventRecord(b));
            CU_CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CU_CHECK(cudaEventElapsedTime(&ms, a, b));
            return static_cast<double>(ms);
        };

        report("1. sequential (memcpy shape)",
               bench([&] { probe_seq<<<1024, 256>>>(reinterpret_cast<const uint4*>(dw), dsink, bytes / 16); }));
        report("2. warp-per-row (fp32 mc shape)",
               bench([&] { probe_row<<<(out + 3) / 4, 128>>>(dw, dsink, row_bytes, out); }));
        report("3a. int8 tile 8x4", bench([&] { probe_tile<8, 4><<<(out + 31) / 32, 128>>>(dw, dsink, row_bytes, out); }));
        report("3b. int8 tile 4x8", bench([&] { probe_tile<4, 8><<<(out + 15) / 16, 128>>>(dw, dsink, row_bytes, out); }));
        report("3c. int8 tile 2x16", bench([&] { probe_tile<2, 16><<<(out + 7) / 8, 128>>>(dw, dsink, row_bytes, out); }));
        report("3d. int8 tile 32x1", bench([&] { probe_tile<32, 1><<<(out + 127) / 128, 128>>>(dw, dsink, row_bytes, out); }));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        cudaFree(dw);
        cudaFree(dsink);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "b11-roof: error: %s\n", e.what());
        return 1;
    }
}
