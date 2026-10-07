// gemv-bench: standalone microbenchmark for the quantized gemv kernels on the
// real weight shapes of the Q4_K_M model. Prints effective GB/s per shape.
//   gemv-bench <q4km.gguf> [iters]

#include "gguf.h"
#include "gpu_quants.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

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

int bench(const ri::GGUFFile& f, const char* name, int iters) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) { std::fprintf(stderr, "tensor %s missing\n", name); return 1; }
    const int in = static_cast<int>(t->dims[0]);
    const int out = static_cast<int>(t->dims[1]);
    uint8_t* dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dx, in * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy, out * sizeof(float)));
    CU_CHECK(cudaMemcpy(dw, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(dx, 0, in * sizeof(float)));

    const bool q4 = t->ggml_type == 12;
    auto launch = [&] {
        if (q4) ri::gpuq::gemv_q4k(dw, dx, dy, in, out);
        else ri::gpuq::gemv_q6k(dw, dx, dy, in, out);
    };
    launch();  // warmup
    CU_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CU_CHECK(cudaEventCreate(&a));
    CU_CHECK(cudaEventCreate(&b));
    CU_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CU_CHECK(cudaEventRecord(b));
    CU_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CU_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    const double gbps = static_cast<double>(t->n_bytes) * iters / (ms * 1e-3) / 1e9;
    std::printf("%-26s %s %5dx%-6d %8.2f MB  %7.3f us/call  %6.1f GB/s\n", name,
                q4 ? "Q4_K" : "Q6_K", in, out, t->n_bytes / 1e6,
                ms * 1000.0 / iters, gbps);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gemv-bench <q4km.gguf> [iters]\n");
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    try {
        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        int rc = 0;
        rc += bench(f, "blk.0.attn_q.weight", iters);     // Q4_K 2048x2048
        rc += bench(f, "blk.0.attn_k.weight", iters);     // Q4_K 2048x1024
        rc += bench(f, "blk.0.ffn_gate.weight", iters);   // Q4_K 2048x6144
        rc += bench(f, "blk.0.attn_v.weight", iters);     // Q6_K 2048x1024
        rc += bench(f, "blk.27.ffn_down.weight", iters);  // Q6_K 6144x2048
        rc += bench(f, "token_embd.weight", iters / 10);  // Q6_K 2048x151936
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemv-bench: error: %s\n", e.what());
        return 1;
    }
}
