// gemv-bench: standalone microbenchmark for the quantized gemv kernels on the
// real weight shapes of the Q4_K_M model. Prints effective GB/s per shape.
//   gemv-bench <q4km.gguf> [iters] [--no-int8]
//
// B11 adds the int8-activation route: same shapes, activations quantized to
// 32-value symmetric int8 groups, dp4a dot products. It sweeps the warp tile
// (rows x columns) because the tile is a pure memory-shape knob -- it does not
// change the arithmetic, only which addresses a warp touches (docs/34 §7).

#include "gguf.h"
#include "gpu_quants.h"
#include "int8_act.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    CU_CHECK(cudaMalloc(&dx, static_cast<size_t>(16) * in * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy, static_cast<size_t>(16) * out * sizeof(float)));
    CU_CHECK(cudaMemcpy(dw, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(dx, 0, static_cast<size_t>(16) * in * sizeof(float)));

    const bool q4 = t->ggml_type == 12;
    auto run = [&](int ncol) {
        if (ncol <= 1) {
            if (q4) ri::gpuq::gemv_q4k(dw, dx, dy, in, out);
            else ri::gpuq::gemv_q6k(dw, dx, dy, in, out);
        } else {
            if (q4) ri::gpuq::gemv_q4k_mc(dw, dx, dy, in, out, ncol);
            else ri::gpuq::gemv_q6k_mc(dw, dx, dy, in, out, ncol);
        }
    };
    for (int ncol : {1, 4, 8, 16}) {
        run(ncol);  // warmup
        CU_CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CU_CHECK(cudaEventCreate(&a));
        CU_CHECK(cudaEventCreate(&b));
        CU_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) run(ncol);
        CU_CHECK(cudaEventRecord(b));
        CU_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CU_CHECK(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        const double gbps = static_cast<double>(t->n_bytes) * iters / (ms * 1e-3) / 1e9;
        std::printf("%-26s %s %5dx%-6d ncol=%2d  %8.2f MB  %8.3f us/call  %6.1f GB/s\n",
                    name, q4 ? "Q4_K" : "Q6_K", in, out, ncol, t->n_bytes / 1e6,
                    ms * 1000.0 / iters, gbps);
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    return 0;
}

// B11: int8-activation route, swept over the warp tile. Only Q4_K has an int8
// kernel (docs/34 §6 keeps Q6_K on the fp32 route until the measurement says
// otherwise). `ncol` columns are quantized; the layout always carries 16.
int bench_int8(const ri::GGUFFile& f, const char* name, int iters) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) { std::fprintf(stderr, "tensor %s missing\n", name); return 1; }
    if (t->ggml_type != 12) return 0;   // Q4_K only
    const int in = static_cast<int>(t->dims[0]);
    const int out = static_cast<int>(t->dims[1]);
    constexpr int LC = 16;

    uint8_t* dw = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    int8_t* dqs = nullptr;
    ri::Q8_1Ds* dds = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dx, static_cast<size_t>(LC) * in * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy, static_cast<size_t>(LC) * out * sizeof(float)));
    CU_CHECK(cudaMalloc(&dqs, static_cast<size_t>(LC) * in));
    CU_CHECK(cudaMalloc(&dds, static_cast<size_t>(LC) * (in / 32) * sizeof(ri::Q8_1Ds)));
    CU_CHECK(cudaMemcpy(dw, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));

    // deterministic activations, same shape as the reference generator
    std::vector<float> x(static_cast<size_t>(LC) * in);
    for (int c = 0; c < LC; ++c)
        for (int i = 0; i < in; ++i)
            x[static_cast<size_t>(c) * in + i] =
                (sinf(i * 0.017f) + 0.5f * cosf(i * 0.043f)) * (0.1f + 0.3f * (c % 5));
    CU_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    ri::gpuq::quantize_q8_1(dx, dqs, dds, in, LC);

    const int tiles[][2] = {{2, 16}, {4, 8}, {8, 4}, {16, 2}, {32, 1}};
    for (const auto& tile : tiles) {
        ri::gpuq::gemv_q4k_q8_1_tile(dw, dqs, dds, dy, in, out, LC, tile[0], tile[1]);
        CU_CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CU_CHECK(cudaEventCreate(&a));
        CU_CHECK(cudaEventCreate(&b));
        CU_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i)
            ri::gpuq::gemv_q4k_q8_1_tile(dw, dqs, dds, dy, in, out, LC, tile[0], tile[1]);
        CU_CHECK(cudaEventRecord(b));
        CU_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CU_CHECK(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        const double gbps = static_cast<double>(t->n_bytes) * iters / (ms * 1e-3) / 1e9;
        std::printf("%-26s %s %5dx%-6d int8 tile=%2dx%-2d (ncol=16)  %8.3f us/call  %6.1f GB/s\n",
                    name, "Q4_K", in, out, tile[0], tile[1], ms * 1000.0 / iters, gbps);
    }
    // quantizer cost on its own (it runs once per gemv call in the engine)
    {
        cudaEvent_t a, b;
        CU_CHECK(cudaEventCreate(&a));
        CU_CHECK(cudaEventCreate(&b));
        CU_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) ri::gpuq::quantize_q8_1(dx, dqs, dds, in, LC);
        CU_CHECK(cudaEventRecord(b));
        CU_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CU_CHECK(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        std::printf("%-26s %s %5dx%-6d quantize_q8_1              %8.3f us/call\n",
                    name, "Q4_K", in, out, ms * 1000.0 / iters);
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dqs);
    cudaFree(dds);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gemv-bench <q4km.gguf> [iters] [--no-int8]\n");
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    bool int8 = true;
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], "--no-int8") == 0) int8 = false;
    try {
        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        int rc = 0;
        rc += bench(f, "blk.0.attn_q.weight", iters);     // Q4_K 2048x2048
        rc += bench_int8(f, "blk.0.attn_q.weight", iters);
        rc += bench(f, "blk.0.attn_k.weight", iters);     // Q4_K 2048x1024
        rc += bench_int8(f, "blk.0.attn_k.weight", iters);
        rc += bench(f, "blk.0.ffn_gate.weight", iters);   // Q4_K 2048x6144
        rc += bench_int8(f, "blk.0.ffn_gate.weight", iters);
        rc += bench(f, "blk.0.attn_v.weight", iters);     // Q6_K 2048x1024
        rc += bench(f, "blk.27.ffn_down.weight", iters);  // Q6_K 6144x2048
        rc += bench(f, "token_embd.weight", iters / 10);  // Q6_K 2048x151936
        if (!int8) { /* int8 sections already skipped for Q6_K */ }
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemv-bench: error: %s\n", e.what());
        return 1;
    }
}
