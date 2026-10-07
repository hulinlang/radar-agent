// gpu-quants-test: B7 unit cross-check for the CUDA quantized kernels.
//   1. dequant_q4k/q6k (GPU) vs dequantize_row_q4_K/q6_K (CPU): bit-exact.
//   2. gemv_q4k/q6k (GPU, fused) vs matvec_q4_k/q6_k (CPU): allclose
//      (fp32 accumulation order differs).
//   3. embed_row_q6k vs CPU row dequant: bit-exact.
//
//   gpu-quants-test <q4km.gguf>

#include "gguf.h"
#include "gpu_quants.h"
#include "quants.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
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

int64_t numel(const ri::TensorInfo* t) {
    int64_t n = 1;
    for (uint64_t d : t->dims) n *= static_cast<int64_t>(d);
    return n;
}

std::vector<float> cpu_dequant(const ri::TensorInfo* t, const uint8_t* raw) {
    std::vector<float> y(numel(t));
    if (t->ggml_type == 12) ri::dequantize_row_q4_K(raw, y.data(), numel(t));
    else ri::dequantize_row_q6_K(raw, y.data(), numel(t));
    return y;
}

int check_dequant(const ri::GGUFFile& f, const char* name) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) { std::fprintf(stderr, "tensor %s missing\n", name); return 1; }
    if (t->ggml_type != 12 && t->ggml_type != 14) {
        std::fprintf(stderr, "tensor %s unexpected type %u\n", name, t->ggml_type);
        return 1;
    }
    const int64_t n = numel(t);
    const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));
    const std::vector<float> ref = cpu_dequant(t, raw);

    uint8_t* dw = nullptr;
    float* dy = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CU_CHECK(cudaMemcpy(dw, raw, t->n_bytes, cudaMemcpyHostToDevice));
    if (t->ggml_type == 12) ri::gpuq::dequant_q4k(dw, dy, n);
    else ri::gpuq::dequant_q6k(dw, dy, n);
    std::vector<float> got(n);
    CU_CHECK(cudaMemcpy(got.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dw);
    cudaFree(dy);

    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - ref[i]));
    const bool ok = max_abs == 0.0;
    std::printf("dequant %-26s type=%u n=%lld max_abs_diff=%g %s\n", name,
                t->ggml_type, static_cast<long long>(n), max_abs,
                ok ? "EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}

int check_gemv(const ri::GGUFFile& f, const char* name) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) { std::fprintf(stderr, "tensor %s missing\n", name); return 1; }
    const int in = static_cast<int>(t->dims[0]);
    const int out = static_cast<int>(t->dims[1]);
    const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));

    std::vector<float> x(in);
    for (int i = 0; i < in; ++i) x[i] = sinf(i * 0.017f) * 0.8f + cosf(i * 0.043f) * 0.2f;
    std::vector<float> ref(out);
    if (t->ggml_type == 12) ri::matvec_q4_k(raw, x.data(), ref.data(), in, out);
    else ri::matvec_q6_k(raw, x.data(), ref.data(), in, out);

    uint8_t* dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dx, in * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy, out * sizeof(float)));
    CU_CHECK(cudaMemcpy(dw, raw, t->n_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(dx, x.data(), in * sizeof(float), cudaMemcpyHostToDevice));
    if (t->ggml_type == 12) ri::gpuq::gemv_q4k(dw, dx, dy, in, out);
    else ri::gpuq::gemv_q6k(dw, dx, dy, in, out);
    std::vector<float> got(out);
    CU_CHECK(cudaMemcpy(got.data(), dy, out * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);

    double max_abs = 0.0;
    int bad_n = 0;
    for (int i = 0; i < out; ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
        max_abs = std::max(max_abs, diff);
        if (diff > 1e-3 + 1e-4 * std::fabs(static_cast<double>(ref[i]))) ++bad_n;
    }
    const bool ok = bad_n == 0;
    std::printf("gemv    %-26s type=%u %dx%d max_abs_diff=%g viol=%d %s\n", name,
                t->ggml_type, in, out, max_abs, bad_n, ok ? "ALLCLOSE" : "MISMATCH");
    return ok ? 0 : 1;
}

// B8: multi-column gemv must be bitwise-identical per column to single runs.
int check_gemv_mc(const ri::GGUFFile& f, const char* name, int ncol) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) { std::fprintf(stderr, "tensor %s missing\n", name); return 1; }
    const int in = static_cast<int>(t->dims[0]);
    const int out = static_cast<int>(t->dims[1]);
    const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));

    std::vector<float> x(static_cast<size_t>(ncol) * in);
    for (int c = 0; c < ncol; ++c)
        for (int i = 0; i < in; ++i)
            x[static_cast<size_t>(c) * in + i] =
                sinf(i * 0.017f + c * 0.73f) * 0.8f + cosf(i * 0.043f) * 0.2f;

    uint8_t* dw = nullptr;
    float *dx = nullptr, *dy = nullptr, *dy1 = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy, static_cast<size_t>(ncol) * out * sizeof(float)));
    CU_CHECK(cudaMalloc(&dy1, out * sizeof(float)));
    CU_CHECK(cudaMemcpy(dw, raw, t->n_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (t->ggml_type == 12) ri::gpuq::gemv_q4k_mc(dw, dx, dy, in, out, ncol);
    else ri::gpuq::gemv_q6k_mc(dw, dx, dy, in, out, ncol);
    std::vector<float> mc(static_cast<size_t>(ncol) * out);
    CU_CHECK(cudaMemcpy(mc.data(), dy, mc.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int fails = 0;
    for (int c = 0; c < ncol; ++c) {
        if (t->ggml_type == 12)
            ri::gpuq::gemv_q4k(dw, dx + static_cast<size_t>(c) * in, dy1, in, out);
        else
            ri::gpuq::gemv_q6k(dw, dx + static_cast<size_t>(c) * in, dy1, in, out);
        std::vector<float> ref(out);
        CU_CHECK(cudaMemcpy(ref.data(), dy1, out * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < out; ++i) {
            if (std::memcmp(&mc[static_cast<size_t>(c) * out + i], &ref[i], 4) != 0) {
                ++fails;
                break;
            }
        }
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dy1);
    const bool ok = fails == 0;
    std::printf("gemv-mc %-22s type=%u %dx%d ncol=%d %s\n", name, t->ggml_type, in, out,
                ncol, ok ? "BITWISE-PARITY" : "MISMATCH");
    return ok ? 0 : 1;
}

int check_embed(const ri::GGUFFile& f, int token) {
    const ri::TensorInfo* t = f.tensor("token_embd.weight");
    if (!t || t->ggml_type != 14) {
        std::fprintf(stderr, "token_embd.weight missing or not Q6_K\n");
        return 1;
    }
    const int hidden = static_cast<int>(t->dims[0]);
    const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));
    const size_t row_bytes = static_cast<size_t>(hidden) / 256 * 210;
    std::vector<float> ref(hidden);
    ri::dequantize_row_q6_K(raw + static_cast<size_t>(token) * row_bytes,
                            ref.data(), hidden);

    uint8_t* dw = nullptr;
    float* dy = nullptr;
    int* dtok = nullptr;
    CU_CHECK(cudaMalloc(&dw, t->n_bytes));
    CU_CHECK(cudaMalloc(&dy, hidden * sizeof(float)));
    CU_CHECK(cudaMalloc(&dtok, sizeof(int)));
    CU_CHECK(cudaMemcpy(dw, raw, t->n_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(dtok, &token, sizeof(int), cudaMemcpyHostToDevice));
    ri::gpuq::embed_row_q6k(dw, dy, dtok, hidden);
    std::vector<float> got(hidden);
    CU_CHECK(cudaMemcpy(got.data(), dy, hidden * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(dtok);

    double max_abs = 0.0;
    for (int i = 0; i < hidden; ++i)
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - ref[i]));
    const bool ok = max_abs == 0.0;
    std::printf("embed   token=%d hidden=%d max_abs_diff=%g %s\n", token, hidden,
                max_abs, ok ? "EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gpu-quants-test <q4km.gguf>\n");
        return 2;
    }
    try {
        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        int fails = 0;
        fails += check_dequant(f, "blk.0.attn_q.weight");     // Q4_K 2048x2048
        fails += check_dequant(f, "blk.10.ffn_up.weight");    // Q4_K 2048x6144
        fails += check_dequant(f, "blk.0.attn_v.weight");     // Q6_K 2048x1024
        fails += check_dequant(f, "blk.27.ffn_down.weight");  // Q6_K 6144x2048
        fails += check_gemv(f, "blk.0.attn_q.weight");
        fails += check_gemv(f, "blk.10.ffn_up.weight");
        fails += check_gemv(f, "blk.0.attn_v.weight");
        fails += check_gemv(f, "blk.27.ffn_down.weight");
        fails += check_embed(f, 12345);
        fails += check_gemv_mc(f, "blk.0.attn_q.weight", 8);   // Q4_K
        fails += check_gemv_mc(f, "blk.10.ffn_up.weight", 16);  // Q4_K, wide
        fails += check_gemv_mc(f, "blk.0.attn_v.weight", 8);   // Q6_K
        std::printf("GPU-QUANTS: %s\n", fails == 0 ? "all pass" : "FAILURES");
        return fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu-quants-test: error: %s\n", e.what());
        return 1;
    }
}
