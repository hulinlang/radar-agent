// simd-test: B6a unit cross-check. Compares scalar vs AVX2 matvec kernels on
// real tensors (Q4_K_M file for Q4_K/Q6_K, F16 file for F16) with a
// deterministic pseudo-random activation vector.
//   simd-test <q4km.gguf> <f16.gguf>

#include "f16.h"
#include "gguf.h"
#include "ops.h"
#include "quants.h"
#include "simd_avx2.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

std::vector<float> make_x(int n, unsigned seed) {
    std::vector<float> x(n);
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = static_cast<float>(static_cast<int>(s >> 8) % 65536) / 32768.0f - 1.0f;
    }
    return x;
}

int check(const char* name, const uint8_t* w, uint32_t type, int in, int out) {
    const std::vector<float> x = make_x(in, 12345u + in);
    std::vector<float> ys(out), ya(out);
    // double-precision ground truth per row, and per-row term magnitude
    std::vector<double> ref(out), scale(out, 0.0);
    std::vector<float> wb(in);
    for (int j = 0; j < out; ++j) {
        const uint8_t* row;
        switch (type) {
            case 1: row = w + static_cast<size_t>(j) * in * 2; break;
            case 12: row = w + static_cast<size_t>(j) * (in / 256) * 144; break;
            default: row = w + static_cast<size_t>(j) * (in / 256) * 210; break;
        }
        // dequant/convert row to float, then dot in double
        if (type == 12) ri::dequantize_row_q4_K(row, wb.data(), in);
        else if (type == 14) ri::dequantize_row_q6_K(row, wb.data(), in);
        else {
            const uint16_t* r = reinterpret_cast<const uint16_t*>(row);
            for (int i = 0; i < in; ++i) wb[i] = ri::f16_to_f32(r[i]);
        }
        double dot = 0.0, mag = 0.0;
        for (int i = 0; i < in; ++i) {
            dot += static_cast<double>(wb[i]) * x[i];
            mag += std::fabs(static_cast<double>(wb[i]) * x[i]);
        }
        ref[j] = dot;
        scale[j] = mag;
    }
    switch (type) {
        case 1:
            ri::matvec_f16(reinterpret_cast<const uint16_t*>(w), x.data(), ys.data(), in, out);
            ri::matvec_f16_avx2(reinterpret_cast<const uint16_t*>(w), x.data(), ya.data(), in, out);
            break;
        case 12:
            ri::matvec_q4_k(w, x.data(), ys.data(), in, out);
            ri::matvec_q4_k_avx2(w, x.data(), ya.data(), in, out);
            break;
        case 14:
            ri::matvec_q6_k(w, x.data(), ys.data(), in, out);
            ri::matvec_q6_k_avx2(w, x.data(), ya.data(), in, out);
            break;
        default:
            std::fprintf(stderr, "bad type\n");
            return 1;
    }
    // error normalized by per-row |term| sum: pure f32 rounding sits ~1e-7..1e-6
    double err_s = 0.0, err_a = 0.0;
    for (int j = 0; j < out; ++j) {
        const double denom = std::max(1e-6, scale[j]);
        err_s = std::max(err_s, std::fabs(ys[j] - ref[j]) / denom);
        err_a = std::max(err_a, std::fabs(ya[j] - ref[j]) / denom);
    }
    const bool ok = err_a < 1e-5 && err_a <= 10.0 * err_s + 1e-7;
    std::printf("%-26s type=%u [%dx%d] err_scalar=%.3g err_avx2=%.3g %s\n", name, type, in, out,
                err_s, err_a, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}

int check_tensor(const ri::GGUFFile& f, const char* name) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) {
        std::fprintf(stderr, "tensor %s missing\n", name);
        return 1;
    }
    const int in = static_cast<int>(t->dims[0]);
    const int out = static_cast<int>(t->dims[1]);
    return check(name, static_cast<const uint8_t*>(f.tensor_data(*t)), t->ggml_type, in, out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: simd-test <q4km.gguf> <f16.gguf>\n");
        return 2;
    }
    try {
        if (!ri::have_avx2()) {
            std::fprintf(stderr, "built without AVX2\n");
            return 1;
        }
        int fails = 0;
        {
            ri::GGUFFile q4 = ri::GGUFFile::open(argv[1]);
            fails += check_tensor(q4, "blk.0.attn_q.weight");
            fails += check_tensor(q4, "blk.0.ffn_gate.weight");
            fails += check_tensor(q4, "blk.0.attn_v.weight");
            fails += check_tensor(q4, "blk.0.ffn_down.weight");
        }
        {
            ri::GGUFFile f16 = ri::GGUFFile::open(argv[2]);
            fails += check_tensor(f16, "blk.0.attn_q.weight");
            fails += check_tensor(f16, "blk.0.ffn_down.weight");
        }
        std::printf("SIMD: %s\n", fails == 0 ? "all ok" : "MISMATCHES");
        return fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "simd-test: error: %s\n", e.what());
        return 1;
    }
}
