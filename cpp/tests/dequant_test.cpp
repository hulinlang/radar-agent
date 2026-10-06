// dequant-test: B5c unit cross-check. Dequantizes whole tensors from the
// Q4_K_M file with our kernels and with libggml's reference implementations,
// and compares element-wise. Built in the guest (see tests/test_b5.sh).
//
//   dequant-test <q4km.gguf>

#include "gguf.h"
#include "quants.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

extern "C" {
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
}

namespace {

int check(const ri::GGUFFile& f, const char* name) {
    const ri::TensorInfo* t = f.tensor(name);
    if (!t) {
        std::fprintf(stderr, "tensor %s missing\n", name);
        return 1;
    }
    int64_t n = 1;
    for (uint64_t d : t->dims) n *= static_cast<int64_t>(d);
    const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));
    std::vector<float> ours(n), ref(n);
    if (t->ggml_type == 12) {
        ri::dequantize_row_q4_K(raw, ours.data(), n);
        ::dequantize_row_q4_K(raw, ref.data(), n);
    } else if (t->ggml_type == 14) {
        ri::dequantize_row_q6_K(raw, ours.data(), n);
        ::dequantize_row_q6_K(raw, ref.data(), n);
    } else {
        std::fprintf(stderr, "tensor %s has unexpected type %u\n", name, t->ggml_type);
        return 1;
    }
    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(ours[i]) - ref[i]));
    }
    const bool ok = max_abs == 0.0;
    std::printf("%-28s type=%u n=%lld max_abs_diff=%g %s\n", name, t->ggml_type,
                static_cast<long long>(n), max_abs, ok ? "EXACT" : "MISMATCH");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dequant-test <q4km.gguf>\n");
        return 2;
    }
    try {
        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        int fails = 0;
        fails += check(f, "blk.0.attn_q.weight");     // Q4_K
        fails += check(f, "blk.0.ffn_gate.weight");   // Q4_K
        fails += check(f, "blk.27.ffn_down.weight");  // Q4_K or Q6_K by position
        fails += check(f, "blk.0.attn_v.weight");     // Q6_K
        fails += check(f, "token_embd.weight");       // Q6_K
        std::printf("DEQUANT: %s\n", fails == 0 ? "all exact" : "MISMATCHES");
        return fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dequant-test: error: %s\n", e.what());
        return 1;
    }
}
