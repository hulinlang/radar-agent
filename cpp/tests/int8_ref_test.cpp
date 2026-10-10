// B11 layer-1 cross-check: our int8 route vs the independent NumPy reference.
//
//   int8-ref-test <model.gguf> <tag.meta.txt>
//
// Compares, in this order:
//   1. quantize_q8_1_host() vs the reference activation blocks -- EXACT bytes
//      (the quantizer is integer + one half rounding, so "close" is not a pass)
//   2. matvec_q*_k_q8_1_host() vs the reference outputs -- bit-exact fp32
//   3. (only when built with USE_CUDA) gpuq::gemv_*_q8_1 vs the same reference
//
// Step 2/3 are the point of the design: "same parenthesisation, or it does not
// count" (docs/34 3.4). A tolerance here would hide exactly the class of bug
// this layer exists to catch.
//
// Usage is driven by the meta file so the two sides cannot disagree about the
// shapes: python3 cpp/tests/ref_q4_int8.py ... writes it.

#include "gguf.h"
#include "int8_act.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(USE_CUDA)
#include "gpu_quants.h"

#include <cuda_runtime.h>
#endif

namespace {

struct Meta {
    std::string tensor;
    long type = 0, in = 0, out = 0, nb = 0, ncol = 0, nrow = 0;
    uint64_t x_fnv = 0, q8_fnv = 0, y_fnv = 0;
    double rel_act = 0.0, rel_ord = 0.0, y_scale = 0.0;
};

Meta read_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open meta " + path);
    Meta m;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "tensor") is >> m.tensor;
        else if (key == "ggml_type") is >> m.type;
        else if (key == "in") is >> m.in;
        else if (key == "out") is >> m.out;
        else if (key == "nb") is >> m.nb;
        else if (key == "ncol") is >> m.ncol;
        else if (key == "nrow") is >> m.nrow;
        else if (key == "x_fnv") is >> std::hex >> m.x_fnv;
        else if (key == "q8_fnv") is >> std::hex >> m.q8_fnv;
        else if (key == "y_fnv") is >> std::hex >> m.y_fnv;
        else if (key == "int8_vs_fp32act_max_rel") is >> m.rel_act;
        else if (key == "int8_vs_dequantact_max_rel") is >> m.rel_ord;
        else if (key == "y_scale") is >> m.y_scale;
    }
    return m;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(b.data()), n);
    return b;
}

int64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint8_t byte : b) {
        h ^= byte;
        h *= 0x100000001b3ULL;
    }
    return static_cast<int64_t>(h);
}

// Bit-exact compare of two fp32 arrays; reports the first differing element.
std::string tag_of(const std::string& tensor, long type) {
    std::string base = tensor;
    const size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    std::string out;
    for (char ch : base) out.push_back(ch == '.' ? '_' : ch);
    out += (type == 12) ? "_q4k" : "_q6k";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: int8-ref-test <model.gguf> <tag.meta.txt>\n");
        return 2;
    }
    int fails = 0;
    try {
        const Meta m = read_meta(argv[2]);
        if (m.in <= 0 || m.in % 256 != 0 || m.ncol <= 0 || m.nrow <= 0) {
            throw std::runtime_error("bad meta shapes");
        }
        std::string dir = argv[2];
        const size_t slash = dir.find_last_of("/\\");
        dir = (slash == std::string::npos) ? std::string(".") : dir.substr(0, slash);
        const std::string tag = tag_of(m.tensor, m.type);

        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        const ri::TensorInfo* t = f.tensor(m.tensor);
        if (!t) throw std::runtime_error("tensor missing: " + m.tensor);
        if (static_cast<long>(t->ggml_type) != m.type) throw std::runtime_error("type mismatch");
        const uint8_t* raw = static_cast<const uint8_t*>(f.tensor_data(*t));

        const std::vector<uint8_t> xb = slurp(dir + "/" + tag + ".x.f32");
        const std::vector<uint8_t> qb = slurp(dir + "/" + tag + ".q8.bin");
        const std::vector<uint8_t> yb = slurp(dir + "/" + tag + ".y.f32");
        // meta `nb` counts 256-value weight super-blocks; activations are
        // quantized per 32 values, so there are 8x as many of those blocks.
        const size_t nq = static_cast<size_t>(m.ncol) * m.nb * 8;   // 32-value blocks
        const size_t qs_bytes = static_cast<size_t>(m.ncol) * m.in;
        const size_t ds_bytes = nq * sizeof(ri::Q8_1Ds);
        if (xb.size() != static_cast<size_t>(m.ncol) * m.in * 4 ||
            qb.size() != qs_bytes + ds_bytes ||
            yb.size() != static_cast<size_t>(m.ncol) * m.nrow * 4) {
            std::fprintf(stderr, "sizes: x=%zu (want %zu) q8=%zu (want %zu) y=%zu (want %zu)\n",
                         xb.size(), static_cast<size_t>(m.ncol) * m.in * 4,
                         qb.size(), qs_bytes + ds_bytes,
                         yb.size(), static_cast<size_t>(m.ncol) * m.nrow * 4);
            throw std::runtime_error("reference file sizes do not match the meta shapes");
        }
        std::printf("meta %-30s type=%ld in=%ld out=%ld nb=%ld ncol=%ld nrow=%ld\n",
                    m.tensor.c_str(), m.type, m.in, m.out, m.nb, m.ncol, m.nrow);
        std::printf("     payload fnv: x=%016llx q8=%016llx y=%016llx\n",
                    static_cast<unsigned long long>(fnv1a(xb)),
                    static_cast<unsigned long long>(fnv1a(qb)),
                    static_cast<unsigned long long>(fnv1a(yb)));

        const float* x = reinterpret_cast<const float*>(xb.data());

        // ---- 1. activation quantizer, exact bytes ----
        std::vector<int8_t> qsv(qs_bytes);
        std::vector<ri::Q8_1Ds> dsv(nq);
        ri::quantize_q8_1_host(x, static_cast<int>(m.ncol), static_cast<int>(m.in),
                               qsv.data(), dsv.data());
        {
            size_t first = qb.size();
            for (size_t i = 0; i < qs_bytes; ++i) {
                if (qsv[i] != static_cast<int8_t>(qb[i])) { first = i; break; }
            }
            const uint8_t* gotds = reinterpret_cast<const uint8_t*>(dsv.data());
            if (first == qb.size()) {
                for (size_t i = 0; i < ds_bytes; ++i) {
                    if (gotds[i] != qb[qs_bytes + i]) { first = qs_bytes + i; break; }
                }
            }
            if (first == qb.size()) {
                std::printf("quantize_q8_1  %zu blocks  EXACT\n", nq);
            } else if (first < qs_bytes) {
                std::printf("quantize_q8_1  MISMATCH at qs byte %zu (block %zu lane %zu)\n",
                            first, first / 32, first % 32);
                ++fails;
            } else {
                const size_t o = first - qs_bytes;
                std::printf("quantize_q8_1  MISMATCH at ds byte %zu (block %zu field %zu)\n",
                            o, o / 8, o % 8);
                ++fails;
            }
        }

        // ---- 2. host int8 gemv, bit-exact ----
        std::vector<float> got(static_cast<size_t>(m.ncol) * m.out);
        if (m.type == 12) {
            ri::matvec_q4_k_q8_1_host(raw, qsv.data(), dsv.data(), got.data(),
                                      static_cast<int>(m.in), static_cast<int>(m.out),
                                      static_cast<int>(m.ncol));
        } else {
            ri::matvec_q6_k_q8_1_host(raw, qsv.data(), dsv.data(), got.data(),
                                      static_cast<int>(m.in), static_cast<int>(m.out),
                                      static_cast<int>(m.ncol));
        }
        {
            // reference layout is [nrow][ncol]; ours is [ncol][out]
            const float* ref = reinterpret_cast<const float*>(yb.data());
            size_t first = 0, bad = 0;
            double maxabs = 0.0, worst = 0.0;
            size_t worst_c = 0, worst_r = 0;
            for (long c = 0; c < m.ncol; ++c) {
                for (long r = 0; r < m.nrow; ++r) {
                    const float a = got[static_cast<size_t>(c) * m.out + r];
                    const float b = ref[static_cast<size_t>(r) * m.ncol + c];
                    if (std::memcmp(&a, &b, 4) != 0) {
                        if (bad == 0) first = static_cast<size_t>(c) * m.out + r;
                        ++bad;
                        const double d = std::fabs(static_cast<double>(a) - b);
                        if (d > maxabs) { maxabs = d; worst_c = c; worst_r = r; }
                    }
                }
            }
            worst = m.y_scale > 0 ? maxabs / m.y_scale : maxabs;
            if (bad == 0) {
                std::printf("host %s int8 gemv  %ld x %ld x %ld  BIT-EXACT\n",
                            m.type == 12 ? "Q4_K" : "Q6_K", m.ncol, m.in, m.out);
            } else {
                std::printf("host %s int8 gemv  MISMATCH %zu/%ld elems, first idx %zu, "
                            "max|d|=%.6g (%.2e of y_scale) at col %zu row %zu\n",
                            m.type == 12 ? "Q4_K" : "Q6_K", bad,
                            static_cast<long>(m.nrow) * m.ncol, first, maxabs, worst,
                            worst_c, worst_r);
                ++fails;
            }
        }

        std::printf("     numpy: int8 vs fp32-activation %.3e ; vs dequant-activation %.3e\n",
                    m.rel_act, m.rel_ord);

#if defined(USE_CUDA)
        // ---- 3. CUDA int8 path, bit-exact vs the same reference ----
        // 3a. the CUDA *quantizer* too: the engine uses this one, not the host
        //     one, and the first B11 attempt shipped a broken __reduce_add_sync
        //     that this check now catches (it produced garbage sum_q -> garbage
        //     Q4_K offset term -> teacher-agreement 0/50).
        {
            float* dx = nullptr;
            int8_t* dqs = nullptr;
            ri::Q8_1Ds* dds = nullptr;
            auto ck = [](cudaError_t e, const char* what) {
                if (e != cudaSuccess) {
                    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
                }
            };
            ck(cudaMalloc(&dx, xb.size()), "cudaMalloc x");
            ck(cudaMalloc(&dqs, qs_bytes), "cudaMalloc qs");
            ck(cudaMalloc(&dds, ds_bytes), "cudaMalloc ds");
            ck(cudaMemcpy(dx, xb.data(), xb.size(), cudaMemcpyHostToDevice), "memcpy x");
            ri::gpuq::quantize_q8_1(dx, dqs, dds, static_cast<int>(m.in),
                                    static_cast<int>(m.ncol));
            std::vector<int8_t> cqs(qs_bytes);
            std::vector<ri::Q8_1Ds> cds(nq);
            ck(cudaMemcpy(cqs.data(), dqs, qs_bytes, cudaMemcpyDeviceToHost), "memcpy qs back");
            ck(cudaMemcpy(cds.data(), dds, ds_bytes, cudaMemcpyDeviceToHost), "memcpy ds back");
            cudaFree(dx);
            cudaFree(dqs);
            cudaFree(dds);
            size_t bad_qs = 0, bad_ds = 0, first = 0;
            for (size_t i = 0; i < qs_bytes; ++i)
                if (cqs[i] != qsv[i]) { if (!bad_qs) first = i; ++bad_qs; }
            const uint8_t* cdsb = reinterpret_cast<const uint8_t*>(cds.data());
            const uint8_t* rdsb = reinterpret_cast<const uint8_t*>(dsv.data());
            for (size_t i = 0; i < ds_bytes; ++i)
                if (cdsb[i] != rdsb[i]) { if (!bad_ds) first = qs_bytes + i; ++bad_ds; }
            if (bad_qs == 0 && bad_ds == 0) {
                std::printf("cuda quantize_q8_1   %zu blocks  EXACT (qs+ds)\n", nq);
            } else {
                std::printf("cuda quantize_q8_1   MISMATCH qs=%zu ds=%zu bytes, first %zu\n",
                            bad_qs, bad_ds, first);
                ++fails;
            }
        }
        // 3b. the gemv. Only Q4_K has a CUDA int8 kernel: docs/34 section 6 keeps
        //     Q6_K on the fp32 route until D3 says whether it is needed.
        if (m.type != 12) {
            std::printf("cuda Q6_K int8 gemv  SKIPPED (still on the fp32 route)\n");
        } else {
            uint8_t* dw = nullptr;
            int8_t* dqs = nullptr;
            ri::Q8_1Ds* dds = nullptr;
            float* dy = nullptr;
            auto ck = [](cudaError_t e, const char* what) {
                if (e != cudaSuccess) {
                    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
                }
            };
            ck(cudaMalloc(&dw, t->n_bytes), "cudaMalloc w");
            ck(cudaMalloc(&dqs, qs_bytes), "cudaMalloc qs");
            ck(cudaMalloc(&dds, ds_bytes), "cudaMalloc ds");
            ck(cudaMalloc(&dy, got.size() * sizeof(float)), "cudaMalloc y");
            ck(cudaMemcpy(dw, raw, t->n_bytes, cudaMemcpyHostToDevice), "memcpy w");
            ck(cudaMemcpy(dqs, qsv.data(), qs_bytes, cudaMemcpyHostToDevice), "memcpy qs");
            ck(cudaMemcpy(dds, dsv.data(), ds_bytes, cudaMemcpyHostToDevice), "memcpy ds");
            ri::gpuq::gemv_q4k_q8_1(dw, dqs, dds, dy, static_cast<int>(m.in),
                                    static_cast<int>(m.out), static_cast<int>(m.ncol));
            std::vector<float> cg(got.size());
            ck(cudaMemcpy(cg.data(), dy, cg.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "memcpy y back");
            cudaFree(dw);
            cudaFree(dqs);
            cudaFree(dds);
            cudaFree(dy);

            const float* ref = reinterpret_cast<const float*>(yb.data());
            size_t bad = 0, first = 0;
            double maxabs = 0.0;
            size_t wc = 0, wr = 0;
            for (long c = 0; c < m.ncol; ++c) {
                for (long r = 0; r < m.nrow; ++r) {
                    const float a = cg[static_cast<size_t>(c) * m.out + r];
                    const float b = ref[static_cast<size_t>(r) * m.ncol + c];
                    if (std::memcmp(&a, &b, 4) != 0) {
                        if (bad == 0) first = static_cast<size_t>(c) * m.out + r;
                        ++bad;
                        const double d = std::fabs(static_cast<double>(a) - b);
                        if (d > maxabs) { maxabs = d; wc = c; wr = r; }
                    }
                }
            }
            if (bad == 0) {
                std::printf("cuda Q4_K int8 gemv  %ld x %ld x %ld  BIT-EXACT\n",
                            m.ncol, m.in, m.out);
            } else {
                std::printf("cuda Q4_K int8 gemv  MISMATCH %zu/%ld elems, first idx %zu, "
                            "max|d|=%.6g (%.2e of y_scale) at col %zu row %zu\n",
                            bad, static_cast<long>(m.nrow) * m.ncol, first, maxabs,
                            m.y_scale > 0 ? maxabs / m.y_scale : maxabs, wc, wr);
                ++fails;
            }
        }
#endif

        std::printf("INT8-REF %s: %s\n", m.tensor.c_str(), fails == 0 ? "all pass" : "FAILURES");
        return fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "int8-ref-test: error: %s\n", e.what());
        return 1;
    }
}
