// forward-test: B3 acceptance harness.
// usage: forward-test <model.gguf> <golden_dir>
// Replays the golden prompt token-by-token (decode-style + KV cache) and
// compares per-layer hidden states / final norm / logits against the HF
// float32 reference dumps. Thresholds per docs/23.

#include "gguf.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kAtol = 1e-3;
constexpr double kRtol = 1e-4;   // allclose: |diff| <= atol + rtol*|ref|
constexpr double kCosTol = 0.99999;

std::vector<float> read_f32(const std::string& path, size_t expect) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const size_t bytes = static_cast<size_t>(in.tellg());
    if (bytes != expect * sizeof(float)) {
        throw std::runtime_error(path + ": size mismatch, expected " +
                                 std::to_string(expect * 4) + " got " + std::to_string(bytes));
    }
    std::vector<float> v(expect);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// Minimal extraction of "ids": [n, n, ...] from meta.json (no JSON library).
std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const size_t k = s.find("\"ids\"");
    if (k == std::string::npos) throw std::runtime_error("meta.json: no ids");
    const size_t lb = s.find('[', k), rb = s.find(']', k);
    std::vector<int32_t> ids;
    size_t i = lb + 1;
    while (i < rb) {
        char* end = nullptr;
        long v = std::strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) { ++i; continue; }
        ids.push_back(static_cast<int32_t>(v));
        i = static_cast<size_t>(end - s.c_str());
    }
    return ids;
}

int read_int(const std::string& path, const std::string& key) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) throw std::runtime_error("meta.json: no " + key);
    return std::atoi(s.c_str() + s.find(':', k) + 1);
}

struct Diff {
    double max_abs = 0.0;
    double max_scaled = 0.0;  // max |diff| / (atol + rtol*|ref|); <= 1 means pass
    double cosine = 1.0;
};

Diff compare(const float* a, const float* b, size_t n) {
    Diff d;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
        if (diff > d.max_abs) d.max_abs = diff;
        const double scaled = diff / (kAtol + kRtol * std::fabs(static_cast<double>(b[i])));
        if (scaled > d.max_scaled) d.max_scaled = scaled;
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na > 0 && nb > 0) d.cosine = dot / std::sqrt(na * nb);
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: forward-test <model.gguf> <golden_dir>\n");
        return 2;
    }
    try {
        const std::string dir = argv[2];
        const std::string meta = dir + "/meta.json";
        const std::vector<int32_t> ids = read_ids(meta);
        const int T = static_cast<int>(ids.size());
        const int hidden = read_int(meta, "hidden");
        const int model_n_layer = read_int(meta, "model_n_layer");
        const int raw_layers = read_int(meta, "raw_layer_dumps");
        const int vocab = read_int(meta, "vocab");
        const int argmax_want = read_int(meta, "argmax_last");
        std::fprintf(stderr, "golden: T=%d hidden=%d model_layers=%d raw_dumps=%d vocab=%d\n",
                     T, hidden, model_n_layer, raw_layers, vocab);

        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        ri::Model model = ri::Model::load(f, T + 8, /*trace=*/true);
        const ri::ModelConfig& c = model.config();
        if (c.hidden != hidden || c.n_layer != model_n_layer || c.vocab != vocab) {
            throw std::runtime_error("model config mismatch with golden meta");
        }

        const std::vector<float> emb_g = read_f32(dir + "/embedding.f32", static_cast<size_t>(T) * hidden);
        // hs 元组末位是 final_norm 输出，裸最后一层不可直接对拍；
        // layer_XX 文件只覆盖 hs[1..entries-2]（raw_layers 个）。
        const int n_layer = raw_layers;
        std::vector<std::vector<float>> layers_g(n_layer);
        for (int l = 0; l < n_layer; ++l) {
            char name[64];
            std::snprintf(name, sizeof(name), "/layer_%02d.f32", l);
            layers_g[l] = read_f32(dir + name, static_cast<size_t>(T) * hidden);
        }
        const std::vector<float> final_g = read_f32(dir + "/final_norm.f32", static_cast<size_t>(T) * hidden);
        const std::vector<float> logits_g = read_f32(dir + "/logits.f32", vocab);

        int failures = 0;
        std::vector<double> worst_scaled(n_layer, 0.0), worst_abs(n_layer, 0.0);
        double worst_cos = 1.0;

        for (int t = 0; t < T; ++t) {
            const std::vector<float>& logits = model.forward(ids[t], t);

            // embedding: exact (F16 -> F32 is lossless row lookup)
            const Diff de = compare(model.trace_embedding().data(), emb_g.data() + static_cast<size_t>(t) * hidden, hidden);
            if (de.max_abs != 0.0) {
                std::fprintf(stderr, "EMBEDDING nonzero diff at t=%d: %g\n", t, de.max_abs);
                ++failures;
            }
            for (int l = 0; l < n_layer; ++l) {
                const Diff d = compare(model.trace_layer(l).data(),
                                       layers_g[l].data() + static_cast<size_t>(t) * hidden, hidden);
                if (d.max_scaled > worst_scaled[l]) worst_scaled[l] = d.max_scaled;
                if (d.max_abs > worst_abs[l]) worst_abs[l] = d.max_abs;
                if (d.cosine < worst_cos) worst_cos = d.cosine;
            }
            if (t == T - 1) {
                const Diff df = compare(model.trace_final_norm().data(),
                                        final_g.data() + static_cast<size_t>(t) * hidden, hidden);
                std::printf("final_norm: max_abs=%.3g scaled=%.3g cos=%.8f\n",
                            df.max_abs, df.max_scaled, df.cosine);
                if (df.max_scaled > 1.0) { std::fprintf(stderr, "FINAL NORM over tol\n"); ++failures; }
                const Diff dl = compare(logits.data(), logits_g.data(), vocab);
                int argmax_got = 0;
                for (int i = 1; i < vocab; ++i) if (logits[i] > logits[argmax_got]) argmax_got = i;
                std::printf("logits: max_abs=%.3g scaled=%.3g cos=%.8f argmax got=%d want=%d\n",
                            dl.max_abs, dl.max_scaled, dl.cosine, argmax_got, argmax_want);
                if (dl.max_scaled > 1.0 || argmax_got != argmax_want) {
                    std::fprintf(stderr, "LOGITS check failed\n");
                    ++failures;
                }
            }
        }

        std::printf("\nper-layer worst (allclose atol=%.0e rtol=%.0e):\n", kAtol, kRtol);
        for (int l = 0; l < n_layer; ++l) {
            const bool ok = worst_scaled[l] <= 1.0;
            std::printf("  layer_%02d: scaled=%.3g abs=%.3g %s\n", l, worst_scaled[l], worst_abs[l],
                        ok ? "" : "  <-- OVER");
            if (!ok) ++failures;
        }
        std::printf("worst cosine: %.8f (threshold %.5f)\n", worst_cos, kCosTol);
        if (worst_cos < kCosTol) ++failures;

        // Debug: FORWARD_DEBUG_LAYER=N dumps per-position detail for one layer.
        const char* dbg = std::getenv("FORWARD_DEBUG_LAYER");
        if (dbg) {
            const int dl = std::atoi(dbg);
            ri::Model m2 = ri::Model::load(f, T + 8, true);
            for (int t = 0; t < T; ++t) {
                m2.forward(ids[t], t);
                const float* got = m2.trace_layer(dl).data();
                const float* want = layers_g[dl].data() + static_cast<size_t>(t) * hidden;
                const Diff d = compare(got, want, hidden);
                std::printf("debug layer_%02d t=%d: max_abs=%.3g cos=%.6f\n", dl, t, d.max_abs, d.cosine);
                std::vector<size_t> idx(hidden);
                for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&](size_t a, size_t b) {
                    return std::fabs(got[a] - want[a]) > std::fabs(got[b] - want[b]);
                });
                for (int k = 0; k < 3; ++k) {
                    std::printf("    dim %4zu: got %12.6g want %12.6g\n", idx[k], got[idx[k]], want[idx[k]]);
                }
            }
        }

        std::printf("B3 FORWARD: %s\n", failures == 0 ? "all checks passed" : "FAILURES");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forward-test: error: %s\n", e.what());
        return 1;
    }
}
