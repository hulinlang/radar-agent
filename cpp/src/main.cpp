// radar-infer: end-to-end CLI (B4).
//   radar-infer --model <gguf> (--prompt <text> | --prompt-file <path>)
//               [--n-predict 100] [--kv-cap N] [--ids-out FILE]
// Greedy decoding only; stops at EOS or n-predict. Prints decoded text and
// prefill/decode timings; --ids-out writes "PROMPT ..."/"GEN ..." id lines
// for the llama.cpp cross-check harness.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#ifdef USE_CUDA
#include "gpu_model.h"
#endif

namespace {

int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = static_cast<int32_t>(i);
    }
    return best;
}

// RADAR_DEBUG_TOPK=K: per generated step, print top-K (id:logit) to stderr.
void debug_topk(const std::vector<float>& v, int step) {
    const char* env = std::getenv("RADAR_DEBUG_TOPK");
    if (!env) return;
    const int k = std::atoi(env);
    std::vector<int32_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int32_t>(i);
    const int kk = k < static_cast<int>(v.size()) ? k : static_cast<int>(v.size());
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int32_t a, int32_t b) { return v[a] > v[b]; });
    std::fprintf(stderr, "step %d:", step);
    for (int i = 0; i < kk; ++i) std::fprintf(stderr, " %d:%.6f", idx[i], v[idx[i]]);
    std::fprintf(stderr, "\n");
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_ids(const std::string& path, const std::vector<int32_t>& prompt,
               const std::vector<int32_t>& gen) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "PROMPT";
    for (int32_t id : prompt) out << " " << id;
    out << "\nGEN";
    for (int32_t id : gen) out << " " << id;
    out << "\n";
}

std::vector<int32_t> read_id_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<int32_t> ids;
    std::string tok;
    while (in >> tok) {
        size_t b = 0;
        while (b < tok.size()) {
            size_t e = tok.find(',', b);
            std::string part = tok.substr(b, e == std::string::npos ? e : e - b);
            if (!part.empty() && part != "PROMPT" && part != "GEN") ids.push_back(std::stoi(part));
            if (e == std::string::npos) break;
            b = e + 1;
        }
    }
    return ids;
}

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt, ids_out, teacher_path, backend = "omp";
    int n_predict = 100;
    int kv_cap = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--model") model_path = need("--model");
        else if (a == "--prompt") prompt = need("--prompt");
        else if (a == "--prompt-file") prompt = read_file(need("--prompt-file"));
        else if (a == "--n-predict") n_predict = std::stoi(need("--n-predict"));
        else if (a == "--kv-cap") kv_cap = std::stoi(need("--kv-cap"));
        else if (a == "--ids-out") ids_out = need("--ids-out");
        else if (a == "--teacher-file") teacher_path = need("--teacher-file");
        else if (a == "--backend") backend = need("--backend");
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || prompt.empty()) {
        std::fprintf(stderr, "usage: radar-infer --model <gguf> (--prompt <text> | --prompt-file <path>) "
                             "[--n-predict N] [--kv-cap N] [--ids-out FILE]\n");
        return 2;
    }
    try {
        ri::GGUFFile f = ri::GGUFFile::open(model_path);
        ri::Tokenizer tok = ri::Tokenizer::from_gguf(f);

        const std::vector<int32_t> ids = tok.encode(prompt);
        if (ids.empty()) throw std::runtime_error("prompt encodes to zero tokens");
        if (kv_cap <= 0) kv_cap = static_cast<int>(ids.size()) + n_predict + 8;
        const bool use_blas = backend == "blas";
        const bool use_cuda = backend == "cuda";
        if (backend != "omp" && !use_blas && !use_cuda) throw std::runtime_error("unknown --backend " + backend);
        std::fprintf(stderr, "prompt: %zu tokens, kv_cap=%d, eos=%d, backend=%s\n",
                     ids.size(), kv_cap, tok.eos(), backend.c_str());

#ifdef USE_CUDA
        if (use_cuda) {
            ri::GpuModel gpu = ri::GpuModel::load(f, kv_cap);
            const auto t0 = Clock::now();
            int32_t next = 0;
            for (size_t i = 0; i < ids.size(); ++i) next = gpu.forward_argmax(ids[i], static_cast<int>(i));
            const auto t1 = Clock::now();

            if (!teacher_path.empty()) {
                const std::vector<int32_t> teacher = read_id_list(teacher_path);
                int agree = 0, pos = static_cast<int>(ids.size());
                int32_t cur = next;
                for (int32_t forced : teacher) {
                    if (cur == forced) ++agree;
                    cur = gpu.forward_argmax(forced, pos++);
                }
                std::printf("teacher-agreement: %d/%zu (%.1f%%)\n", agree, teacher.size(),
                            100.0 * agree / teacher.size());
                return 0;
            }

            std::vector<int32_t> gen;
            int pos = static_cast<int>(ids.size());
            while (static_cast<int>(gen.size()) < n_predict && next != tok.eos()) {
                gen.push_back(next);
                next = gpu.forward_argmax(next, pos++);
            }
            const auto t2 = Clock::now();
            const double pre_s = secs(t0, t1), dec_s = secs(t1, t2);
            std::fprintf(stderr,
                         "prefill: %zu tokens in %.2fs (%.1f tok/s)\n"
                         "decode:  %zu tokens in %.2fs (%.2f tok/s)\n",
                         ids.size(), pre_s, ids.size() / pre_s,
                         gen.size(), dec_s, dec_s > 0 ? gen.size() / dec_s : 0.0);
            std::printf("%s\n", tok.decode(gen).c_str());
            if (!ids_out.empty()) write_ids(ids_out, ids, gen);
            return 0;
        }
#else
        if (use_cuda) throw std::runtime_error("built without CUDA support");
#endif

        ri::Model model = ri::Model::load(f, kv_cap, false, use_blas);

        const auto t0 = Clock::now();
        const std::vector<float>* logits = nullptr;
        for (size_t i = 0; i < ids.size(); ++i) logits = &model.forward(ids[i], static_cast<int>(i));
        const auto t1 = Clock::now();

        std::vector<int32_t> gen;
        int pos = static_cast<int>(ids.size());
        int32_t next = argmax(*logits);
        debug_topk(*logits, 0);

        if (!teacher_path.empty()) {
            // Teacher forcing: feed the reference trajectory, score top-1 agreement.
            const std::vector<int32_t> teacher = read_id_list(teacher_path);
            int agree = 0, total = 0;
            int32_t cur = next;  // prediction for teacher[0]
            for (size_t i = 0; i < teacher.size(); ++i) {
                if (cur == teacher[i]) ++agree;
                ++total;
                logits = &model.forward(teacher[i], pos++);
                cur = argmax(*logits);
            }
            std::printf("teacher-agreement: %d/%d (%.1f%%)\n", agree, total,
                        100.0 * agree / total);
            return 0;
        }

        while (static_cast<int>(gen.size()) < n_predict && next != tok.eos()) {
            gen.push_back(next);
            logits = &model.forward(next, pos++);
            next = argmax(*logits);
            debug_topk(*logits, static_cast<int>(gen.size()));
        }
        const auto t2 = Clock::now();

        const double pre_s = secs(t0, t1), dec_s = secs(t1, t2);
        std::fprintf(stderr,
                     "prefill: %zu tokens in %.2fs (%.1f tok/s)\n"
                     "decode:  %zu tokens in %.2fs (%.2f tok/s)\n",
                     ids.size(), pre_s, ids.size() / pre_s,
                     gen.size(), dec_s, dec_s > 0 ? gen.size() / dec_s : 0.0);

        std::printf("%s\n", tok.decode(gen).c_str());
        if (!ids_out.empty()) write_ids(ids_out, ids, gen);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "radar-infer: error: %s\n", e.what());
        return 1;
    }
}
