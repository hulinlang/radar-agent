// radar-infer: end-to-end CLI (B4).
//   radar-infer --model <gguf> (--prompt <text> | --prompt-file <path>)
//               [--n-predict 100] [--kv-cap N] [--ids-out FILE]
// Greedy decoding only; stops at EOS or n-predict. Prints decoded text and
// prefill/decode timings; --ids-out writes "PROMPT ..."/"GEN ..." id lines
// for the llama.cpp cross-check harness.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "model.h"
#include "tokenizer.h"

namespace {

int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = static_cast<int32_t>(i);
    }
    return best;
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

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt, ids_out;
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
        std::fprintf(stderr, "prompt: %zu tokens, kv_cap=%d, eos=%d\n", ids.size(), kv_cap, tok.eos());

        ri::Model model = ri::Model::load(f, kv_cap);

        const auto t0 = Clock::now();
        const std::vector<float>* logits = nullptr;
        for (size_t i = 0; i < ids.size(); ++i) logits = &model.forward(ids[i], static_cast<int>(i));
        const auto t1 = Clock::now();

        std::vector<int32_t> gen;
        int pos = static_cast<int>(ids.size());
        int32_t next = argmax(*logits);
        while (static_cast<int>(gen.size()) < n_predict && next != tok.eos()) {
            gen.push_back(next);
            logits = &model.forward(next, pos++);
            next = argmax(*logits);
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
