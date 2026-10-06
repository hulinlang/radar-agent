// llama_ref_generate: reference driver linked against libllama (B4 cross-check).
// Loads the same F16 GGUF, greedy-generates, prints "PROMPT ..."/"GEN ..." id
// lines to stdout -- the exact contract radar-infer --ids-out produces.
//   llama_ref_generate --model <gguf> --prompt-file <path> [--n-predict 100]
// Built in the WSL guest against ~/llama.cpp (see tests/test_b4.sh).

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int32_t argmax(const float* v, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

void print_ids(const char* tag, const std::vector<llama_token>& ids) {
    std::printf("%s", tag);
    for (llama_token t : ids) std::printf(" %d", t);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt;
    int n_predict = 100;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--model") model_path = need("--model");
        else if (a == "--prompt-file") prompt = read_file(need("--prompt-file"));
        else if (a == "--n-predict") n_predict = std::stoi(need("--n-predict"));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || prompt.empty()) {
        std::fprintf(stderr, "usage: llama_ref_generate --model <gguf> --prompt-file <path> [--n-predict N]\n");
        return 2;
    }
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;  // CPU reference semantics
        llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) throw std::runtime_error("failed to load model");
        const llama_vocab* vocab = llama_model_get_vocab(model);

        std::vector<llama_token> toks(prompt.size() + 16);
        const int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                     toks.data(), static_cast<int32_t>(toks.size()),
                                     /*add_special=*/false, /*parse_special=*/true);
        if (n < 0) throw std::runtime_error("tokenize failed");
        toks.resize(n);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(n + n_predict + 8);
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("failed to create context");

        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> gen;

        llama_batch batch = llama_batch_get_one(toks.data(), n);
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill decode failed");
        int32_t next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);

        while (static_cast<int>(gen.size()) < n_predict && !llama_vocab_is_eog(vocab, next)) {
            gen.push_back(next);
            batch = llama_batch_get_one(&gen.back(), 1);
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("decode failed");
            next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }

        print_ids("PROMPT", toks);
        print_ids("GEN", gen);

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llama_ref_generate: error: %s\n", e.what());
        return 1;
    }
}
