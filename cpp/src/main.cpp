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
#include "sampler.h"
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

#ifdef USE_CUDA
// B10: one-line KV pool report. `worst` is what the pre-B10 layout would have
// reserved (one contiguous kv_cap region per slot); `alloc` is what paging
// actually allocates. The ratio is the memory win.
void kv_pool_report(const ri::GpuModel& gpu, const char* mode) {
    const double alloc = static_cast<double>(gpu.kv_bytes()) / 1048576.0;
    const double worst = static_cast<double>(gpu.kv_worst_case_bytes()) / 1048576.0;
    std::fprintf(stderr,
                 "kv pool [B10/%s]: block=%d num_blocks=%d used=%d | allocated=%.1f MiB "
                 "| pre-B10-equivalent=%.1f MiB | saved=%.1fx\n",
                 mode, gpu.block_size(), gpu.num_blocks(), gpu.used_blocks(),
                 alloc, worst, alloc > 0.0 ? worst / alloc : 0.0);
}
#endif

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt, ids_out, teacher_path, backend = "omp";
    std::vector<std::string> prompt_files;
    std::string ids_out_prefix;
    int n_predict = 100;
    int kv_cap = 0;
    int max_batch = 8;
    int num_blocks = 0;   // B10: 0 = auto (pre-B10 equivalent)
    bool batch_prefill = true;
    ri::Sampler sampler;
    std::vector<int32_t> bans;
    auto parse_bans = [&](const std::string& s) {
        size_t b = 0;
        while (b < s.size()) {
            size_t e = s.find(',', b);
            bans.push_back(std::stoi(s.substr(b, e == std::string::npos ? e : e - b)));
            if (e == std::string::npos) break;
            b = e + 1;
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--model") model_path = need("--model");
        else if (a == "--prompt") prompt = need("--prompt");
        else if (a == "--prompt-file") prompt_files.push_back(need("--prompt-file"));
        else if (a == "--n-predict") n_predict = std::stoi(need("--n-predict"));
        else if (a == "--kv-cap") kv_cap = std::stoi(need("--kv-cap"));
        else if (a == "--max-batch") max_batch = std::stoi(need("--max-batch"));
        else if (a == "--num-blocks") num_blocks = std::stoi(need("--num-blocks"));
        else if (a == "--ids-out-prefix") ids_out_prefix = need("--ids-out-prefix");
        else if (a == "--ids-out") ids_out = need("--ids-out");
        else if (a == "--teacher-file") teacher_path = need("--teacher-file");
        else if (a == "--backend") backend = need("--backend");
        else if (a == "--no-batch-prefill") batch_prefill = false;
        else if (a == "--temp") sampler.temp = std::stof(need("--temp"));
        else if (a == "--top-k") sampler.top_k = std::stoi(need("--top-k"));
        else if (a == "--top-p") sampler.top_p = std::stof(need("--top-p"));
        else if (a == "--seed") sampler.seed(std::stoull(need("--seed")));
        else if (a == "--ban") parse_bans(need("--ban"));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (prompt_files.size() == 1) prompt = read_file(prompt_files[0]);
    if (model_path.empty() || (prompt.empty() && prompt_files.size() <= 1)) {
        std::fprintf(stderr, "usage: radar-infer --model <gguf> (--prompt <text> | --prompt-file <path>) "
                             "[--n-predict N] [--kv-cap N] [--ids-out FILE]\n");
        return 2;
    }
    try {
        ri::GGUFFile f = ri::GGUFFile::open(model_path);
        ri::Tokenizer tok = ri::Tokenizer::from_gguf(f);

#ifdef USE_CUDA
        if (backend == "cuda" && prompt_files.size() > 1) {
            // B8b continuous batching: KV slot pool + per-step padded decode
            // batch (padding rows run on a scratch slot and are discarded).
            struct Req {
                std::vector<int32_t> prompt, gen;
                int pos = 0;
                int32_t next = 0;
            };
            std::vector<Req> reqs(prompt_files.size());
            int max_len = kv_cap;  // >0 = user override; else grow to fit all
            for (size_t i = 0; i < prompt_files.size(); ++i) {
                reqs[i].prompt = tok.encode(read_file(prompt_files[i]));
                if (reqs[i].prompt.empty())
                    throw std::runtime_error("empty prompt: " + prompt_files[i]);
                max_len = std::max(max_len,
                                   static_cast<int>(reqs[i].prompt.size()) + n_predict + 8);
            }
            ri::GpuModel gpu = ri::GpuModel::load(f, max_len, max_batch, num_blocks);
            kv_pool_report(gpu, "batch");
            const int B = gpu.max_batch();
            if (B <= 1) throw std::runtime_error("max_batch degraded to 1 (VRAM); batch mode needs >1");
            std::fprintf(stderr, "batch mode: %zu requests, max_batch=%d, kv_cap=%d, eos=%d\n",
                         reqs.size(), B, max_len, tok.eos());

            std::vector<int> slot_req(B, -1);
            size_t next_req = 0;
            std::vector<float> logits(gpu.config().vocab);
            double prefill_s = 0.0;
            int reused_total = 0;   // B10b: tokens served from the prefix cache
            auto assign = [&](int s) {
                if (next_req >= reqs.size()) return;
                Req& r = reqs[next_req];
                // B10b: the slot is being handed to a new request, so release the
                // previous one's blocks and forget its token history.
                gpu.begin_sequence(s);
                const auto a = Clock::now();
                gpu.prefill_batch(r.prompt.data(), static_cast<int>(r.prompt.size()),
                                  logits.data(), s);
                prefill_s += secs(a, Clock::now());
                reused_total += gpu.last_reused_tokens();
                r.next = argmax(logits);
                r.pos = static_cast<int>(r.prompt.size());
                slot_req[s] = static_cast<int>(next_req);
                ++next_req;
            };
            for (int s = 0; s < B; ++s) assign(s);

            std::vector<int32_t> btok(B), bnext(B);
            std::vector<int> bpos(B), bslot(B);
            std::vector<char> emitted(B, 0);  // slot's pending next already pushed?
            int active = 0;
            for (int s = 0; s < B; ++s) if (slot_req[s] >= 0) ++active;
            double decode_prefill_s = 0.0;
            auto finish = [&](int s) {
                Req& r = reqs[slot_req[s]];
                std::fprintf(stderr, "req %d done (slot %d): prompt %zu tok, gen %zu tok%s\n",
                             slot_req[s], s, r.prompt.size(), r.gen.size(),
                             r.next == tok.eos() ? " (eos)" : "");
                if (!ids_out_prefix.empty())
                    write_ids(ids_out_prefix + std::to_string(slot_req[s]) + ".ids",
                              r.prompt, r.gen);
                slot_req[s] = -1;
                emitted[s] = 0;
                --active;
                if (next_req < reqs.size()) {
                    const double pre = prefill_s;
                    assign(s);
                    decode_prefill_s += prefill_s - pre;
                    ++active;
                }
            };
            const auto t0 = Clock::now();
            while (active > 0) {
                // emit pending predictions first (matches the single-sequence
                // loop: push, then step). A request assigned mid-flight has an
                // un-emitted prediction (emitted=0) and must not be stepped
                // yet — that was the lost-first-token bug.
                for (int s = 0; s < B; ++s) {
                    if (slot_req[s] < 0 || emitted[s]) continue;
                    Req& r = reqs[slot_req[s]];
                    if (r.next == tok.eos()) {
                        finish(s);
                    } else {
                        r.gen.push_back(r.next);
                        emitted[s] = 1;
                        if (static_cast<int>(r.gen.size()) >= n_predict) finish(s);
                    }
                }
                if (active == 0) break;
                for (int s = 0; s < B; ++s) {
                    if (slot_req[s] >= 0 && emitted[s]) {
                        const Req& r = reqs[slot_req[s]];
                        btok[s] = r.next;
                        bpos[s] = r.pos;
                        bslot[s] = s;
                    } else {
                        btok[s] = 0;      // padding row: scratch slot, discarded
                        bpos[s] = 0;
                        bslot[s] = B;
                    }
                }
                gpu.decode_batch(btok.data(), bpos.data(), bslot.data(), bnext.data());
                for (int s = 0; s < B; ++s) {
                    if (slot_req[s] < 0 || !emitted[s]) continue;
                    Req& r = reqs[slot_req[s]];
                    r.next = bnext[s];
                    ++r.pos;
                    emitted[s] = 0;
                }
            }
            const double wall = secs(t0, Clock::now());
            const size_t total_gen = [&] {
                size_t t = 0;
                for (const Req& r : reqs) t += r.gen.size();
                return t;
            }();
            std::fprintf(stderr,
                         "batch decode: %zu tokens in %.2fs -> %.1f tok/s aggregate\n"
                         "  (decode steps wall %.2fs; interleaved prefill %.2fs; initial prefill %.2fs)\n"
                         "  [b10] prefix cache: reused_prompt_tokens=%d hits=%d misses=%d\n",
                         total_gen, wall, total_gen / wall,
                         wall - decode_prefill_s, decode_prefill_s, prefill_s - decode_prefill_s,
                         reused_total, gpu.prefix_hits(), gpu.prefix_misses());
            for (size_t i = 0; i < reqs.size(); ++i)
                std::printf("=== req %zu ===\n%s\n", i, tok.decode(reqs[i].gen).c_str());
            return 0;
        }
#endif
        if (prompt_files.size() > 1)
            throw std::runtime_error("multi-prompt batch mode requires --backend cuda");

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
            ri::GpuModel gpu = ri::GpuModel::load(f, kv_cap, 1, num_blocks);
            kv_pool_report(gpu, "single");
            const auto t0 = Clock::now();
            std::vector<float> logits(gpu.config().vocab);
            std::vector<float> scratch;
            int32_t next = 0;
            if (batch_prefill) {
                // B8a: weights are read once per 16-token chunk instead of
                // once per token; bitwise-identical KV/logits state.
                gpu.prefill_batch(ids.data(), static_cast<int>(ids.size()), logits.data());
                next = ri::sample_from(sampler, bans, logits.data(), logits.size(), scratch);
            } else {
                for (size_t i = 0; i < ids.size(); ++i) {
                    gpu.forward_logits(ids[i], static_cast<int>(i), logits.data());
                    next = ri::sample_from(sampler, bans, logits.data(), logits.size(), scratch);
                }
            }
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
                gpu.forward_logits(next, pos++, logits.data());
                next = ri::sample_from(sampler, bans, logits.data(), logits.size(), scratch);
            }
            const auto t2 = Clock::now();
            // B10a-4: snapshot AFTER the run — this is the number that shows the
            // point of paging ("used 8 blocks while 256 were reserved").
            kv_pool_report(gpu, "single-final");
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
        std::vector<float> scratch;
        int pos = static_cast<int>(ids.size());
        int32_t next = ri::sample_from(sampler, bans, logits->data(), logits->size(), scratch);
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
            next = ri::sample_from(sampler, bans, logits->data(), logits->size(), scratch);
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
