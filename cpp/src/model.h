#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ri {

class GGUFFile;

struct ModelConfig {
    int n_layer = 0;
    int hidden = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    int ffn = 0;
    int vocab = 0;
    float rms_eps = 0.0f;
    float rope_theta = 0.0f;
};

// The Qwen3-family text decoder: pre-norm transformer with GQA, per-head
// QK-RMSNorm (before RoPE), NeoX RoPE, SwiGLU, tied embeddings.
// Weights are zero-copy views into the mmap'd GGUF; all compute is F32.
// Single code path: prefill is a loop of decode-style steps over the KV cache.
class Model {
public:
    // kv_cap: max tokens the KV cache can hold (prompt + generated).
    // trace: when true, each layer's output hidden state is copied into trace().
    static Model load(const GGUFFile& f, int kv_cap, bool trace = false);

    const ModelConfig& config() const { return cfg_; }

    // Runs one token at absolute position pos; returns the logits buffer
    // (size = vocab). Invalidated by the next call.
    const std::vector<float>& forward(int32_t token, int pos);

    // Trace accessors (valid when constructed with trace=true):
    // embedding output and per-layer outputs of the most recent forward call.
    const std::vector<float>& trace_embedding() const { return trace_embed_; }
    const std::vector<float>& trace_layer(int l) const { return trace_layers_[l]; }
    const std::vector<float>& trace_final_norm() const { return trace_final_; }

private:
    struct LayerWeights {
        const float* attn_norm;    // F32 [hidden]
        const uint16_t* wq;        // F16 [hidden, n_head*head_dim]
        const uint16_t* wk;        // F16 [hidden, n_kv_head*head_dim]
        const uint16_t* wv;
        const float* q_norm;       // F32 [head_dim]
        const float* k_norm;       // F32 [head_dim]
        const uint16_t* wo;        // F16 [n_head*head_dim, hidden]
        const float* ffn_norm;     // F32 [hidden]
        const uint16_t* wg;        // F16 [hidden, ffn]
        const uint16_t* wu;        // F16 [hidden, ffn]
        const uint16_t* wd;        // F16 [ffn, hidden]
    };

    ModelConfig cfg_;
    const uint16_t* token_embd_ = nullptr;  // F16 [hidden, vocab]
    std::vector<float> output_norm_;        // F32 [hidden]
    std::vector<LayerWeights> layers_;
    // Norm weights converted to F32 at load (small); owned here.
    std::vector<std::vector<float>> norm_store_;

    // KV cache: [layer][k|v][kv_head][pos][head_dim], contiguous.
    std::vector<float> kv_;
    int kv_cap_ = 0;
    float* kv_ptr(int layer, int kv, int head, int pos);

    // Scratch buffers (allocated once).
    std::vector<float> x_, h_, q_, k_, v_, att_, scores_, gate_, up_, ffn_, logits_;

    bool trace_ = false;
    std::vector<float> trace_embed_;
    std::vector<std::vector<float>> trace_layers_;
    std::vector<float> trace_final_;
};

}  // namespace ri
