#include "model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "f16.h"
#include "ggml_types.h"
#include "gguf.h"
#include "ops.h"

namespace ri {
namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("model: " + msg);
}

int kv_u32(const GGUFFile& f, const std::string& key) {
    const Value* v = f.kv(key);
    if (!v) fail("missing metadata " + key);
    if (v->type == ValueType::UINT32 || v->type == ValueType::UINT64) {
        return static_cast<int>(std::get<uint64_t>(v->data));
    }
    return static_cast<int>(std::get<int64_t>(v->data));
}

float kv_f32(const GGUFFile& f, const std::string& key) {
    const Value* v = f.kv(key);
    if (!v) fail("missing metadata " + key);
    return static_cast<float>(std::get<double>(v->data));
}

std::string kv_str(const GGUFFile& f, const std::string& key) {
    const Value* v = f.kv(key);
    if (!v || v->type != ValueType::STRING) fail("missing metadata " + key);
    return std::get<std::string>(v->data);
}

struct TensorView {
    const TensorInfo* info;
    const void* data;
};

TensorView need_tensor(const GGUFFile& f, const std::string& name) {
    const TensorInfo* t = f.tensor(name);
    if (!t) fail("missing tensor " + name);
    return {t, f.tensor_data(*t)};
}

void expect_dims(const std::string& name, const TensorInfo* t, std::vector<uint64_t> dims) {
    if (t->dims != dims) {
        std::string got, want;
        for (uint64_t d : t->dims) got += std::to_string(d) + ",";
        for (uint64_t d : dims) want += std::to_string(d) + ",";
        fail("tensor " + name + " dims [" + got + "] != expected [" + want + "]");
    }
}

const float* as_f32(const GGUFFile& f, const std::string& name, std::vector<uint64_t> dims) {
    TensorView v = need_tensor(f, name);
    if (v.info->ggml_type != 0) fail("tensor " + name + " must be F32");
    expect_dims(name, v.info, std::move(dims));
    return static_cast<const float*>(v.data);
}

const uint16_t* as_f16(const GGUFFile& f, const std::string& name, std::vector<uint64_t> dims) {
    TensorView v = need_tensor(f, name);
    if (v.info->ggml_type != 1) fail("tensor " + name + " must be F16");
    expect_dims(name, v.info, std::move(dims));
    return static_cast<const uint16_t*>(v.data);
}

}  // namespace

Model Model::load(const GGUFFile& f, int kv_cap, bool trace) {
    Model m;
    const std::string arch = kv_str(f, "general.architecture");
    const std::string p = arch + ".";

    ModelConfig& c = m.cfg_;
    c.n_layer = kv_u32(f, p + "block_count");
    c.hidden = kv_u32(f, p + "embedding_length");
    c.n_head = kv_u32(f, p + "attention.head_count");
    c.n_kv_head = kv_u32(f, p + "attention.head_count_kv");
    c.head_dim = kv_u32(f, p + "attention.key_length");
    c.ffn = kv_u32(f, p + "feed_forward_length");
    c.rms_eps = kv_f32(f, p + "attention.layer_norm_rms_epsilon");
    c.rope_theta = kv_f32(f, p + "rope.freq_base");
    if (c.n_head % c.n_kv_head != 0) fail("n_head not divisible by n_kv_head");
    if (kv_cap <= 0) fail("kv_cap must be positive");

    const uint64_t H = static_cast<uint64_t>(c.hidden);
    {
        TensorView embd = need_tensor(f, "token_embd.weight");
        if (embd.info->ggml_type != 1) fail("tensor token_embd.weight must be F16");
        if (embd.info->dims.size() != 2 || embd.info->dims[0] != H) {
            fail("token_embd.weight has unexpected dims");
        }
        c.vocab = static_cast<int>(embd.info->dims[1]);
        m.token_embd_ = static_cast<const uint16_t*>(embd.data);
    }

    {
        const float* on = as_f32(f, "output_norm.weight", {H});
        m.output_norm_.assign(on, on + H);
    }

    m.layers_.reserve(c.n_layer);
    m.norm_store_.reserve(c.n_layer * 4);
    auto keep_f32 = [&](const float* src, int n) -> const float* {
        m.norm_store_.emplace_back(src, src + n);
        return m.norm_store_.back().data();
    };
    const uint64_t Q = static_cast<uint64_t>(c.n_head) * c.head_dim;
    const uint64_t KV = static_cast<uint64_t>(c.n_kv_head) * c.head_dim;
    const uint64_t FF = static_cast<uint64_t>(c.ffn);
    for (int i = 0; i < c.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        LayerWeights w;
        w.attn_norm = keep_f32(as_f32(f, b + "attn_norm.weight", {H}), c.hidden);
        w.wq = as_f16(f, b + "attn_q.weight", {H, Q});
        w.wk = as_f16(f, b + "attn_k.weight", {H, KV});
        w.wv = as_f16(f, b + "attn_v.weight", {H, KV});
        w.q_norm = keep_f32(as_f32(f, b + "attn_q_norm.weight", {static_cast<uint64_t>(c.head_dim)}), c.head_dim);
        w.k_norm = keep_f32(as_f32(f, b + "attn_k_norm.weight", {static_cast<uint64_t>(c.head_dim)}), c.head_dim);
        w.wo = as_f16(f, b + "attn_output.weight", {Q, H});
        w.ffn_norm = keep_f32(as_f32(f, b + "ffn_norm.weight", {H}), c.hidden);
        w.wg = as_f16(f, b + "ffn_gate.weight", {H, FF});
        w.wu = as_f16(f, b + "ffn_up.weight", {H, FF});
        w.wd = as_f16(f, b + "ffn_down.weight", {FF, H});
        m.layers_.push_back(w);
    }

    m.kv_cap_ = kv_cap;
    const size_t kv_elems = static_cast<size_t>(c.n_layer) * 2 * c.n_kv_head * kv_cap * c.head_dim;
    m.kv_.assign(kv_elems, 0.0f);

    const int q_dim = c.n_head * c.head_dim;
    const int kv_dim = c.n_kv_head * c.head_dim;
    m.x_.assign(c.hidden, 0.0f);
    m.h_.assign(c.hidden, 0.0f);
    m.q_.assign(q_dim, 0.0f);
    m.k_.assign(kv_dim, 0.0f);
    m.v_.assign(kv_dim, 0.0f);
    m.att_.assign(q_dim, 0.0f);
    m.scores_.assign(kv_cap, 0.0f);
    m.gate_.assign(c.ffn, 0.0f);
    m.up_.assign(c.ffn, 0.0f);
    m.ffn_.assign(c.hidden, 0.0f);
    m.logits_.assign(c.vocab, 0.0f);

    m.trace_ = trace;
    if (trace) {
        m.trace_embed_.assign(c.hidden, 0.0f);
        m.trace_layers_.assign(c.n_layer, std::vector<float>(c.hidden, 0.0f));
        m.trace_final_.assign(c.hidden, 0.0f);
    }
    return m;
}

float* Model::kv_ptr(int layer, int kv, int head, int pos) {
    const ModelConfig& c = cfg_;
    const size_t idx = (((static_cast<size_t>(layer) * 2 + kv) * c.n_kv_head + head) *
                        static_cast<size_t>(kv_cap_) + pos) * c.head_dim;
    return kv_.data() + idx;
}

const std::vector<float>& Model::forward(int32_t token, int pos) {
    const ModelConfig& c = cfg_;
    if (token < 0 || token >= c.vocab) fail("token id out of range");
    if (pos < 0 || pos >= kv_cap_) fail("position exceeds kv_cap");
    const int q_dim = c.n_head * c.head_dim;
    const int kv_dim = c.n_kv_head * c.head_dim;
    const int group = c.n_head / c.n_kv_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    // Embedding row lookup (F16 -> F32, exact).
    const uint16_t* erow = token_embd_ + static_cast<size_t>(token) * c.hidden;
    for (int i = 0; i < c.hidden; ++i) x_[i] = f16_to_f32(erow[i]);
    if (trace_) trace_embed_ = x_;

    for (int l = 0; l < c.n_layer; ++l) {
        const LayerWeights& w = layers_[l];

        rmsnorm(x_.data(), w.attn_norm, h_.data(), c.hidden, c.rms_eps);
        matvec_f16(w.wq, h_.data(), q_.data(), c.hidden, q_dim);
        matvec_f16(w.wk, h_.data(), k_.data(), c.hidden, kv_dim);
        matvec_f16(w.wv, h_.data(), v_.data(), c.hidden, kv_dim);

        // QK-RMSNorm per head, before RoPE.
        for (int hh = 0; hh < c.n_head; ++hh) {
            rmsnorm(q_.data() + hh * c.head_dim, w.q_norm, q_.data() + hh * c.head_dim,
                    c.head_dim, c.rms_eps);
        }
        for (int hh = 0; hh < c.n_kv_head; ++hh) {
            rmsnorm(k_.data() + hh * c.head_dim, w.k_norm, k_.data() + hh * c.head_dim,
                    c.head_dim, c.rms_eps);
        }

        rope_neox(q_.data(), c.n_head, c.head_dim, pos, c.rope_theta);
        rope_neox(k_.data(), c.n_kv_head, c.head_dim, pos, c.rope_theta);

        for (int hh = 0; hh < c.n_kv_head; ++hh) {
            std::memcpy(kv_ptr(l, 0, hh, pos), k_.data() + hh * c.head_dim,
                        c.head_dim * sizeof(float));
            std::memcpy(kv_ptr(l, 1, hh, pos), v_.data() + hh * c.head_dim,
                        c.head_dim * sizeof(float));
        }

        for (int qh = 0; qh < c.n_head; ++qh) {
            const int kvh = qh / group;
            const float* q = q_.data() + qh * c.head_dim;
            for (int t = 0; t <= pos; ++t) {
                const float* kk = kv_ptr(l, 0, kvh, t);
                float acc = 0.0f;
                for (int i = 0; i < c.head_dim; ++i) acc += q[i] * kk[i];
                scores_[t] = acc * scale;
            }
            softmax(scores_.data(), pos + 1);
            float* out = att_.data() + qh * c.head_dim;
            for (int i = 0; i < c.head_dim; ++i) out[i] = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* vv = kv_ptr(l, 1, kvh, t);
                const float p = scores_[t];
                for (int i = 0; i < c.head_dim; ++i) out[i] += p * vv[i];
            }
        }

        matvec_f16(w.wo, att_.data(), ffn_.data(), q_dim, c.hidden);
        for (int i = 0; i < c.hidden; ++i) x_[i] += ffn_[i];

        rmsnorm(x_.data(), w.ffn_norm, h_.data(), c.hidden, c.rms_eps);
        matvec_f16(w.wg, h_.data(), gate_.data(), c.hidden, c.ffn);
        matvec_f16(w.wu, h_.data(), up_.data(), c.hidden, c.ffn);
        for (int i = 0; i < c.ffn; ++i) gate_[i] = silu(gate_[i]) * up_[i];
        matvec_f16(w.wd, gate_.data(), ffn_.data(), c.ffn, c.hidden);
        for (int i = 0; i < c.hidden; ++i) x_[i] += ffn_[i];

        if (trace_) trace_layers_[l] = x_;
    }

    rmsnorm(x_.data(), output_norm_.data(), h_.data(), c.hidden, c.rms_eps);
    if (trace_) trace_final_ = h_;
    matvec_f16(token_embd_, h_.data(), logits_.data(), c.hidden, c.vocab);
    return logits_;
}

}  // namespace ri
