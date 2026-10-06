// B6b CUDA backend. F16 weights in VRAM; decode gemv via custom kernel
// (F16 storage, F32 activations/accumulation; memory-bound shapes — see
// docs/26 and the cublas note in docs/26 results). Small orchestration
// kernels handwritten. Only F16 tensors on this path (Q4 on GPU is later).

#include "gpu_model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gguf.h"
#include "model.h"  // ModelConfig

namespace ri {
namespace {

#define CU_CHECK(call)                                                     \
    do {                                                                   \
        cudaError_t e_ = (call);                                           \
        if (e_ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error ") +           \
                cudaGetErrorString(e_) + " at " __FILE__ ":" +             \
                std::to_string(__LINE__));                                 \
        }                                                                  \
    } while (0)

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("gpu_model: " + msg); }

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

// ---------------- device upload helpers ----------------

__half* upload_f16(const GGUFFile& f, const std::string& name, int64_t in, int64_t out) {
    const TensorInfo* t = f.tensor(name);
    if (!t) fail("missing tensor " + name);
    if (t->ggml_type != 1) fail("tensor " + name + " must be F16 for the CUDA path");
    if (t->dims.size() != 2 || static_cast<int64_t>(t->dims[0]) != in ||
        static_cast<int64_t>(t->dims[1]) != out) {
        fail("tensor " + name + " unexpected dims");
    }
    __half* dev = nullptr;
    CU_CHECK(cudaMalloc(&dev, t->n_bytes));
    CU_CHECK(cudaMemcpy(dev, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));
    return dev;
}

float* upload_f32(const GGUFFile& f, const std::string& name, int64_t n) {
    const TensorInfo* t = f.tensor(name);
    if (!t) fail("missing tensor " + name);
    if (t->ggml_type != 0) fail("tensor " + name + " must be F32");
    (void)n;  // size taken from the tensor itself
    float* dev = nullptr;
    CU_CHECK(cudaMalloc(&dev, t->n_bytes));
    CU_CHECK(cudaMemcpy(dev, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));
    return dev;
}

// ---------------- kernels ----------------

__global__ void embed_row(const __half* embd, float* x, int token, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[i] = __half2float(embd[static_cast<size_t>(token) * hidden + i]);
}

// one block per call; n up to a few thousand
__global__ void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float smem[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    smem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * scale * w[i];
}

// per-head RMSNorm over head_dim: one block per head
__global__ void rmsnorm_heads(float* v, const float* w, int head_dim, float eps) {
    float* x = v + blockIdx.x * head_dim;
    __shared__ float smem[128];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += x[i] * x[i];
    smem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) x[i] = x[i] * scale * w[i];
}

// NeoX rope: one block per head (q heads then k heads), head_dim/2 threads
__global__ void rope_qk(float* q, float* k, int n_q_head, int head_dim, int pos, float theta) {
    const int h = blockIdx.x;
    float* v = (h < n_q_head) ? q + h * head_dim : k + (h - n_q_head) * head_dim;
    const int half = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const float freq = 1.0f / powf(theta, 2.0f * i / head_dim);
    const float c = cosf(pos * freq), s = sinf(pos * freq);
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
}

__global__ void kv_store(const float* k, const float* v, float* kv,
                         size_t layer_k_stride, size_t layer_v_stride,
                         size_t head_stride, int n_kv_head, int head_dim, int pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv_head * head_dim;
    if (i >= total) return;
    const int h = i / head_dim, d = i % head_dim;
    kv[layer_k_stride + h * head_stride + static_cast<size_t>(pos) * head_dim + d] = k[i];
    kv[layer_v_stride + h * head_stride + static_cast<size_t>(pos) * head_dim + d] = v[i];
}

// decode attention: one block per q head
__global__ void attention(const float* q, const float* kv, float* att, float* scores,
                          size_t layer_k_stride, size_t layer_v_stride, size_t head_stride,
                          int n_head, int n_kv_head, int head_dim, int pos, float scale) {
    const int qh = blockIdx.x;
    const int kvh = qh / (n_head / n_kv_head);
    const float* q_h = q + qh * head_dim;
    const float* k_base = kv + layer_k_stride + kvh * head_stride;
    const float* v_base = kv + layer_v_stride + kvh * head_stride;
    const int T = pos + 1;

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float* kk = k_base + static_cast<size_t>(t) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; ++i) acc += q_h[i] * kk[i];
        scores[t] = acc * scale;
    }
    __syncthreads();

    __shared__ float red[128];
    float mx = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) mx = fmaxf(mx, scores[t]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        scores[t] = expf(scores[t] - mx);
        sum += scores[t];
    }
    __syncthreads();
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    if (threadIdx.x < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t < T; ++t) {
            acc += scores[t] * inv * v_base[static_cast<size_t>(t) * head_dim + threadIdx.x];
        }
        att[qh * head_dim + threadIdx.x] = acc;
    }
}

__global__ void silu_up(float* gate, const float* up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

__global__ void add_inplace(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// decode attention 等编排 kernel 之上是 decode gemv（见下）
__global__ void gemv_f16(const __half* w, const float* x, float* y, int in, int out) {
    const int row = blockIdx.x;
    if (row >= out) return;
    const __half* wr = w + static_cast<size_t>(row) * in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in; i += blockDim.x) {
        acc += __half2float(wr[i]) * x[i];
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = smem[0];
}

// ---------------- impl ----------------

struct LayerDev {
    float* attn_norm;
    __half *wq, *wk, *wv;
    float *q_norm, *k_norm;
    __half* wo;
    float* ffn_norm;
    __half *wg, *wu, *wd;
};

}  // namespace

struct GpuModel::Impl {
    ModelConfig cfg;
    __half* d_embd = nullptr;         // [hidden, vocab]
    float* d_out_norm = nullptr;
    std::vector<LayerDev> layers;
    float* d_kv = nullptr;
    int kv_cap = 0;
    float *d_x, *d_h, *d_q, *d_k, *d_v, *d_att, *d_scores, *d_gate, *d_up, *d_ffn, *d_logits;
    float* h_logits = nullptr;        // pinned

    ~Impl() {
        auto free_if = [](void* p) { if (p) cudaFree(p); };
        free_if(d_embd); free_if(d_out_norm); free_if(d_kv);
        for (LayerDev& l : layers) {
            free_if(l.attn_norm); free_if(l.wq); free_if(l.wk); free_if(l.wv);
            free_if(l.q_norm); free_if(l.k_norm); free_if(l.wo);
            free_if(l.ffn_norm); free_if(l.wg); free_if(l.wu); free_if(l.wd);
        }
        free_if(d_x); free_if(d_h); free_if(d_q); free_if(d_k); free_if(d_v);
        free_if(d_att); free_if(d_scores); free_if(d_gate); free_if(d_up);
        free_if(d_ffn); free_if(d_logits);
        if (h_logits) cudaFreeHost(h_logits);
    }

    // y[out] = W_rm[out][in] · x[in]; custom decode gemv (F16 weights, F32
    // activations and accumulation — numerics match the CPU reference).
    void gemv(const __half* w, const float* x, float* y, int in, int out) {
        gemv_f16<<<out, 256>>>(w, x, y, in, out);
    }
};

GpuModel::GpuModel(GpuModel&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }
GpuModel::~GpuModel() { delete impl_; }
const ModelConfig& GpuModel::config() const { return impl_->cfg; }

GpuModel GpuModel::load(const GGUFFile& f, int kv_cap) {
    GpuModel gm;
    Impl* m = new Impl();
    gm.impl_ = m;
    const std::string arch = kv_str(f, "general.architecture");
    const std::string p = arch + ".";

    ModelConfig& c = m->cfg;
    c.n_layer = kv_u32(f, p + "block_count");
    c.hidden = kv_u32(f, p + "embedding_length");
    c.n_head = kv_u32(f, p + "attention.head_count");
    c.n_kv_head = kv_u32(f, p + "attention.head_count_kv");
    c.head_dim = kv_u32(f, p + "attention.key_length");
    c.ffn = kv_u32(f, p + "feed_forward_length");
    c.rms_eps = kv_f32(f, p + "attention.layer_norm_rms_epsilon");
    c.rope_theta = kv_f32(f, p + "rope.freq_base");
    {
        const TensorInfo* t = f.tensor("token_embd.weight");
        if (!t) fail("missing token_embd.weight");
        c.vocab = static_cast<int>(t->dims[1]);
    }
    if (kv_cap <= 0) fail("kv_cap must be positive");
    m->kv_cap = kv_cap;

    const int64_t H = c.hidden, Q = c.n_head * c.head_dim, KV = c.n_kv_head * c.head_dim, FF = c.ffn;
    m->d_embd = upload_f16(f, "token_embd.weight", H, c.vocab);
    m->d_out_norm = upload_f32(f, "output_norm.weight", H);
    m->layers.reserve(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        LayerDev w;
        w.attn_norm = upload_f32(f, b + "attn_norm.weight", H);
        w.wq = upload_f16(f, b + "attn_q.weight", H, Q);
        w.wk = upload_f16(f, b + "attn_k.weight", H, KV);
        w.wv = upload_f16(f, b + "attn_v.weight", H, KV);
        w.q_norm = upload_f32(f, b + "attn_q_norm.weight", c.head_dim);
        w.k_norm = upload_f32(f, b + "attn_k_norm.weight", c.head_dim);
        w.wo = upload_f16(f, b + "attn_output.weight", Q, H);
        w.ffn_norm = upload_f32(f, b + "ffn_norm.weight", H);
        w.wg = upload_f16(f, b + "ffn_gate.weight", H, FF);
        w.wu = upload_f16(f, b + "ffn_up.weight", H, FF);
        w.wd = upload_f16(f, b + "ffn_down.weight", FF, H);
        m->layers.push_back(w);
    }

    const size_t kv_elems = static_cast<size_t>(c.n_layer) * 2 * c.n_kv_head * kv_cap * c.head_dim;
    CU_CHECK(cudaMalloc(&m->d_kv, kv_elems * sizeof(float)));
    CU_CHECK(cudaMalloc(&m->d_x, H * 4));
    CU_CHECK(cudaMalloc(&m->d_h, H * 4));
    CU_CHECK(cudaMalloc(&m->d_q, Q * 4));
    CU_CHECK(cudaMalloc(&m->d_k, KV * 4));
    CU_CHECK(cudaMalloc(&m->d_v, KV * 4));
    CU_CHECK(cudaMalloc(&m->d_att, Q * 4));
    CU_CHECK(cudaMalloc(&m->d_scores, kv_cap * 4));
    CU_CHECK(cudaMalloc(&m->d_gate, FF * 4));
    CU_CHECK(cudaMalloc(&m->d_up, FF * 4));
    CU_CHECK(cudaMalloc(&m->d_ffn, H * 4));
    CU_CHECK(cudaMalloc(&m->d_logits, c.vocab * 4));
    CU_CHECK(cudaMallocHost(&m->h_logits, c.vocab * 4));
    CU_CHECK(cudaDeviceSynchronize());
    return gm;
}

int32_t GpuModel::forward_argmax(int32_t token, int pos) {
    Impl* m = impl_;
    const ModelConfig& c = m->cfg;
    if (token < 0 || token >= c.vocab) fail("token out of range");
    if (pos < 0 || pos >= m->kv_cap) fail("position exceeds kv_cap");
    const int H = c.hidden, Q = c.n_head * c.head_dim, KV = c.n_kv_head * c.head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(c.head_dim));
    const size_t layer_k_stride = static_cast<size_t>(c.n_kv_head) * m->kv_cap * c.head_dim;
    const size_t layer_v_stride = layer_k_stride;
    const size_t head_stride = static_cast<size_t>(m->kv_cap) * c.head_dim;

    embed_row<<<(H + 255) / 256, 256>>>(m->d_embd, m->d_x, token, H);

    for (int l = 0; l < c.n_layer; ++l) {
        const LayerDev& w = m->layers[l];
        const size_t lk = static_cast<size_t>(l) * 2 * layer_k_stride;
        const size_t lv = lk + layer_v_stride;

        rmsnorm<<<1, 256>>>(m->d_x, w.attn_norm, m->d_h, H, c.rms_eps);
        m->gemv(w.wq, m->d_h, m->d_q, H, Q);
        m->gemv(w.wk, m->d_h, m->d_k, H, KV);
        m->gemv(w.wv, m->d_h, m->d_v, H, KV);
        rmsnorm_heads<<<c.n_head, 128>>>(m->d_q, w.q_norm, c.head_dim, c.rms_eps);
        rmsnorm_heads<<<c.n_kv_head, 128>>>(m->d_k, w.k_norm, c.head_dim, c.rms_eps);
        rope_qk<<<c.n_head + c.n_kv_head, c.head_dim / 2>>>(m->d_q, m->d_k, c.n_head,
                                                            c.head_dim, pos, c.rope_theta);
        kv_store<<<(KV + 255) / 256, 256>>>(m->d_k, m->d_v, m->d_kv, lk, lv,
                                            head_stride, c.n_kv_head, c.head_dim, pos);
        attention<<<c.n_head, 128>>>(m->d_q, m->d_kv, m->d_att, m->d_scores, lk, lv,
                                     head_stride, c.n_head, c.n_kv_head, c.head_dim, pos, scale);
        m->gemv(w.wo, m->d_att, m->d_ffn, Q, H);
        add_inplace<<<(H + 255) / 256, 256>>>(m->d_x, m->d_ffn, H);
        rmsnorm<<<1, 256>>>(m->d_x, w.ffn_norm, m->d_h, H, c.rms_eps);
        m->gemv(w.wg, m->d_h, m->d_gate, H, c.ffn);
        m->gemv(w.wu, m->d_h, m->d_up, H, c.ffn);
        silu_up<<<(c.ffn + 255) / 256, 256>>>(m->d_gate, m->d_up, c.ffn);
        m->gemv(w.wd, m->d_gate, m->d_ffn, c.ffn, H);
        add_inplace<<<(H + 255) / 256, 256>>>(m->d_x, m->d_ffn, H);
    }

    rmsnorm<<<1, 256>>>(m->d_x, m->d_out_norm, m->d_h, H, c.rms_eps);
    m->gemv(m->d_embd, m->d_h, m->d_logits, H, c.vocab);
    CU_CHECK(cudaMemcpy(m->h_logits, m->d_logits, c.vocab * sizeof(float), cudaMemcpyDeviceToHost));

    int32_t best = 0;
    for (int i = 1; i < c.vocab; ++i) if (m->h_logits[i] > m->h_logits[best]) best = i;
    return best;
}

}  // namespace ri
