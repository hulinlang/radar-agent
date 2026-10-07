// B6b CUDA backend. F16 or Q4_K/Q6_K weights in VRAM (quantized tensors
// uploaded as raw bytes, B7); decode gemv via custom kernels (F32
// activations/accumulation; memory-bound shapes — see docs/26 and the cublas
// note in docs/26 results). Small orchestration kernels handwritten.

#include "gpu_model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gguf.h"
#include "gpu_quants.h"
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

// Weight tensors may be F16 (1), Q4_K (12) or Q6_K (14); quantized tensors
// are uploaded as raw bytes and dequantized inside the fused gemv kernels.
struct DevWeight {
    void* p = nullptr;
    uint32_t type = 0;
};

DevWeight upload_weight(const GGUFFile& f, const std::string& name, int64_t in, int64_t out) {
    const TensorInfo* t = f.tensor(name);
    if (!t) fail("missing tensor " + name);
    if (t->ggml_type != 1 && t->ggml_type != 12 && t->ggml_type != 14) {
        fail("tensor " + name + " unsupported type " + std::to_string(t->ggml_type));
    }
    if (t->dims.size() != 2 || static_cast<int64_t>(t->dims[0]) != in ||
        static_cast<int64_t>(t->dims[1]) != out) {
        fail("tensor " + name + " unexpected dims");
    }
    if (t->ggml_type != 1 && in % 256 != 0) fail("tensor " + name + " in not a multiple of 256");
    DevWeight w;
    w.type = t->ggml_type;
    CU_CHECK(cudaMalloc(&w.p, t->n_bytes));
    CU_CHECK(cudaMemcpy(w.p, f.tensor_data(*t), t->n_bytes, cudaMemcpyHostToDevice));
    return w;
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

__global__ void embed_row(const __half* embd, float* x, const int* token_p, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[i] = __half2float(embd[static_cast<size_t>(*token_p) * hidden + i]);
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

// NeoX rope: one block per head (q heads then k heads), head_dim/2 threads.
// pos comes from device memory so the decode step is graph-capturable.
__global__ void rope_qk(float* q, float* k, int n_q_head, int head_dim,
                        const int* pos_p, float theta) {
    const int pos = *pos_p;
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
                         size_t head_stride, int n_kv_head, int head_dim,
                         const int* pos_p) {
    const int pos = *pos_p;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv_head * head_dim;
    if (i >= total) return;
    const int h = i / head_dim, d = i % head_dim;
    kv[layer_k_stride + h * head_stride + static_cast<size_t>(pos) * head_dim + d] = k[i];
    kv[layer_v_stride + h * head_stride + static_cast<size_t>(pos) * head_dim + d] = v[i];
}

// decode attention: one block per q head; pos from device memory (graph-safe)
__global__ void attention(const float* q, const float* kv, float* att, float* scores,
                          size_t layer_k_stride, size_t layer_v_stride, size_t head_stride,
                          int n_head, int n_kv_head, int head_dim, const int* pos_p,
                          float scale) {
    const int pos = *pos_p;
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
// v2: one warp per row, half2 vectorized loads (full 128B cacheline per warp
// transaction), warp-shuffle reduction (no shared memory chain).
__global__ void gemv_f16_v2(const __half* w, const float* x, float* y, int in, int out) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= out) return;
    const __half2* wr = reinterpret_cast<const __half2*>(w + static_cast<size_t>(row) * in);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    float acc = 0.0f;
    const int n2 = in / 2;
    for (int i = lane; i < n2; i += 32) {
        const float2 wv = __half22float2(wr[i]);
        const float2 xv = x2[i];
        acc += wv.x * xv.x + wv.y * xv.y;
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// ---------------- impl ----------------

struct LayerDev {
    float* attn_norm;
    DevWeight wq, wk, wv;
    float *q_norm, *k_norm;
    DevWeight wo;
    float* ffn_norm;
    DevWeight wg, wu, wd;
};

}  // namespace

struct GpuModel::Impl {
    ModelConfig cfg;
    DevWeight d_embd;                   // [hidden, vocab]
    float* d_out_norm = nullptr;
    std::vector<LayerDev> layers;
    float* d_kv = nullptr;
    int kv_cap = 0;
    float *d_x, *d_h, *d_q, *d_k, *d_v, *d_att, *d_scores, *d_gate, *d_up, *d_ffn, *d_logits;
    float* h_logits = nullptr;        // pinned
    // decode-step streaming state: token/pos live in device memory so the
    // whole step (~450 kernel launches) can be CUDA-graph captured (B7).
    int* d_step = nullptr;            // [token, pos]
    int* h_step = nullptr;            // pinned staging
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool use_graph = false;           // off when profiling (events stay eager)

    ~Impl() {
        auto free_if = [](void* p) { if (p) cudaFree(p); };
        free_if(d_embd.p); free_if(d_out_norm); free_if(d_kv);
        for (LayerDev& l : layers) {
            free_if(l.attn_norm); free_if(l.wq.p); free_if(l.wk.p); free_if(l.wv.p);
            free_if(l.q_norm); free_if(l.k_norm); free_if(l.wo.p);
            free_if(l.ffn_norm); free_if(l.wg.p); free_if(l.wu.p); free_if(l.wd.p);
        }
        free_if(d_x); free_if(d_h); free_if(d_q); free_if(d_k); free_if(d_v);
        free_if(d_att); free_if(d_scores); free_if(d_gate); free_if(d_up);
        free_if(d_ffn); free_if(d_logits);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        free_if(d_step);
        if (h_step) cudaFreeHost(h_step);
        prof_print();
        prof_destroy();
        if (h_logits) cudaFreeHost(h_logits);
        if (stream) cudaStreamDestroy(stream);
    }

    // y[out] = W_rm[out][in] · x[in]; decode gemv dispatch. F16: half2
    // vectorized kernel; Q4_K/Q6_K: fused dequant-dot (B7). F32 activations
    // and accumulation either way — numerics match the CPU reference.
    void gemv(const DevWeight& w, const float* x, float* y, int in, int out) {
        if (w.type == 1) {
            gemv_f16_v2<<<(out + 3) / 4, 128, 0, stream>>>(static_cast<const __half*>(w.p), x, y, in, out);
        } else if (w.type == 12) {
            gpuq::gemv_q4k(static_cast<const uint8_t*>(w.p), x, y, in, out, stream);
        } else {
            gpuq::gemv_q6k(static_cast<const uint8_t*>(w.p), x, y, in, out, stream);
        }
    }

    // One decode step: embed + layers + final norm + logits gemv. Every
    // launch goes to `stream` and token/pos come from device memory, so the
    // whole step is capturable into a CUDA graph (one replay per token
    // instead of ~450 launches).
    void step_kernels() {
        const int H = cfg.hidden, Q = cfg.n_head * cfg.head_dim, KV = cfg.n_kv_head * cfg.head_dim;
        const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
        const size_t layer_k_stride = static_cast<size_t>(cfg.n_kv_head) * kv_cap * cfg.head_dim;
        const size_t layer_v_stride = layer_k_stride;
        const size_t head_stride = static_cast<size_t>(kv_cap) * cfg.head_dim;
        const int* d_pos = d_step + 1;

        if (d_embd.type == 1) {
            embed_row<<<(H + 255) / 256, 256, 0, stream>>>(static_cast<const __half*>(d_embd.p),
                                                           d_x, d_step, H);
        } else {
            gpuq::embed_row_q6k(static_cast<const uint8_t*>(d_embd.p), d_x, d_step, H, stream);
        }

        for (int l = 0; l < cfg.n_layer; ++l) {
            const LayerDev& w = layers[l];
            const size_t lk = static_cast<size_t>(l) * 2 * layer_k_stride;
            const size_t lv = lk + layer_v_stride;

            rmsnorm<<<1, 256, 0, stream>>>(d_x, w.attn_norm, d_h, H, cfg.rms_eps);
            prof_begin(PH_GEMV); gemv(w.wq, d_h, d_q, H, Q); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv(w.wk, d_h, d_k, H, KV); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv(w.wv, d_h, d_v, H, KV); prof_end(PH_GEMV);
            rmsnorm_heads<<<cfg.n_head, 128, 0, stream>>>(d_q, w.q_norm, cfg.head_dim, cfg.rms_eps);
            rmsnorm_heads<<<cfg.n_kv_head, 128, 0, stream>>>(d_k, w.k_norm, cfg.head_dim, cfg.rms_eps);
            rope_qk<<<cfg.n_head + cfg.n_kv_head, cfg.head_dim / 2, 0, stream>>>(
                d_q, d_k, cfg.n_head, cfg.head_dim, d_pos, cfg.rope_theta);
            kv_store<<<(KV + 255) / 256, 256, 0, stream>>>(d_k, d_v, d_kv, lk, lv,
                                                           head_stride, cfg.n_kv_head, cfg.head_dim, d_pos);
            prof_begin(PH_ATTN);
            attention<<<cfg.n_head, 128, 0, stream>>>(d_q, d_kv, d_att, d_scores, lk, lv,
                                                      head_stride, cfg.n_head, cfg.n_kv_head,
                                                      cfg.head_dim, d_pos, scale);
            prof_end(PH_ATTN);
            prof_begin(PH_GEMV); gemv(w.wo, d_att, d_ffn, Q, H); prof_end(PH_GEMV);
            add_inplace<<<(H + 255) / 256, 256, 0, stream>>>(d_x, d_ffn, H);
            rmsnorm<<<1, 256, 0, stream>>>(d_x, w.ffn_norm, d_h, H, cfg.rms_eps);
            prof_begin(PH_GEMV); gemv(w.wg, d_h, d_gate, H, cfg.ffn); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv(w.wu, d_h, d_up, H, cfg.ffn); prof_end(PH_GEMV);
            silu_up<<<(cfg.ffn + 255) / 256, 256, 0, stream>>>(d_gate, d_up, cfg.ffn);
            prof_begin(PH_GEMV); gemv(w.wd, d_gate, d_ffn, cfg.ffn, H); prof_end(PH_GEMV);
            add_inplace<<<(H + 255) / 256, 256, 0, stream>>>(d_x, d_ffn, H);
        }

        rmsnorm<<<1, 256, 0, stream>>>(d_x, d_out_norm, d_h, H, cfg.rms_eps);
        prof_begin(PH_LOGITS); gemv(d_embd, d_h, d_logits, H, cfg.vocab); prof_end(PH_LOGITS);
    }

    // ---- profiling (RADAR_GPU_PROFILE=1): per-phase CUDA event timing ----
    bool prof = false;
    static constexpr int PH_GEMV = 0, PH_ATTN = 1, PH_LOGITS = 2, PH_D2H = 3, PH_TOTAL = 4, NPH = 5;
    double acc_ms[NPH] = {0};
    int64_t acc_n = 0;
    cudaEvent_t ev[512];
    int ev_phase[512];
    int nev = 0;
    void prof_init() {
        prof = std::getenv("RADAR_GPU_PROFILE") != nullptr;
        if (prof) for (int i = 0; i < 512; ++i) CU_CHECK(cudaEventCreate(&ev[i]));
    }
    void prof_begin(int phase) {
        if (!prof) return;
        ev_phase[nev] = phase;
        CU_CHECK(cudaEventRecord(ev[nev++], stream));
    }
    void prof_end(int phase) {
        if (!prof) return;
        ev_phase[nev] = ~phase;
        CU_CHECK(cudaEventRecord(ev[nev++], stream));
    }
    void prof_collect() {
        if (!prof) return;
        CU_CHECK(cudaDeviceSynchronize());
        int open[NPH] = {-1, -1, -1, -1, -1};
        for (int i = 0; i < nev; ++i) {
            const int ph = ev_phase[i];
            if (ph >= 0) open[ph] = i;
            else {
                const int p = ~ph;
                float ms = 0;
                CU_CHECK(cudaEventElapsedTime(&ms, ev[open[p]], ev[i]));
                acc_ms[p] += ms;
            }
        }
        nev = 0;
        ++acc_n;
    }
    void prof_print() {
        if (!prof || acc_n == 0) return;
        const char* names[NPH] = {"gemv(weights)", "attention", "logits gemv", "logits D2H+argmax", "total"};
        std::fprintf(stderr, "== GPU profile (%lld tokens, ms/token) ==\n", (long long)acc_n);
        for (int i = 0; i < NPH; ++i) {
            std::fprintf(stderr, "  %-18s %7.3f ms (%5.1f%%)\n", names[i], acc_ms[i] / acc_n,
                         100.0 * acc_ms[i] / (acc_ms[PH_TOTAL] > 0 ? acc_ms[PH_TOTAL] : 1));
        }
    }
    void prof_destroy() { if (prof) for (int i = 0; i < 512; ++i) cudaEventDestroy(ev[i]); }
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
    m->d_embd = upload_weight(f, "token_embd.weight", H, c.vocab);
    m->d_out_norm = upload_f32(f, "output_norm.weight", H);
    m->layers.reserve(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        LayerDev w;
        w.attn_norm = upload_f32(f, b + "attn_norm.weight", H);
        w.wq = upload_weight(f, b + "attn_q.weight", H, Q);
        w.wk = upload_weight(f, b + "attn_k.weight", H, KV);
        w.wv = upload_weight(f, b + "attn_v.weight", H, KV);
        w.q_norm = upload_f32(f, b + "attn_q_norm.weight", c.head_dim);
        w.k_norm = upload_f32(f, b + "attn_k_norm.weight", c.head_dim);
        w.wo = upload_weight(f, b + "attn_output.weight", Q, H);
        w.ffn_norm = upload_f32(f, b + "ffn_norm.weight", H);
        w.wg = upload_weight(f, b + "ffn_gate.weight", H, FF);
        w.wu = upload_weight(f, b + "ffn_up.weight", H, FF);
        w.wd = upload_weight(f, b + "ffn_down.weight", FF, H);
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
    CU_CHECK(cudaMalloc(&m->d_step, 2 * sizeof(int)));
    CU_CHECK(cudaMallocHost(&m->h_step, 2 * sizeof(int)));
    CU_CHECK(cudaStreamCreate(&m->stream));
    CU_CHECK(cudaDeviceSynchronize());
    m->prof_init();
    // CUDA graph replay per decode token (RADAR_NO_CUDA_GRAPH=1 to force eager;
    // profiling implies eager so the per-phase events stay meaningful).
    m->use_graph = !m->prof && std::getenv("RADAR_NO_CUDA_GRAPH") == nullptr;
    return gm;
}

int32_t GpuModel::forward_argmax(int32_t token, int pos) {
    Impl* m = impl_;
    const ModelConfig& c = m->cfg;
    if (token < 0 || token >= c.vocab) fail("token out of range");
    if (pos < 0 || pos >= m->kv_cap) fail("position exceeds kv_cap");

    m->h_step[0] = token;
    m->h_step[1] = pos;
    CU_CHECK(cudaMemcpyAsync(m->d_step, m->h_step, 2 * sizeof(int),
                             cudaMemcpyHostToDevice, m->stream));

    m->prof_begin(Impl::PH_TOTAL);
    if (m->use_graph && pos > 0) {
        if (!m->graph_exec) {
            // pos==0 ran eager (warmup); capture here, then replay forever.
            cudaGraph_t g = nullptr;
            CU_CHECK(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
            m->step_kernels();
            CU_CHECK(cudaStreamEndCapture(m->stream, &g));
            CU_CHECK(cudaGraphInstantiate(&m->graph_exec, g, nullptr, nullptr, 0));
            CU_CHECK(cudaGraphDestroy(g));
        }
        CU_CHECK(cudaGraphLaunch(m->graph_exec, m->stream));
    } else {
        m->step_kernels();
    }
    m->prof_begin(Impl::PH_D2H);
    CU_CHECK(cudaMemcpyAsync(m->h_logits, m->d_logits, c.vocab * sizeof(float),
                             cudaMemcpyDeviceToHost, m->stream));
    CU_CHECK(cudaStreamSynchronize(m->stream));

    int32_t best = 0;
    for (int i = 1; i < c.vocab; ++i) if (m->h_logits[i] > m->h_logits[best]) best = i;
    m->prof_end(Impl::PH_D2H);
    m->prof_end(Impl::PH_TOTAL);
    m->prof_collect();
    return best;
}

}  // namespace ri
