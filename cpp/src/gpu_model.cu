// B6b CUDA backend. F16 or Q4_K/Q6_K weights in VRAM (quantized tensors
// uploaded as raw bytes, B7); decode gemv via custom kernels (F32
// activations/accumulation; memory-bound shapes — see docs/26 and the cublas
// note in docs/26 results). Small orchestration kernels handwritten.

#include "gpu_model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gguf.h"
#include "gpu_quants.h"
#include "gpu_quants.cuh"  // gpuq::detail::q6k_at for embed_rows_q6k
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

// Environment flag with "off" expressible: present and not "0". A bare
// getenv-presence test cannot be turned off from the outside, which is how an A/B
// script ended up comparing MMA against MMA (see the RADAR_Q4_MMA site).
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    return !(v[0] == '0' && v[1] == '\0');
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

// ---------------- B10: paged KV ----------------
// KV lives in a pool of fixed-size blocks (BLOCK tokens each). A sequence's
// blocks are described by a device-side block table
// (block_table[seq * max_blocks + b]) plus a per-sequence valid-block count
// (nblk[seq]). Both live in device memory on purpose: the batch-decode CUDA
// graph bakes pointers and launch shapes, not buffer *contents*, so a sequence
// growing into a new block does not invalidate an already-captured graph.
//
// BLOCK is a compile-time power of two, so pos/BLOCK and pos%BLOCK compile to
// a shift and a mask (no integer division in the inner loops).
constexpr int BLOCK = 16;

__global__ void kv_store(const float* k, const float* v, float* kv,
                         size_t layer_k_stride, size_t layer_v_stride,
                         size_t block_stride, size_t head_stride,
                         int n_kv_head, int head_dim,
                         const int* block_table, const int* pos_p) {
    const int pos = *pos_p;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv_head * head_dim;
    if (i >= total) return;
    const int h = i / head_dim, d = i % head_dim;
    const size_t off = static_cast<size_t>(block_table[pos / BLOCK]) * block_stride +
                       static_cast<size_t>(h) * head_stride +
                       static_cast<size_t>(pos % BLOCK) * head_dim + d;
    kv[layer_k_stride + off] = k[i];
    kv[layer_v_stride + off] = v[i];
}

// decode attention: one block per q head; pos from device memory (graph-safe).
// scores is PER-HEAD scratch (stride scores_stride) — sharing one buffer
// across head blocks is a latent data race (fixed in B8; long prompts
// corrupted the softmax and produced NaN -> argmax 0 loops).
__global__ void attention(const float* q, const float* kv, float* att, float* scores,
                          size_t layer_k_stride, size_t layer_v_stride,
                          size_t block_stride, size_t head_stride,
                          int n_head, int n_kv_head, int head_dim, const int* pos_p,
                          float scale, int scores_stride,
                          const int* block_table, const int* nblk_p) {
    const int pos = *pos_p;
    const int qh = blockIdx.x;
    scores += static_cast<size_t>(qh) * scores_stride;
    const int kvh = qh / (n_head / n_kv_head);
    const float* q_h = q + qh * head_dim;
    const int T = pos + 1;
    const int nb = *nblk_p;   // device-side: keeps the CUDA graph valid

    // score phase: one block indirection per t. Each t is written exactly once
    // with the same inner reduction over head_dim as the contiguous version, so
    // the thread-to-t mapping change does not affect any value.
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float* kb = kv + layer_k_stride +
                          static_cast<size_t>(block_table[t / BLOCK]) * block_stride +
                          kvh * head_stride;
        const float* kk = kb + static_cast<size_t>(t % BLOCK) * head_dim;
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
        // ⚠️ 必须按「块递增 → 块内位置递增」遍历：原实现是 for t=0..T 顺序累加，
        //    保持同一枚举顺序才能保证浮点逐位相同 —— 这就是 DoD#1 硬口径的依据。
        for (int b = 0; b < nb; ++b) {
            const float* vb = kv + layer_v_stride +
                              static_cast<size_t>(block_table[b]) * block_stride +
                              kvh * head_stride;
            const int lo = b * BLOCK;
            const int jmax = (lo + BLOCK < T) ? BLOCK : (T - lo);
            for (int j = 0; j < jmax; ++j)
                acc += scores[lo + j] * inv * vb[static_cast<size_t>(j) * head_dim + threadIdx.x];
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

// ---------------- B8a batch-prefill kernels ----------------
// Row r of a chunk is sequence position pos_base + r. Every kernel's per-row
// math (loop order, reduction tree, block size) is identical to the
// single-token kernel it replaces, so batch prefill is bitwise-equivalent to
// n sequential single-token steps.

__global__ void embed_rows_f16(const __half* embd, float* x, const int* tokens,
                               int hidden) {
    const int tok = tokens[blockIdx.x];
    float* xr = x + static_cast<size_t>(blockIdx.x) * hidden;
    const __half* er = embd + static_cast<size_t>(tok) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) xr[i] = __half2float(er[i]);
}

__global__ void embed_rows_q6k(const uint8_t* embd, float* x, const int* tokens,
                               int hidden) {
    const int tok = tokens[blockIdx.x];
    float* xr = x + static_cast<size_t>(blockIdx.x) * hidden;
    const uint8_t* er = embd + static_cast<size_t>(tok) * (hidden / 256) * 210;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        xr[i] = gpuq::detail::q6k_at(er + (i / 256) * 210, i % 256);
}

// block per row; blockDim 256, same reduction as rmsnorm
__global__ void rmsnorm_many(const float* x, const float* w, float* y, int n, float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
    float* yr = y + static_cast<size_t>(blockIdx.x) * n;
    __shared__ float smem[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    smem[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(smem[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

// block per (row, head); blockDim 128, same reduction as rmsnorm_heads
__global__ void rmsnorm_heads_many(float* v, const float* w, int head_dim, float eps,
                                   int n_head) {
    const int row = blockIdx.x / n_head, h = blockIdx.x % n_head;
    float* x = v + (static_cast<size_t>(row) * n_head + h) * head_dim;
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

// block per (row, head): q heads then k heads; pos = pos_base + row
__global__ void rope_rows(float* q, float* k, int n_q_head, int n_kv_head,
                          int head_dim, int pos_base, float theta) {
    const int nh = n_q_head + n_kv_head;
    const int row = blockIdx.x / nh, h = blockIdx.x % nh;
    const int pos = pos_base + row;
    float* v = (h < n_q_head)
        ? q + (static_cast<size_t>(row) * n_q_head + h) * head_dim
        : k + (static_cast<size_t>(row) * n_kv_head + (h - n_q_head)) * head_dim;
    const int half = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const float freq = 1.0f / powf(theta, 2.0f * i / head_dim);
    const float c = cosf(pos * freq), s = sinf(pos * freq);
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
}

// block per row: copy k/v row into the cache at pos_base + row
__global__ void kv_store_rows(const float* k, const float* v, float* kv,
                              size_t layer_k_stride, size_t layer_v_stride,
                              size_t block_stride, size_t head_stride,
                              int n_kv_head, int head_dim,
                              const int* block_table, int pos_base) {
    const int pos = pos_base + blockIdx.x;
    const int total = n_kv_head * head_dim;
    const float* kr = k + static_cast<size_t>(blockIdx.x) * total;
    const float* vr = v + static_cast<size_t>(blockIdx.x) * total;
    const size_t off = static_cast<size_t>(block_table[pos / BLOCK]) * block_stride +
                       static_cast<size_t>(pos % BLOCK) * head_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int h = i / head_dim, d = i % head_dim;
        kv[layer_k_stride + off + h * head_stride + d] = kr[i];
        kv[layer_v_stride + off + h * head_stride + d] = vr[i];
    }
}

// causal prefill attention: block per (row, q head); query row r attends to
// keys 0..pos_base+r. Scores staged in dynamic smem.
// Phase 1 (scores): warp per t, lanes over head_dim (coalesced K reads) +
//   shuffle reduce — a thread-per-t layout reads K rows strided and stalls.
// Phase 3 (output): warp per 32-dim slice, lanes over d (coalesced V reads),
//   sequential over t — the decode kernel's thread-per-d layout costs a
//   serial T-loop of 512B-strided loads per thread and dominates at long T.
// NOTE: the per-query arithmetic is unchanged, but the reduction ORDER inside
// the score dot differs from the decode kernel (shuffle tree vs serial), so
// prefill is allclose- but not bitwise-equivalent to sequential steps.
__global__ void attention_prefill(const float* q, const float* kv, float* att,
                                  size_t layer_k_stride, size_t layer_v_stride,
                                  size_t block_stride, size_t head_stride,
                                  int n_head, int n_kv_head,
                                  int head_dim, const int* block_table, int pos_base,
                                  float scale) {
    extern __shared__ float sm[];  // [T_max scores][128 reduction]
    float* scores = sm;
    float* red = sm + (pos_base + gridDim.x / n_head);  // upper bound T
    const int r = blockIdx.x / n_head;
    const int qh = blockIdx.x % n_head;
    const int kvh = qh / (n_head / n_kv_head);
    const float* q_h = q + (static_cast<size_t>(r) * n_head + qh) * head_dim;
    const int T = pos_base + r + 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;

    // phase 1: scores (warp per key position)
    float qpart[4];  // head_dim=128: each lane caches its 4 q values
#pragma unroll
    for (int i = 0; i < 4; ++i) qpart[i] = q_h[32 * i + lane];
    for (int t = warp; t < T; t += nwarp) {
        const float* kk = kv + layer_k_stride +
                          static_cast<size_t>(block_table[t / BLOCK]) * block_stride +
                          kvh * head_stride + static_cast<size_t>(t % BLOCK) * head_dim;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc += qpart[i] * kk[32 * i + lane];
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncthreads();

    // phase 2: softmax (block tree reductions, as the decode kernel)
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

    // phase 3: output — warp per 32-dim slice, lanes over d, sequential t
    const int d = warp * 32 + lane;
    if (d < head_dim) {
        float acc = 0.0f;
        for (int b = 0; b * BLOCK < T; ++b) {
            const float* vb = kv + layer_v_stride +
                              static_cast<size_t>(block_table[b]) * block_stride +
                              kvh * head_stride;
            const int lo = b * BLOCK;
            const int jmax = (lo + BLOCK < T) ? BLOCK : (T - lo);
            for (int j = 0; j < jmax; ++j)
                acc += scores[lo + j] * inv * vb[static_cast<size_t>(j) * head_dim + d];
        }
        att[(static_cast<size_t>(r) * n_head + qh) * head_dim + d] = acc;
    }
}

__global__ void silu_up_many(float* gate, const float* up, int64_t n) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

__global__ void add_many(float* x, const float* y, int64_t n) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// ---------------- B8b batch-decode kernels ----------------
// Row = one sequence in the (padded) decode batch; per-row pos/slot come from
// device arrays so the whole batch step is graph-capturable. Per-row math is
// identical to the single-sequence kernels (bitwise per row).

// block per (row, head): q heads then k heads; pos = pos_arr[row]
__global__ void rope_batch(float* q, float* k, int n_q_head, int n_kv_head,
                           int head_dim, const int* pos_arr, float theta) {
    const int nh = n_q_head + n_kv_head;
    const int row = blockIdx.x / nh, h = blockIdx.x % nh;
    const int pos = pos_arr[row];
    float* v = (h < n_q_head)
        ? q + (static_cast<size_t>(row) * n_q_head + h) * head_dim
        : k + (static_cast<size_t>(row) * n_kv_head + (h - n_q_head)) * head_dim;
    const int half = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half) return;
    const float freq = 1.0f / powf(theta, 2.0f * i / head_dim);
    const float c = cosf(pos * freq), s = sinf(pos * freq);
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
}

// block per row: copy k/v row into slot slots[row] at position pos_arr[row]
__global__ void kv_store_batch(const float* k, const float* v, float* kv,
                               size_t layer_k_stride, size_t layer_v_stride,
                               size_t block_stride, size_t head_stride,
                               int n_kv_head, int head_dim,
                               const int* block_table, int max_blocks,
                               const int* pos_arr, const int* slot_arr) {
    const int row = blockIdx.x;
    const int pos = pos_arr[row];
    const int* bt = block_table + static_cast<size_t>(slot_arr[row]) * max_blocks;
    const size_t base = static_cast<size_t>(bt[pos / BLOCK]) * block_stride +
                        static_cast<size_t>(pos % BLOCK) * head_dim;
    const int total = n_kv_head * head_dim;
    const float* kr = k + static_cast<size_t>(row) * total;
    const float* vr = v + static_cast<size_t>(row) * total;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int h = i / head_dim, d = i % head_dim;
        kv[layer_k_stride + base + h * head_stride + d] = kr[i];
        kv[layer_v_stride + base + h * head_stride + d] = vr[i];
    }
}

// block per (row, q head); per-block math identical to the decode attention
// kernel (same loop/reduction order -> bitwise per row)
__global__ void attention_batch(const float* q, const float* kv, float* att, float* scores,
                                size_t layer_k_stride, size_t layer_v_stride,
                                size_t block_stride, size_t head_stride,
                                int n_head, int n_kv_head, int head_dim,
                                const int* block_table, const int* nblk, int max_blocks,
                                const int* pos_arr, const int* slot_arr, float scale,
                                int scores_stride) {
    const int row = blockIdx.x / n_head, qh = blockIdx.x % n_head;
    const int pos = pos_arr[row];
    const int kvh = qh / (n_head / n_kv_head);
    const int seq = slot_arr[row];
    const int* bt = block_table + static_cast<size_t>(seq) * max_blocks;
    const int nb = nblk[seq];   // device-side: keeps the CUDA graph valid
    const float* q_h = q + (static_cast<size_t>(row) * n_head + qh) * head_dim;
    scores += static_cast<size_t>(blockIdx.x) * scores_stride;
    const int T = pos + 1;

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float* kk = kv + layer_k_stride +
                          static_cast<size_t>(bt[t / BLOCK]) * block_stride +
                          kvh * head_stride + static_cast<size_t>(t % BLOCK) * head_dim;
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
        // 同单序列 kernel：块递增 → 块内位置递增，等价于原来的 for t=0..T。
        for (int b = 0; b < nb; ++b) {
            const float* vb = kv + layer_v_stride +
                              static_cast<size_t>(bt[b]) * block_stride +
                              kvh * head_stride;
            const int lo = b * BLOCK;
            const int jmax = (lo + BLOCK < T) ? BLOCK : (T - lo);
            for (int j = 0; j < jmax; ++j)
                acc += scores[lo + j] * inv * vb[static_cast<size_t>(j) * head_dim + threadIdx.x];
        }
        att[(static_cast<size_t>(row) * n_head + qh) * head_dim + threadIdx.x] = acc;
    }
}

// block per row; (value, index) reduction with first-index tie-break, exactly
// matching the host argmax loop (`if (v[i] > v[best]) best = i`)
__global__ void argmax_rows(const float* x, int n, int32_t* out) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
    __shared__ float sv[256];
    __shared__ int32_t si[256];
    float bv = -1e30f;
    int32_t bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int32_t oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
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

// B8 multi-column F16 gemv: warp per R consecutive rows, NC activation
// columns; x loaded once per (column, i) and reused across R rows (see the
// Q4_K mc comment for why R>1 is required to stay DRAM-bound). Each
// column/row bitwise-matches a single-column run.
template <int R, int NC>
__global__ void gemv_f16_mc_kernel(const __half* w, const float* x, float* y,
                                   int in, int out) {
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    if (row0 >= out) return;
    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    const int n2 = in / 2;
    for (int i = lane; i < n2; i += 32) {
        float2 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wv[r] = (row0 + r < out)
                ? __half22float2(*reinterpret_cast<const __half2*>(
                      w + static_cast<size_t>(row0 + r) * in + 2 * i))
                : make_float2(0.0f, 0.0f);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float2 xv = *reinterpret_cast<const float2*>(x + static_cast<size_t>(c) * in + 2 * i);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r][c] += wv[r].x * xv.x + wv[r].y * xv.y;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r >= out) break;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float a = acc[r][c];
            for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) y[static_cast<size_t>(c) * out + row0 + r] = a;
        }
    }
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

namespace gpuq {

void gemv_f16_mc(const __half* w, const float* x, float* y, int in, int out,
                 int ncol, cudaStream_t stream) {
    const int grid = (out + 3) / 4;
    if (ncol <= 1) { gemv_f16_v2<<<grid, 128, 0, stream>>>(w, x, y, in, out); return; }
    auto grid_for = [out](int r) { return ((out + r - 1) / r + 3) / 4; };
    if (ncol <= 2) gemv_f16_mc_kernel<4, 2><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 4) gemv_f16_mc_kernel<4, 4><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else if (ncol <= 8) gemv_f16_mc_kernel<4, 8><<<grid_for(4), 128, 0, stream>>>(w, x, y, in, out);
    else gemv_f16_mc_kernel<2, 16><<<grid_for(2), 128, 0, stream>>>(w, x, y, in, out);
}

}  // namespace gpuq

struct GpuModel::Impl {
    ModelConfig cfg;
    DevWeight d_embd;                   // [hidden, vocab]
    float* d_out_norm = nullptr;
    std::vector<LayerDev> layers;
    float* d_kv = nullptr;
    int kv_cap = 0;
    float *d_x, *d_h, *d_q, *d_k, *d_v, *d_att, *d_scores, *d_gate, *d_up, *d_ffn, *d_logits;
    float* h_logits = nullptr;        // pinned
    // B8a batch-prefill buffers.
    //
    // ⚠️ MAX_MC used to serve TWO roles: the prefill chunk width AND the max_batch
    // cap. They are now split (docs/36 §7.5 point 1): widening the prefill width
    // must not silently raise the batch cap, because d_scores is sized by
    // kv_cap * n_head * max_batch and would grow with it.
    static constexpr int MAX_MC = 16;       // batch cap (B8b padded decode batch)
    // Worst-case prefill width. Buffers are allocated for this; the width actually
    // used is `prefill_w` (runtime, RADAR_PREFILL_W), so a W sweep needs no rebuild.
    // Why widen W at all: B13's ablation showed the Q6_K kernel is neither
    // extraction- nor bandwidth-bound but latency/parallelism-bound -- only ~5.3
    // warps per SM are available at W=16 (out/16 warps total), and widening W
    // lengthens the MMA batch per A-fragment load (docs/37 §7.4).
    static constexpr int MAX_PREFILL = 64;
    float *d_xm = nullptr, *d_hm = nullptr, *d_qm = nullptr, *d_km = nullptr,
          *d_vm = nullptr, *d_attm = nullptr, *d_gatem = nullptr, *d_upm = nullptr,
          *d_ffnm = nullptr;
    // ---- B11: int8 activations (prefill only, quantized weights only) ----
    // RADAR_Q4_FP32_ACT=1 forces the pre-B11 fp32-activation route, which is
    // what makes D3/D4 paired same-session measurements possible (docs/34 §3.5).
    bool int8_act = true;
    int8_t* d_q8s = nullptr;        // [MAX_PREFILL][q8_k] int8 activation quants
    Q8_1Ds* d_q8ds = nullptr;       // [MAX_PREFILL][q8_k/32] scales + exact sums
    int q8_k = 0;                   // allocated k (max of the gemv input dims)
    int prefill_w = MAX_MC;         // actual prefill chunk width (<= MAX_PREFILL)
    int tile_rw = gpuq::B11_TILE_RW;   // RADAR_B11_TILE=RxC, measurement knob only
    int tile_nc = gpuq::B11_TILE_NC;
    // B12: RADAR_Q4_MMA=1 switches the int8 prefill GEMM from __dp4a to the
    // m16n8k32 tensor-core instruction. Off by default so the B11 record stays
    // reproducible; it is bit-exact with the dp4a route either way.
    bool mma = false;
    // B13: RADAR_Q6K_INT8=1 also puts Q6_K on the int8 route (m16n8k16). Off by
    // default: B11/B12 kept Q6_K on the fp32 route, and this changes end-to-end
    // ids (it is a numerical change, not just an instruction swap).
    bool q6k_int8 = false;
    int* d_tokens = nullptr;
    // B8b slot pool + batch decode state
    int max_batch = 1;
    int n_slots = 1;                // max_batch>1 -> max_batch+1 (last = pad scratch)
    size_t slot_stride = 0;         // legacy pre-B10 layout; kernels no longer use it
    // ---- B10: paged KV ----
    int max_blocks = 0;             // blocks per sequence = ceil(kv_cap / BLOCK)
    int num_blocks = 0;             // pool size, in blocks
    size_t block_stride = 0;        // n_kv_head * BLOCK * head_dim (elements per block)
    int* d_block_table = nullptr;   // [n_slots][max_blocks], device
    int* d_nblk = nullptr;          // [n_slots] valid block count, device
    std::vector<int> h_block_table; // host mirror of d_block_table
    std::vector<int> h_nblk;        // host mirror of d_nblk
    std::vector<int> free_blocks;   // free stack (pop from back)
    int used_blocks = 0;
    // ---- B10b: cross-request prefix cache ----
    bool prefix_cache = true;                       // RADAR_NO_PREFIX_CACHE=1 disables
    std::vector<std::vector<int32_t>> hist;         // per-slot token history
    std::vector<char> hist_valid;                   // history seen from position 0?
    std::vector<char> seq_active;                   // slot holds a live sequence?
    std::vector<int> blk_refcount;                  // per-block reference count
    std::vector<char> blk_cached;                   // block registered in prefix_index?
    std::vector<char> blk_free;                     // block currently on the free stack?
    std::vector<uint64_t> blk_hash;                 // cached content hash
    std::vector<std::array<int32_t, BLOCK>> blk_toks;  // cached tokens (collision check)
    std::vector<std::vector<uint64_t>> seq_blk_hash;   // [slot][b] chain hash
    std::unordered_map<uint64_t, int> prefix_index;    // chain hash -> block id
    int prefix_hits = 0, prefix_misses = 0;
    int last_reused = 0;
    int *d_btok = nullptr, *d_bpos = nullptr, *d_bslot = nullptr;
    int32_t* d_next = nullptr;
    float* d_logitsm = nullptr;     // [max_batch][vocab]
    int* h_bstage = nullptr;        // pinned [3*max_batch]: tok|pos|slot
    int32_t* h_next = nullptr;      // pinned [max_batch]
    cudaGraphExec_t graph_exec_b = nullptr;
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
        free_if(d_block_table); free_if(d_nblk);
        for (LayerDev& l : layers) {
            free_if(l.attn_norm); free_if(l.wq.p); free_if(l.wk.p); free_if(l.wv.p);
            free_if(l.q_norm); free_if(l.k_norm); free_if(l.wo.p);
            free_if(l.ffn_norm); free_if(l.wg.p); free_if(l.wu.p); free_if(l.wd.p);
        }
        free_if(d_x); free_if(d_h); free_if(d_q); free_if(d_k); free_if(d_v);
        free_if(d_att); free_if(d_scores); free_if(d_gate); free_if(d_up);
        free_if(d_ffn); free_if(d_logits);
        free_if(d_xm); free_if(d_hm); free_if(d_qm); free_if(d_km); free_if(d_vm);
        free_if(d_attm); free_if(d_gatem); free_if(d_upm); free_if(d_ffnm);
        free_if(d_tokens);
        free_if(d_q8s); free_if(d_q8ds);
        free_if(d_btok); free_if(d_bpos); free_if(d_bslot); free_if(d_next);
        free_if(d_logitsm);
        if (graph_exec_b) cudaGraphExecDestroy(graph_exec_b);
        if (h_bstage) cudaFreeHost(h_bstage);
        if (h_next) cudaFreeHost(h_next);
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

    // B8a multi-column dispatch (prefill chunks, ncol <= MAX_MC).
    void gemv_mc(const DevWeight& w, const float* x, float* y, int in, int out, int ncol) {
        if (w.type == 1) {
            gpuq::gemv_f16_mc(static_cast<const __half*>(w.p), x, y, in, out, ncol, stream);
        } else if (w.type == 12) {
            gpuq::gemv_q4k_mc(static_cast<const uint8_t*>(w.p), x, y, in, out, ncol, stream);
        } else {
            gpuq::gemv_q6k_mc(static_cast<const uint8_t*>(w.p), x, y, in, out, ncol, stream);
        }
    }

    // B11: prefill-chunk dispatch. int8 activations are used only here (the
    // batch-decode path above keeps calling gemv_mc) and only for quantized
    // weights -- F16 stays on the fp32 route, so the F16 440/440 record and the
    // whole decode record are untouched (docs/34 §3.1, §6).
    // B13 adds Q6_K to this route (RADAR_Q6K_INT8=1, default off), because B12
    // measured that the 30% of parameters still on the fp32 route were about half
    // the gemv time (docs/36 §7.8).
    void gemv_prefill(const DevWeight& w, const float* x, float* y, int in, int out, int ncol) {
        const bool int8_ok = int8_act && in <= q8_k &&
                             (w.type == 12 || (w.type == 14 && q6k_int8));
        if (int8_ok) {
            gpuq::quantize_q8_1(x, d_q8s, d_q8ds, in, MAX_PREFILL, stream);
            if (w.type == 12) {
                // B12: tensor-core route. Both branches consume the same Q8_1
                // buffers and are bit-exact with each other, so the switch only
                // moves the inner loop (docs/36 §7.7). `stride = MAX_PREFILL`
                // because that is the allocated column count of the activation
                // layout; `ncol` is the number of VALID columns. The kernels check
                // that the stride covers the last column group and no-op otherwise.
                if (mma && out % 16 == 0) {
                    gpuq::gemv_q4k_mma(static_cast<const uint8_t*>(w.p), d_q8s, d_q8ds, y, in,
                                       out, ncol, MAX_PREFILL, stream);
                } else {
                    gpuq::gemv_q4k_q8_1_tile(static_cast<const uint8_t*>(w.p), d_q8s, d_q8ds, y,
                                             in, out, ncol, tile_rw, tile_nc, stream);
                }
            } else {
                // B13: Q6_K has only the m16n8k16 variant -- K=16 is exactly its
                // scale granularity (docs/37 §7.1). The dp4a intermediate step
                // was skipped as low-value (docs/37 §7.3).
                gpuq::gemv_q6k_mma(static_cast<const uint8_t*>(w.p), d_q8s, d_q8ds, y, in,
                                   out, ncol, MAX_PREFILL, stream);
            }
        } else {
            gemv_mc(w, x, y, in, out, ncol);
        }
    }

    // ---- B10: block pool bookkeeping (host side) ----
    // FNV-1a over the raw bytes of `n` int32 (stable across machines).
    static uint64_t hash_mix(uint64_t h, const int32_t* p, int n) {
        for (int i = 0; i < n; ++i) {
            const uint32_t v = static_cast<uint32_t>(p[i]);
            for (int k = 0; k < 4; ++k) {
                h ^= static_cast<uint64_t>((v >> (8 * k)) & 0xffu);
                h *= 1099511628211ULL;
            }
        }
        return h;
    }

    bool block_matches(int blk, const int32_t* toks) const {
        for (int j = 0; j < BLOCK; ++j)
            if (blk_toks[blk][j] != toks[j]) return false;
        return true;
    }

    // Takes a block off the free stack. A block that was released still sits in
    // the prefix index (that is the cache); reusing it for *different* content
    // must therefore drop its stale entry first.
    int alloc_block() {
        while (!free_blocks.empty()) {
            const int bid = free_blocks.back();
            free_blocks.pop_back();
            blk_free[bid] = 0;
            if (blk_cached[bid]) {
                prefix_index.erase(blk_hash[bid]);
                blk_cached[bid] = 0;
            }
            blk_refcount[bid] = 1;
            return bid;
        }
        return -1;
    }

    // A cache hit on a block whose refcount was 0 must take it OFF the free
    // stack, or a later alloc could hand out the same physical block twice.
    void claim_free_block(int bid) {
        if (!blk_free[bid]) return;
        for (size_t i = 0; i < free_blocks.size(); ++i) {
            if (free_blocks[i] == bid) {
                free_blocks[i] = free_blocks.back();
                free_blocks.pop_back();
                break;
            }
        }
        blk_free[bid] = 0;
    }

    // Ensures sequence `seq` owns blocks covering positions [0, upto).
    // Returns the number of LEADING blocks that came from the cross-request
    // prefix cache — i.e. exactly the tokens a caller may skip computing.
    int ensure_blocks(int seq, int upto) {
        if (upto <= 0) return 0;
        const int need = (upto + BLOCK - 1) / BLOCK;
        if (need > max_blocks) fail("sequence exceeds kv_cap");
        int& have = h_nblk[seq];
        if (need <= have) return 0;
        const std::vector<int32_t>& h = hist[seq];
        const bool can_hash = prefix_cache && hist_valid[seq] != 0;
        int leading = 0;
        uint64_t prev = (have > 0) ? seq_blk_hash[seq][have - 1] : 0;
        int* row = &h_block_table[static_cast<size_t>(seq) * max_blocks];
        while (have < need) {
            const int b = have;
            const int32_t* toks =
                (h.size() >= static_cast<size_t>((b + 1) * BLOCK))
                    ? h.data() + static_cast<size_t>(b) * BLOCK
                    : nullptr;
            int bid = -1;
            if (can_hash && toks) {
                const uint64_t hsh = hash_mix(prev, toks, BLOCK);
                const auto it = prefix_index.find(hsh);
                if (it != prefix_index.end() && block_matches(it->second, toks)) {
                    bid = it->second;          // shared: same tokens, same positions
                    claim_free_block(bid);
                    ++blk_refcount[bid];
                    ++prefix_hits;
                    if (b == leading) ++leading;
                } else {
                    ++prefix_misses;
                }
                if (bid < 0) {
                    bid = alloc_block();
                    if (bid < 0) fail("KV block pool exhausted (raise num_blocks)");
                    std::memcpy(blk_toks[bid].data(), toks, BLOCK * sizeof(int32_t));
                    blk_hash[bid] = hsh;
                    blk_cached[bid] = 1;
                    prefix_index[hsh] = bid;
                }
                seq_blk_hash[seq][b] = hsh;
                prev = hsh;
            } else {
                // Partial tail block (or caching off): never shared, never indexed.
                bid = alloc_block();
                if (bid < 0) fail("KV block pool exhausted (raise num_blocks)");
                seq_blk_hash[seq][b] = 0;
                prev = 0;
            }
            row[b] = bid;
            ++have;
            ++used_blocks;
        }
        // ⚠️ 这两个上传必须在 CUDA graph 捕获/重放**之前**发生。kernel 从显存读
        //    块表与块数，图只固定指针与 launch 形状，所以内容更新不会让图失效。
        CU_CHECK(cudaMemcpyAsync(d_nblk + seq, &h_nblk[seq], sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
        CU_CHECK(cudaMemcpyAsync(d_block_table + static_cast<size_t>(seq) * max_blocks,
                                 row, static_cast<size_t>(have) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
        return leading;
    }

    // Returns a slot's blocks to the pool. A block whose refcount reaches 0 goes
    // back on the free stack but **stays in the prefix index** — that is exactly
    // what lets a later request with the same prefix hit the cache.
    void release_slot_blocks(int seq) {
        const int have = h_nblk[seq];
        int* row = &h_block_table[static_cast<size_t>(seq) * max_blocks];
        for (int b = 0; b < have; ++b) {
            const int bid = row[b];
            if (bid >= 0 && --blk_refcount[bid] == 0) {
                free_blocks.push_back(bid);
                blk_free[bid] = 1;
            }
        }
        used_blocks -= have;
        h_nblk[seq] = 0;
        CU_CHECK(cudaMemcpyAsync(d_nblk + seq, &h_nblk[seq], sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
    }

    // B10b: start a new sequence in `slot`.
    void begin_sequence(int slot) {
        release_slot_blocks(slot);
        hist[slot].clear();
        hist_valid[slot] = 1;
        seq_active[slot] = 1;
        if (!seq_blk_hash[slot].empty())
            std::fill(seq_blk_hash[slot].begin(), seq_blk_hash[slot].end(), 0);
    }

    // One prefill chunk: rows 0..nr-1 of the multi-column buffers are sequence
    // positions pos_base..pos_base+nr-1. Bitwise-equivalent to nr sequential
    // single-token steps (see the B8a kernel block above).
    void prefill_chunk(int pos_base, int nr, int slot) {
        const int H = cfg.hidden, Q = cfg.n_head * cfg.head_dim, KV = cfg.n_kv_head * cfg.head_dim;
        const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
        const size_t layer_k_stride = static_cast<size_t>(num_blocks) * block_stride;
        const size_t layer_v_stride = layer_k_stride;
        const size_t head_stride = static_cast<size_t>(BLOCK) * cfg.head_dim;
        const int* bt = d_block_table + static_cast<size_t>(slot) * max_blocks;

        if (d_embd.type == 1) {
            embed_rows_f16<<<nr, 256, 0, stream>>>(static_cast<const __half*>(d_embd.p),
                                                   d_xm, d_tokens, H);
        } else {
            embed_rows_q6k<<<nr, 256, 0, stream>>>(static_cast<const uint8_t*>(d_embd.p),
                                                   d_xm, d_tokens, H);
        }

        for (int l = 0; l < cfg.n_layer; ++l) {
            const LayerDev& w = layers[l];
            const size_t lk = static_cast<size_t>(l) * 2 * layer_k_stride;
            const size_t lv = lk + layer_v_stride;

            rmsnorm_many<<<nr, 256, 0, stream>>>(d_xm, w.attn_norm, d_hm, H, cfg.rms_eps);
            prof_begin(PH_GEMV); gemv_prefill(w.wq, d_hm, d_qm, H, Q, nr); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv_prefill(w.wk, d_hm, d_km, H, KV, nr); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv_prefill(w.wv, d_hm, d_vm, H, KV, nr); prof_end(PH_GEMV);
            rmsnorm_heads_many<<<nr * cfg.n_head, 128, 0, stream>>>(d_qm, w.q_norm, cfg.head_dim,
                                                                    cfg.rms_eps, cfg.n_head);
            rmsnorm_heads_many<<<nr * cfg.n_kv_head, 128, 0, stream>>>(d_km, w.k_norm, cfg.head_dim,
                                                                       cfg.rms_eps, cfg.n_kv_head);
            rope_rows<<<nr * (cfg.n_head + cfg.n_kv_head), cfg.head_dim / 2, 0, stream>>>(
                d_qm, d_km, cfg.n_head, cfg.n_kv_head, cfg.head_dim, pos_base, cfg.rope_theta);
            kv_store_rows<<<nr, 256, 0, stream>>>(d_km, d_vm, d_kv, lk, lv,
                                                  block_stride, head_stride, cfg.n_kv_head,
                                                  cfg.head_dim, bt, pos_base);
            const int t_max = pos_base + nr;
            prof_begin(PH_ATTN);
            attention_prefill<<<nr * cfg.n_head, 128, (t_max + 128) * sizeof(float), stream>>>(
                d_qm, d_kv, d_attm, lk, lv, block_stride, head_stride, cfg.n_head, cfg.n_kv_head,
                cfg.head_dim, bt, pos_base, scale);
            prof_end(PH_ATTN);
            prof_begin(PH_GEMV); gemv_prefill(w.wo, d_attm, d_ffnm, Q, H, nr); prof_end(PH_GEMV);
            add_many<<<(int)((static_cast<int64_t>(nr) * H + 255) / 256), 256, 0, stream>>>(
                d_xm, d_ffnm, static_cast<int64_t>(nr) * H);
            rmsnorm_many<<<nr, 256, 0, stream>>>(d_xm, w.ffn_norm, d_hm, H, cfg.rms_eps);
            prof_begin(PH_GEMV); gemv_prefill(w.wg, d_hm, d_gatem, H, cfg.ffn, nr); prof_end(PH_GEMV);
            prof_begin(PH_GEMV); gemv_prefill(w.wu, d_hm, d_upm, H, cfg.ffn, nr); prof_end(PH_GEMV);
            silu_up_many<<<(int)((static_cast<int64_t>(nr) * cfg.ffn + 255) / 256), 256, 0, stream>>>(
                d_gatem, d_upm, static_cast<int64_t>(nr) * cfg.ffn);
            prof_begin(PH_GEMV); gemv_prefill(w.wd, d_gatem, d_ffnm, cfg.ffn, H, nr); prof_end(PH_GEMV);
            add_many<<<(int)((static_cast<int64_t>(nr) * H + 255) / 256), 256, 0, stream>>>(
                d_xm, d_ffnm, static_cast<int64_t>(nr) * H);
        }

        rmsnorm_many<<<nr, 256, 0, stream>>>(d_xm, d_out_norm, d_hm, H, cfg.rms_eps);
    }

    // B8b: one decode step for the full padded batch (max_batch rows). Row i
    // reads token/pos/slot from device arrays; padding rows point at the
    // scratch slot (max_batch). All launches on `stream`; graph-capturable.
    void step_batch_kernels() {
        const int B = max_batch;
        const int H = cfg.hidden, Q = cfg.n_head * cfg.head_dim, KV = cfg.n_kv_head * cfg.head_dim;
        const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
        const size_t layer_k_stride = static_cast<size_t>(num_blocks) * block_stride;
        const size_t layer_v_stride = layer_k_stride;
        const size_t head_stride = static_cast<size_t>(BLOCK) * cfg.head_dim;

        if (d_embd.type == 1) {
            embed_rows_f16<<<B, 256, 0, stream>>>(static_cast<const __half*>(d_embd.p),
                                                  d_xm, d_btok, H);
        } else {
            embed_rows_q6k<<<B, 256, 0, stream>>>(static_cast<const uint8_t*>(d_embd.p),
                                                  d_xm, d_btok, H);
        }

        for (int l = 0; l < cfg.n_layer; ++l) {
            const LayerDev& w = layers[l];
            const size_t lk = static_cast<size_t>(l) * 2 * layer_k_stride;
            const size_t lv = lk + layer_v_stride;

            rmsnorm_many<<<B, 256, 0, stream>>>(d_xm, w.attn_norm, d_hm, H, cfg.rms_eps);
            gemv_mc(w.wq, d_hm, d_qm, H, Q, B);
            gemv_mc(w.wk, d_hm, d_km, H, KV, B);
            gemv_mc(w.wv, d_hm, d_vm, H, KV, B);
            rmsnorm_heads_many<<<B * cfg.n_head, 128, 0, stream>>>(d_qm, w.q_norm, cfg.head_dim,
                                                                   cfg.rms_eps, cfg.n_head);
            rmsnorm_heads_many<<<B * cfg.n_kv_head, 128, 0, stream>>>(d_km, w.k_norm, cfg.head_dim,
                                                                      cfg.rms_eps, cfg.n_kv_head);
            rope_batch<<<B * (cfg.n_head + cfg.n_kv_head), cfg.head_dim / 2, 0, stream>>>(
                d_qm, d_km, cfg.n_head, cfg.n_kv_head, cfg.head_dim, d_bpos, cfg.rope_theta);
            kv_store_batch<<<B, 256, 0, stream>>>(d_km, d_vm, d_kv, lk, lv, block_stride,
                                                  head_stride, cfg.n_kv_head, cfg.head_dim,
                                                  d_block_table, max_blocks, d_bpos, d_bslot);
            attention_batch<<<B * cfg.n_head, 128, 0, stream>>>(
                d_qm, d_kv, d_attm, d_scores, lk, lv, block_stride, head_stride,
                cfg.n_head, cfg.n_kv_head, cfg.head_dim, d_block_table, d_nblk, max_blocks,
                d_bpos, d_bslot, scale, kv_cap);
            gemv_mc(w.wo, d_attm, d_ffnm, Q, H, B);
            add_many<<<(int)((static_cast<int64_t>(B) * H + 255) / 256), 256, 0, stream>>>(
                d_xm, d_ffnm, static_cast<int64_t>(B) * H);
            rmsnorm_many<<<B, 256, 0, stream>>>(d_xm, w.ffn_norm, d_hm, H, cfg.rms_eps);
            gemv_mc(w.wg, d_hm, d_gatem, H, cfg.ffn, B);
            gemv_mc(w.wu, d_hm, d_upm, H, cfg.ffn, B);
            silu_up_many<<<(int)((static_cast<int64_t>(B) * cfg.ffn + 255) / 256), 256, 0, stream>>>(
                d_gatem, d_upm, static_cast<int64_t>(B) * cfg.ffn);
            gemv_mc(w.wd, d_gatem, d_ffnm, cfg.ffn, H, B);
            add_many<<<(int)((static_cast<int64_t>(B) * H + 255) / 256), 256, 0, stream>>>(
                d_xm, d_ffnm, static_cast<int64_t>(B) * H);
        }

        rmsnorm_many<<<B, 256, 0, stream>>>(d_xm, d_out_norm, d_hm, H, cfg.rms_eps);
        gemv_mc(d_embd, d_hm, d_logitsm, H, cfg.vocab, B);
        argmax_rows<<<B, 256, 0, stream>>>(d_logitsm, cfg.vocab, d_next);
    }

    // One decode step: embed + layers + final norm + logits gemv. Every
    // launch goes to `stream` and token/pos come from device memory, so the
    // whole step is capturable into a CUDA graph (one replay per token
    // instead of ~450 launches).
    void step_kernels() {
        const int H = cfg.hidden, Q = cfg.n_head * cfg.head_dim, KV = cfg.n_kv_head * cfg.head_dim;
        const float scale = 1.0f / sqrtf(static_cast<float>(cfg.head_dim));
        const size_t layer_k_stride = static_cast<size_t>(num_blocks) * block_stride;
        const size_t layer_v_stride = layer_k_stride;
        const size_t head_stride = static_cast<size_t>(BLOCK) * cfg.head_dim;
        const int* d_pos = d_step + 1;
        const int* bt0 = d_block_table;   // single-sequence path always uses seq 0

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
                                                           block_stride, head_stride, cfg.n_kv_head,
                                                           cfg.head_dim, bt0, d_pos);
            prof_begin(PH_ATTN);
            attention<<<cfg.n_head, 128, 0, stream>>>(d_q, d_kv, d_att, d_scores, lk, lv,
                                                      block_stride, head_stride, cfg.n_head,
                                                      cfg.n_kv_head, cfg.head_dim, d_pos, scale,
                                                      kv_cap, bt0, d_nblk);
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

GpuModel GpuModel::load(const GGUFFile& f, int kv_cap, int max_batch, int num_blocks) {
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

    // ---- B10: paged KV pool ----
    // A block holds BLOCK tokens of K and V for every KV head and layer. The
    // pool replaces the old "one contiguous kv_cap region per slot" layout.
    m->max_blocks = (kv_cap + BLOCK - 1) / BLOCK;
    m->block_stride = static_cast<size_t>(c.n_kv_head) * BLOCK * c.head_dim;
    const size_t block_bytes =
        static_cast<size_t>(c.n_layer) * 2 * m->block_stride * sizeof(float);
    // Worst case = every slot fills kv_cap. `num_blocks == 0` allocates exactly
    // that, i.e. the pre-B10 footprint (so correctness is a like-for-like
    // comparison); a smaller explicit value is where the VRAM saving comes from.
    auto worst_case_blocks = [&](int mb) {
        const int slots = mb > 1 ? mb + 1 : 1;
        return static_cast<size_t>(slots) * static_cast<size_t>(m->max_blocks);
    };
    if (max_batch < 1) fail("max_batch must be >= 1");
    if (max_batch > Impl::MAX_MC) max_batch = Impl::MAX_MC;
    size_t free_bytes = 0, total_bytes = 0;
    CU_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    while (max_batch > 1 && worst_case_blocks(max_batch) * block_bytes > free_bytes * 9 / 10)
        max_batch >>= 1;
    m->max_batch = max_batch;
    m->n_slots = max_batch > 1 ? max_batch + 1 : 1;
    m->num_blocks = num_blocks > 0 ? num_blocks
                                   : static_cast<int>(worst_case_blocks(max_batch));
    m->slot_stride = static_cast<size_t>(c.n_kv_head) * kv_cap * c.head_dim;  // legacy
    // RADAR_MEM_REPORT=1: print the computed pool size next to the allocator's
    // own delta — two independent sources for the same number (§5.8).
    size_t kv_free0 = 0, kv_total0 = 0;
    if (std::getenv("RADAR_MEM_REPORT")) CU_CHECK(cudaMemGetInfo(&kv_free0, &kv_total0));
    CU_CHECK(cudaMalloc(&m->d_kv, static_cast<size_t>(m->num_blocks) * block_bytes));
    if (std::getenv("RADAR_MEM_REPORT")) {
        size_t kv_free1 = 0, kv_total1 = 0;
        CU_CHECK(cudaMemGetInfo(&kv_free1, &kv_total1));
        std::fprintf(stderr,
                     "kv mem [B10]: computed=%.1f MiB | measured cudaMemGetInfo delta=%.1f MiB\n",
                     static_cast<double>(m->num_blocks) * static_cast<double>(block_bytes) / 1048576.0,
                     static_cast<double>(kv_free0 - kv_free1) / 1048576.0);
    }
    CU_CHECK(cudaMalloc(&m->d_block_table,
                        static_cast<size_t>(m->n_slots) * m->max_blocks * sizeof(int)));
    CU_CHECK(cudaMalloc(&m->d_nblk, static_cast<size_t>(m->n_slots) * sizeof(int)));
    m->h_block_table.assign(static_cast<size_t>(m->n_slots) * m->max_blocks, 0);
    m->h_nblk.assign(m->n_slots, 0);
    CU_CHECK(cudaMemset(m->d_block_table, 0,
                        static_cast<size_t>(m->n_slots) * m->max_blocks * sizeof(int)));
    CU_CHECK(cudaMemset(m->d_nblk, 0, static_cast<size_t>(m->n_slots) * sizeof(int)));
    CU_CHECK(cudaMalloc(&m->d_x, H * 4));
    CU_CHECK(cudaMalloc(&m->d_h, H * 4));
    CU_CHECK(cudaMalloc(&m->d_q, Q * 4));
    CU_CHECK(cudaMalloc(&m->d_k, KV * 4));
    CU_CHECK(cudaMalloc(&m->d_v, KV * 4));
    CU_CHECK(cudaMalloc(&m->d_att, Q * 4));
    CU_CHECK(cudaMalloc(&m->d_scores, static_cast<size_t>(kv_cap) * c.n_head * max_batch * 4));
    CU_CHECK(cudaMalloc(&m->d_gate, FF * 4));
    CU_CHECK(cudaMalloc(&m->d_up, FF * 4));
    CU_CHECK(cudaMalloc(&m->d_ffn, H * 4));
    CU_CHECK(cudaMalloc(&m->d_logits, c.vocab * 4));
    CU_CHECK(cudaMallocHost(&m->h_logits, c.vocab * 4));
    CU_CHECK(cudaMalloc(&m->d_step, 2 * sizeof(int)));
    CU_CHECK(cudaMallocHost(&m->h_step, 2 * sizeof(int)));
    CU_CHECK(cudaStreamCreate(&m->stream));
    m->free_blocks.resize(m->num_blocks);
    for (int i = 0; i < m->num_blocks; ++i) m->free_blocks[i] = m->num_blocks - 1 - i;
    // B10b prefix-cache state
    m->prefix_cache = std::getenv("RADAR_NO_PREFIX_CACHE") == nullptr;
    m->hist.resize(m->n_slots);
    m->hist_valid.assign(m->n_slots, 0);
    m->seq_active.assign(m->n_slots, 0);
    m->blk_refcount.assign(m->num_blocks, 0);
    m->blk_cached.assign(m->num_blocks, 0);
    m->blk_free.assign(m->num_blocks, 1);   // all start on the free stack
    m->blk_hash.assign(m->num_blocks, 0);
    m->blk_toks.resize(m->num_blocks);
    m->seq_blk_hash.assign(m->n_slots, std::vector<uint64_t>(m->max_blocks, 0));
    if (max_batch > 1) {
        // B8b batch-decode state
        CU_CHECK(cudaMalloc(&m->d_btok, max_batch * sizeof(int)));
        CU_CHECK(cudaMalloc(&m->d_bpos, max_batch * sizeof(int)));
        CU_CHECK(cudaMalloc(&m->d_bslot, max_batch * sizeof(int)));
        CU_CHECK(cudaMalloc(&m->d_next, max_batch * sizeof(int32_t)));
        CU_CHECK(cudaMalloc(&m->d_logitsm, static_cast<size_t>(max_batch) * c.vocab * 4));
        CU_CHECK(cudaMallocHost(&m->h_bstage, 3 * max_batch * sizeof(int)));
        CU_CHECK(cudaMallocHost(&m->h_next, max_batch * sizeof(int32_t)));
    }
    // B8a batch-prefill buffers, sized for the WORST-CASE prefill width.
    const int MC = Impl::MAX_PREFILL;
    CU_CHECK(cudaMalloc(&m->d_xm, static_cast<size_t>(MC) * H * 4));
    CU_CHECK(cudaMalloc(&m->d_hm, static_cast<size_t>(MC) * H * 4));
    CU_CHECK(cudaMalloc(&m->d_qm, static_cast<size_t>(MC) * Q * 4));
    CU_CHECK(cudaMalloc(&m->d_km, static_cast<size_t>(MC) * KV * 4));
    CU_CHECK(cudaMalloc(&m->d_vm, static_cast<size_t>(MC) * KV * 4));
    CU_CHECK(cudaMalloc(&m->d_attm, static_cast<size_t>(MC) * Q * 4));
    CU_CHECK(cudaMalloc(&m->d_gatem, static_cast<size_t>(MC) * FF * 4));
    CU_CHECK(cudaMalloc(&m->d_upm, static_cast<size_t>(MC) * FF * 4));
    CU_CHECK(cudaMalloc(&m->d_ffnm, static_cast<size_t>(MC) * H * 4));
    CU_CHECK(cudaMalloc(&m->d_tokens, MC * sizeof(int)));
    // B11 int8 activation scratch, sized to the largest gemv input dimension.
    // RADAR_Q4_FP32_ACT=1 restores the pre-B11 fp32-activation route for the
    // quantized weights (same-session A/B, docs/34 §3.5).
    m->int8_act = std::getenv("RADAR_Q4_FP32_ACT") == nullptr;
    // B13 step 8: RADAR_PREFILL_W selects the prefill chunk width (1..MAX_PREFILL).
    // Runtime, not compile time, so a W sweep needs no rebuild. Per-column
    // arithmetic is independent of W (B8a proved multi-column bitwise-equals
    // single-column), so changing W must NOT change any id -- that is the safety net
    // for this refactor, which otherwise touches every B8a buffer.
    m->prefill_w = Impl::MAX_MC;
    if (const char* pw = std::getenv("RADAR_PREFILL_W")) {
        int v = std::atoi(pw);
        if (v >= 1 && v <= Impl::MAX_PREFILL) {
            m->prefill_w = v;
        } else {
            std::fprintf(stderr, "[b13] ignoring RADAR_PREFILL_W=%s (want 1..%d)\n", pw,
                         Impl::MAX_PREFILL);
        }
    }
    std::fprintf(stderr, "[b13] prefill width W=%d (allocated %d; RADAR_PREFILL_W)\n",
                 m->prefill_w, Impl::MAX_PREFILL);
    m->q8_k = std::max(std::max(H, Q), std::max(KV, FF));
    CU_CHECK(cudaMalloc(&m->d_q8s, static_cast<size_t>(MC) * m->q8_k));
    CU_CHECK(cudaMalloc(&m->d_q8ds,
                        static_cast<size_t>(MC) * (m->q8_k / 32) * sizeof(Q8_1Ds)));
    // RADAR_B11_TILE=RxC: same-session A/B of the int8 prefill tile. The tile is
    // a pure memory-shape knob (it never changes the accumulation order), so this
    // is how the default was chosen -- see docs/34 §7.
    if (const char* tile = std::getenv("RADAR_B11_TILE")) {
        int rw = 0, nc = 0;
        if (std::sscanf(tile, "%dx%d", &rw, &nc) == 2 && rw * nc == 32 && rw > 0 && nc > 0) {
            m->tile_rw = rw;
            m->tile_nc = nc;
        } else {
            std::fprintf(stderr, "[b11] ignoring malformed RADAR_B11_TILE=%s\n", tile);
        }
    }
    // B12: RADAR_Q4_MMA=1 -> tensor-core inner loop. Bit-exact with the dp4a
    // route (int8-ref-test proves both against the NumPy reference), so this is
    // purely a throughput switch and can be A/B'd inside one session.
    //
    // ⚠️ Uses env_on, not getenv-presence: a presence test makes `RADAR_Q4_MMA=0`
    // mean ON, which silently made an A/B script compare MMA against MMA and
    // produced a wrong "dp4a and MMA are equal" conclusion (2026-10-11).
    m->mma = env_on("RADAR_Q4_MMA");
    std::fprintf(stderr, "[b11] int8 activations %s (prefill, quantized weights only; tile %dx%d)\n",
                 m->int8_act ? "ON" : "OFF (RADAR_Q4_FP32_ACT)", m->tile_rw, m->tile_nc);
    if (m->int8_act) {
        std::fprintf(stderr, "[b12] prefill GEMM inner loop: %s\n",
                     m->mma ? "int8 MMA (m16n8k32, tensor cores; RADAR_Q4_MMA)"
                            : "dp4a (B11 default; set RADAR_Q4_MMA=1 for the MMA route)");
    }
    // B13: Q6_K on the int8 route. Default OFF -- this one DOES change end-to-end
    // ids (int8 activation quantisation loses precision), unlike B12's pure
    // instruction swap, so it is a Tier-B change (docs/37 §1.1).
    m->q6k_int8 = env_on("RADAR_Q6K_INT8");
    std::fprintf(stderr, "[b13] Q6_K prefill: %s\n",
                 (m->int8_act && m->q6k_int8)
                     ? "int8 m16n8k16 (tensor cores; RADAR_Q6K_INT8)"
                     : "fp32 (B11/B12 default; set RADAR_Q6K_INT8=1 to move it)");
    CU_CHECK(cudaDeviceSynchronize());
    m->prof_init();
    // CUDA graph replay per decode token (RADAR_NO_CUDA_GRAPH=1 to force eager;
    // profiling implies eager so the per-phase events stay meaningful).
    m->use_graph = !m->prof && std::getenv("RADAR_NO_CUDA_GRAPH") == nullptr;
    return gm;
}

void GpuModel::forward_logits(int32_t token, int pos, float* logits_out) {
    Impl* m = impl_;
    const ModelConfig& c = m->cfg;
    if (token < 0 || token >= c.vocab) fail("token out of range");
    if (pos < 0 || pos >= m->kv_cap) fail("position exceeds kv_cap");
    // B10b: keep the token history in sync so a later prefill can hash prefixes.
    if (pos == 0 && !m->seq_active[0]) m->begin_sequence(0);
    if (static_cast<size_t>(pos) < m->hist[0].size()) {
        m->hist[0].resize(static_cast<size_t>(pos));
    } else if (static_cast<size_t>(pos) > m->hist[0].size()) {
        m->hist[0].clear();
        m->hist_valid[0] = 0;
    }
    m->hist[0].push_back(token);
    // B10: make sure slot 0 owns a block covering `pos` (lazy growth).
    m->ensure_blocks(0, pos + 1);

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
    CU_CHECK(cudaMemcpyAsync(logits_out, m->d_logits, c.vocab * sizeof(float),
                             cudaMemcpyDeviceToHost, m->stream));
    CU_CHECK(cudaStreamSynchronize(m->stream));
    m->prof_end(Impl::PH_D2H);
    m->prof_end(Impl::PH_TOTAL);
    m->prof_collect();
}

int32_t GpuModel::forward_argmax(int32_t token, int pos) {
    Impl* m = impl_;
    forward_logits(token, pos, m->h_logits);
    const int vocab = m->cfg.vocab;
    int32_t best = 0;
    for (int i = 1; i < vocab; ++i) if (m->h_logits[i] > m->h_logits[best]) best = i;
    return best;
}

// B8a: batch prefill. Feeds tokens[0..n) at positions 0..n-1 in MAX_MC-column
// chunks (weights read once per chunk). Bitwise-equivalent to n sequential
// forward_argmax state updates; the last position's logits go to logits_out.
void GpuModel::prefill_batch(const int32_t* tokens, int n, float* logits_out,
                             int slot, int pos0) {
    Impl* m = impl_;
    const ModelConfig& c = m->cfg;
    if (n <= 0 || pos0 < 0 || pos0 + n > m->kv_cap) fail("prefill range out of kv_cap");
    if (slot < 0 || slot >= m->n_slots) fail("prefill slot out of range");
    // B10/B10b: pos0 == 0 with no live sequence means "a new sequence starts in
    // this slot" → release the previous one's blocks and reset its token history.
    if (pos0 == 0 && !m->seq_active[slot]) m->begin_sequence(slot);
    // The token history is what the prefix cache hashes. It must mirror what is
    // actually in the slot's KV: truncate to pos0 (the caller is overwriting
    // everything after it), or give up on hashing if we never saw position 0.
    if (static_cast<size_t>(pos0) < m->hist[slot].size()) {
        m->hist[slot].resize(static_cast<size_t>(pos0));
    } else if (static_cast<size_t>(pos0) > m->hist[slot].size()) {
        m->hist[slot].clear();
        m->hist_valid[slot] = 0;
    }
    m->hist[slot].insert(m->hist[slot].end(), tokens, tokens + n);
    // ⭐ B10b: leading blocks may come from the cross-request prefix cache, whose
    // KV is already correct AND identical — so skip computing them. Same argument
    // as B9's session LCP: identical tokens at identical positions produce
    // identical KV, bit for bit.
    const int leading = m->ensure_blocks(slot, pos0 + n);
    int skip = leading * BLOCK;
    if (skip >= n) skip = n - 1;   // always compute one position to get logits
    m->last_reused = skip;
    if (skip > 0)
        std::fprintf(stderr, "[b10] prefix reuse: slot=%d prompt=%d reused=%d computed=%d\n",
                     slot, n, skip, n - skip);
    const int W = m->prefill_w;   // runtime prefill width (<= MAX_PREFILL)
    for (int base = skip; base < n; base += W) {
        const int nr = std::min(W, n - base);
        CU_CHECK(cudaMemcpyAsync(m->d_tokens, tokens + base, nr * sizeof(int),
                                 cudaMemcpyHostToDevice, m->stream));
        m->prof_begin(Impl::PH_TOTAL);
        m->prefill_chunk(pos0 + base, nr, slot);
        m->prof_end(Impl::PH_TOTAL);
        m->prof_collect();
    }
    const int last = (n - 1 - skip) % W;  // row of the last position
    m->prof_begin(Impl::PH_LOGITS);
    m->gemv(m->d_embd, m->d_hm + static_cast<size_t>(last) * c.hidden,
            m->d_logits, c.hidden, c.vocab);
    m->prof_end(Impl::PH_LOGITS);
    m->prof_begin(Impl::PH_D2H);
    CU_CHECK(cudaMemcpyAsync(logits_out, m->d_logits, c.vocab * sizeof(float),
                             cudaMemcpyDeviceToHost, m->stream));
    CU_CHECK(cudaStreamSynchronize(m->stream));
    m->prof_end(Impl::PH_D2H);
    m->prof_collect();
    m->prof_print();
}

int GpuModel::max_batch() const { return impl_->max_batch; }
int GpuModel::num_blocks() const { return impl_->num_blocks; }
int GpuModel::used_blocks() const { return impl_->used_blocks; }
int GpuModel::block_size() const { return BLOCK; }

size_t GpuModel::kv_bytes() const {
    return static_cast<size_t>(impl_->num_blocks) * impl_->block_stride * 2 *
           impl_->cfg.n_layer * sizeof(float);
}

size_t GpuModel::kv_worst_case_bytes() const {
    return static_cast<size_t>(impl_->n_slots) * static_cast<size_t>(impl_->max_blocks) *
           impl_->block_stride * 2 * impl_->cfg.n_layer * sizeof(float);
}

void GpuModel::begin_sequence(int slot) {
    if (slot < 0 || slot >= impl_->n_slots) fail("begin_sequence: slot out of range");
    impl_->begin_sequence(slot);
}
int GpuModel::last_reused_tokens() const { return impl_->last_reused; }
int GpuModel::prefix_hits() const { return impl_->prefix_hits; }
int GpuModel::prefix_misses() const { return impl_->prefix_misses; }

// B8b: one decode step for the padded batch. tokens/positions/slots are
// max_batch-element host arrays; padding rows must use slot == max_batch
// (scratch). Graph-captured at the first call, replayed afterwards.
void GpuModel::decode_batch(const int32_t* tokens, const int* positions, const int* slots,
                            int32_t* next_out) {
    Impl* m = impl_;
    const int B = m->max_batch;
    if (B <= 1) fail("decode_batch requires max_batch > 1");
    // B10/B10b: keep each row's token history in sync, then cover every row —
    // including the padding/scratch slot, whose KV store is real even though its
    // attention output is discarded.
    for (int i = 0; i < B; ++i) {
        const int s = slots[i];
        if (static_cast<size_t>(positions[i]) < m->hist[s].size()) {
            m->hist[s].resize(static_cast<size_t>(positions[i]));
        } else if (static_cast<size_t>(positions[i]) > m->hist[s].size()) {
            m->hist[s].clear();
            m->hist_valid[s] = 0;
        }
        m->hist[s].push_back(tokens[i]);
        m->ensure_blocks(s, positions[i] + 1);
    }
    for (int i = 0; i < B; ++i) {
        m->h_bstage[i] = tokens[i];
        m->h_bstage[B + i] = positions[i];
        m->h_bstage[2 * B + i] = slots[i];
    }
    CU_CHECK(cudaMemcpyAsync(m->d_btok, m->h_bstage, B * sizeof(int),
                             cudaMemcpyHostToDevice, m->stream));
    CU_CHECK(cudaMemcpyAsync(m->d_bpos, m->h_bstage + B, B * sizeof(int),
                             cudaMemcpyHostToDevice, m->stream));
    CU_CHECK(cudaMemcpyAsync(m->d_bslot, m->h_bstage + 2 * B, B * sizeof(int),
                             cudaMemcpyHostToDevice, m->stream));
    if (!m->graph_exec_b) {
        if (std::getenv("RADAR_NO_CUDA_GRAPH") != nullptr) {
            m->step_batch_kernels();  // eager debug path
        } else {
            cudaGraph_t g = nullptr;
            CU_CHECK(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
            m->step_batch_kernels();
            CU_CHECK(cudaStreamEndCapture(m->stream, &g));
            CU_CHECK(cudaGraphInstantiate(&m->graph_exec_b, g, nullptr, nullptr, 0));
            CU_CHECK(cudaGraphDestroy(g));
            CU_CHECK(cudaGraphLaunch(m->graph_exec_b, m->stream));
        }
    } else {
        CU_CHECK(cudaGraphLaunch(m->graph_exec_b, m->stream));
    }
    CU_CHECK(cudaMemcpyAsync(m->h_next, m->d_next, B * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, m->stream));
    CU_CHECK(cudaStreamSynchronize(m->stream));
    std::memcpy(next_out, m->h_next, B * sizeof(int32_t));
}

}  // namespace ri
