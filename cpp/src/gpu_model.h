#pragma once

#include <cstdint>

namespace ri {

class GGUFFile;
struct ModelConfig;

// CUDA backend (B6b/B7): F16 or Q4_K/Q6_K weights resident in VRAM, decode
// gemv via handwritten kernels (F32 activations/accumulation), small
// orchestration kernels (rmsnorm / rope / attention / silu) handwritten.
// Same math as the CPU semantic reference; only the execution site differs.
class GpuModel {
public:
    // Throws std::runtime_error on any CUDA/loading failure.
    // max_batch > 1 enables the B8b slot pool: the KV cache becomes
    // [layer][k/v][slot][kv_head][kv_cap][head_dim] with max_batch+1 slots
    // (the extra slot is scratch for padded rows in batch decode).
    static GpuModel load(const GGUFFile& f, int kv_cap, int max_batch = 1);
    ~GpuModel();
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;
    GpuModel(GpuModel&&) noexcept;

    const ModelConfig& config() const;
    int max_batch() const;

    // Runs one token at absolute position pos; returns the argmax token id.
    int32_t forward_argmax(int32_t token, int pos);

    // B8a: batch prefill. Feeds tokens[0..n) at positions 0..n-1 of KV slot
    // `slot` with multi-column gemv kernels (weights read once per 16-token
    // chunk). Attention score dots use a parallel reduction (allclose to, not
    // bitwise-identical with, sequential steps); everything else preserves
    // per-element op order. Writes the last position's logits (vocab floats)
    // to logits_out. Eager path (not graph-captured).
    void prefill_batch(const int32_t* tokens, int n, float* logits_out, int slot = 0);

    // B8b: one decode step for a padded batch of max_batch rows. Row i feeds
    // tokens[i] at positions[i] of KV slot slots[i]; padding rows must use
    // slot max_batch (scratch). next_out receives max_batch argmax ids
    // (GPU-side argmax, first-index tie-break identical to the host loop).
    // Graph-captured after the first call.
    void decode_batch(const int32_t* tokens, const int* positions, const int* slots,
                      int32_t* next_out);

private:
    GpuModel() = default;
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace ri
