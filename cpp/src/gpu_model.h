#pragma once

#include <cstdint>

namespace ri {

class GGUFFile;
struct ModelConfig;

// CUDA backend (B6b): F16 weights resident in VRAM, gemv via cublasGemmEx
// (F16 storage / F32 compute / F32 output), small orchestration kernels
// (rmsnorm / rope / attention / silu) handwritten. Same math as the CPU
// semantic reference; only the execution site differs.
class GpuModel {
public:
    // Throws std::runtime_error on any CUDA/loading failure.
    static GpuModel load(const GGUFFile& f, int kv_cap);
    ~GpuModel();
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;
    GpuModel(GpuModel&&) noexcept;

    const ModelConfig& config() const;

    // Runs one token at absolute position pos; returns the argmax token id.
    int32_t forward_argmax(int32_t token, int pos);

private:
    GpuModel() = default;
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace ri
