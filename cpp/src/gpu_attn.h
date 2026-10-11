#pragma once

#include <cuda_runtime.h>

// B14: causal prefill attention with K/V tile reuse (FA structure).
//
// Why this exists (docs/38 §0): the shipped attention_prefill is one block per
// (query row, q head), so every query row re-reads all of K and V. At 7525 tokens
// it reaches 1.9% of the FP32 FFMA peak and 36% of the L2 read ceiling, and it is
// 90% of prefill -- which is why B11-B13's 3.33x GEMM win collapses to 1.34x at
// long context.
//
// This kernel keeps a whole prefill chunk (BR rows) of ONE q head per block and
// sweeps K/V in BC-wide tiles with online softmax, so K/V are read once per tile
// instead of once per row. Measured 10.46x on the shipped kernel at T=7525
// (docs/38 §7.4); the config was chosen by that sweep, not by guesswork.
//
// Only the prefill path changes. The decode kernels (attention / attention_batch)
// and the KV page layout are untouched.
namespace ri {
namespace gpua {

// The FA route is specialised for head_dim == 128 (qwen3vl-2b's key_length) and
// for the engine's paginated KV. Both are checked by the launcher rather than
// assumed.
inline constexpr int FA_HEAD_DIM = 128;

enum FaStatus {
    FA_LAUNCHED = 0,      // kernel enqueued on `stream`
    FA_UNSUPPORTED = 1,   // head_dim != 128, or a non-positive row count
    FA_NO_SMEM = 2,       // the opt-in 74 KB dynamic smem was refused
};

// Tile selection. TQ is rows/thread (BR = 8*TQ); TC/CQG/PAD are fixed at the
// values docs/38 §7.4 picked (BC = 32 keys, 256 threads, smem row stride 132).
//
//   cfg 1: choose TQ from `nr`  <- the shipping candidate
//   cfg 2: force TQ=8 (BR=64)   <- for the tile-tolerance net (docs/38 §3.2)
//   cfg 3: force TQ=4 (BR=32)   <- ditto; must agree with cfg 2 to tolerance
//
// `cfg` exists because the acceptance criteria need the result to be
// independent of the tiling, so at least two tilings must be reachable.
FaStatus attention_prefill_fa(const float* q, const float* kv, float* att,
                              size_t layer_k_stride, size_t layer_v_stride,
                              size_t block_stride, size_t head_stride,
                              int n_head, int n_kv_head, int head_dim, int kv_page,
                              const int* block_table, int pos_base, int nr, float scale,
                              int cfg, cudaStream_t stream = nullptr);

// Human-readable name of the tile that `cfg` would pick for this `nr`, for the
// engine banner (so a log says which kernel actually ran).
const char* attention_prefill_fa_tile_name(int cfg, int nr);

}  // namespace gpua
}  // namespace ri
