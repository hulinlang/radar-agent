#pragma once

// Device-side Q4_K/Q6_K scalar dequant helpers, shared by gpu_quants.cu
// (gemv/dequant kernels) and gpu_model.cu (batch embed kernel). Same scalar
// op order as the CPU reference (quants.cpp).

#include <cuda_fp16.h>

namespace ri {
namespace gpuq {
namespace detail {

constexpr int QK_K_D = 256;
constexpr int Q4K_BLOCK_BYTES_D = 144;
constexpr int Q6K_BLOCK_BYTES_D = 210;

// ggml get_scale_min_k4: 8 sub-block scales and mins, 6-bit each, split-packed
// into 12 bytes.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q,
                                                 uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantized value i (0..255) of one Q4_K super-block.
__device__ __forceinline__ float q4k_at(const uint8_t* sb, int i) {
    const float d = __half2float(*reinterpret_cast<const __half*>(sb));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(sb + 2));
    const int j = i >> 5;
    const int l = i & 31;
    uint8_t sc, m;
    get_scale_min_k4(j, sb + 4, &sc, &m);
    const uint8_t byte = sb[16 + (j >> 1) * 32 + l];
    const int nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
    return d * sc * nib - dmin * m;
}

// Dequantized value i (0..255) of one Q6_K super-block.
__device__ __forceinline__ float q6k_at(const uint8_t* sb, int i) {
    const int nn = i >> 7;
    const int r = i & 127;
    const int qq = r >> 5;
    const int l = r & 31;
    const uint8_t* ql = sb + nn * 64;
    const uint8_t* qh = sb + 128 + nn * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
    const float d = __half2float(*reinterpret_cast<const __half*>(sb + 208));
    const uint8_t b = ql[l + (qq & 1) * 32];
    const int nib = (qq < 2) ? (b & 0xF) : (b >> 4);
    const int q = (nib | (((qh[l] >> (2 * qq)) & 3) << 4)) - 32;
    return d * sc[l / 16 + 2 * qq] * q;
}

}  // namespace detail
}  // namespace gpuq
}  // namespace ri
