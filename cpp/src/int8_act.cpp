// B11 host reference: int8 activations (Q8_1-aligned) x Q4_K/Q6_K weights.
//
// Two independent implementations of one written-down spec (docs/34 3.2.1):
// this file and cpp/tests/ref_q4_int8.py. Layer 1 of the B11 cross-check is
// "they agree bit-exactly", first host-vs-NumPy, then CUDA-vs-NumPy.
//
// The fp32 parenthesisation is part of the spec, so:
//   * every product/add is written as its own statement (no reassociation),
//   * int8_act.cpp is compiled with -ffp-contract=off (CMake) so GCC cannot
//     fuse `a*b + c` into an FMA and change the rounding,
//   * the CUDA kernel uses __fmul_rn/__fadd_rn for the same reason.
// Round-to-nearest int8 quantization is half-away-from-zero (C roundf /
// llama.cpp's roundf), which is *not* what np.round does -- see the NumPy twin.

#include "int8_act.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "f16.h"

namespace ri {
namespace {

constexpr int QK_K = 256;
constexpr int Q4K_BLOCK_BYTES = 144;
constexpr int Q6K_BLOCK_BYTES = 210;

uint16_t load_f16_bits(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

// fp32 -> fp16 round-to-nearest-even (the software twin of __float2half_rn, so
// the reference does not depend on F16C being enabled).
uint16_t f32_to_f16_rn(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp == 0xFFu) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    const int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return sign;
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t hm = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (hm & 1u))) ++hm;
        return static_cast<uint16_t>(sign | hm);
    }
    uint32_t half = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // carry -> exp is correct
    return static_cast<uint16_t>(sign | half);
}

// ggml get_scale_min_k4 (identical to the CPU reference in quants.cpp).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

}  // namespace

void quantize_q8_1_host(const float* x, int ncol, int k, int8_t* qs, Q8_1Ds* ds) {
    const int nb = k / 32;
    for (int c = 0; c < ncol; ++c) {
        const float* xc = x + static_cast<size_t>(c) * k;
        for (int b = 0; b < nb; ++b) {
            const float* xb = xc + static_cast<size_t>(b) * 32;
            float amax = 0.0f;
            for (int l = 0; l < 32; ++l) {
                const float a = std::fabs(xb[l]);
                if (a > amax) amax = a;
            }
            const float d32 = amax / 127.0f;
            const bool zero = (d32 == 0.0f);
            int32_t sum = 0;
            for (int l = 0; l < 32; ++l) {
                // divide (not multiply by a reciprocal): llama.cpp CUDA rounds
                // x/d, and the two differ in the last bit.
                const float v = zero ? xb[l] : xb[l] / d32;
                const float r = std::round(v);              // half away from zero
                const int32_t q = zero ? 0 : static_cast<int32_t>(r);
                qs[static_cast<size_t>(b) * ncol * 32 + static_cast<size_t>(c) * 32 + l] =
                    static_cast<int8_t>(q);
                sum += q;
            }
            ds[static_cast<size_t>(b) * ncol + c].sum_q = sum;
            ds[static_cast<size_t>(b) * ncol + c].d = f16_to_f32(f32_to_f16_rn(d32));
        }
    }
}

// Activation index for (32-value block `blk`, column `c`, lane `l`) under the
// block-major [k/32][ncol][32] layout.
static inline size_t q8_index(int blk, int c, int l, int ncol) {
    return static_cast<size_t>(blk) * ncol * 32 + static_cast<size_t>(c) * 32 + l;
}

void matvec_q4_k_q8_1_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                           float* y, int in, int out, int ncol) {
    const int nb = in / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * Q4K_BLOCK_BYTES;
    std::vector<float> sd(ncol), sm(ncol), acc(ncol);
    for (int r = 0; r < out; ++r) {
        const uint8_t* wr = w + static_cast<size_t>(r) * row_bytes;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int b = 0; b < nb; ++b) {
            const uint8_t* sb = wr + static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
            const float d4 = f16_to_f32(load_f16_bits(sb));
            const float dmin = f16_to_f32(load_f16_bits(sb + 2));
            std::fill(sd.begin(), sd.end(), 0.0f);
            std::fill(sm.begin(), sm.end(), 0.0f);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                get_scale_min_k4(j, sb + 4, &sc, &m);
                const uint8_t* qsj = sb + 16 + (j >> 1) * 32;
                const int shift = (j & 1) * 4;
                for (int c = 0; c < ncol; ++c) {
                    const int32_t dot2 = ds[static_cast<size_t>(b * 8 + j) * ncol + c].sum_q;
                    int32_t dot1 = 0;
                    for (int l = 0; l < 32; ++l) {
                        const int nib = (qsj[l] >> shift) & 0xF;
                        dot1 += nib * static_cast<int32_t>(
                            qs[q8_index(b * 8 + j, c, l, ncol)]);
                    }
                    const float d8 = ds[static_cast<size_t>(b * 8 + j) * ncol + c].d;
                    sd[c] += d8 * static_cast<float>(dot1 * static_cast<int32_t>(sc));
                    sm[c] += d8 * static_cast<float>(dot2 * static_cast<int32_t>(m));
                }
            }
            for (int c = 0; c < ncol; ++c) {
                acc[c] += d4 * sd[c] - dmin * sm[c];
            }
        }
        for (int c = 0; c < ncol; ++c) y[static_cast<size_t>(c) * out + r] = acc[c];
    }
}

void matvec_q6_k_q8_1_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                           float* y, int in, int out, int ncol) {
    const int nb = in / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * Q6K_BLOCK_BYTES;
    std::vector<float> sd(ncol), acc(ncol);
    for (int r = 0; r < out; ++r) {
        const uint8_t* wr = w + static_cast<size_t>(r) * row_bytes;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int b = 0; b < nb; ++b) {
            const uint8_t* sb = wr + static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
            const float d6 = f16_to_f32(load_f16_bits(sb + 208));
            std::fill(sd.begin(), sd.end(), 0.0f);
            for (int nn = 0; nn < 2; ++nn) {
                const uint8_t* ql = sb + nn * 64;
                const uint8_t* qh = sb + 128 + nn * 32;
                const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
                for (int qq = 0; qq < 4; ++qq) {
                    const int g = nn * 4 + qq;
                    const int vshift = 2 * qq;
                    const int8_t* qb = qs + q8_index(b * 8 + g, 0, 0, ncol);
                    for (int c = 0; c < ncol; ++c) {
                        const float d8 = ds[static_cast<size_t>(b * 8 + g) * ncol + c].d;
                        const int8_t* qbc = qb + static_cast<size_t>(c) * 32;
                        for (int half = 0; half < 2; ++half) {
                            int32_t dot = 0;
                            for (int l = half * 16; l < half * 16 + 16; ++l) {
                                const uint32_t b8 = (qq & 1) ? ql[l + 32] : ql[l];
                                const int nib = (qq < 2) ? (b8 & 0xF) : (b8 >> 4);
                                const int q6 = (nib | (((qh[l] >> vshift) & 3) << 4)) - 32;
                                dot += q6 * static_cast<int32_t>(qbc[l]);
                            }
                            const int32_t scg = static_cast<int32_t>(sc[half + 2 * qq]);
                            sd[c] += d8 * static_cast<float>(dot * scg);
                        }
                    }
                }
            }
            for (int c = 0; c < ncol; ++c) {
                acc[c] += d6 * sd[c];
            }
        }
        for (int c = 0; c < ncol; ++c) y[static_cast<size_t>(c) * out + r] = acc[c];
    }
}

}  // namespace ri
