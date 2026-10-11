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

// ---------------------------------------------------------------- B12 step 3
// m16n8k32 with 8-bit operands: A is 16x32 int8 (row-major), B is 32x8 int8
// (column-major), C/D is 16x8 int32. Each 32-bit register holds 4 int8 along K.
// lane = gid*4 + tig, gid = lane>>2, tig = lane&3.
namespace {

inline void mma_a_map(int lane, int reg, int b, int* row, int* k) {
    const int gid = lane >> 2, tig = lane & 3;
    *row = gid + ((reg & 1) ? 8 : 0);
    *k = ((reg >> 1) ? 16 : 0) + tig * 4 + b;
}

inline void mma_b_map(int lane, int reg, int b, int* k, int* col) {
    const int gid = lane >> 2, tig = lane & 3;
    *k = (reg ? 16 : 0) + tig * 4 + b;
    *col = gid;
}

inline void mma_c_map(int lane, int reg, int* row, int* col) {
    const int gid = lane >> 2, tig = lane & 3;
    *row = gid + ((reg >= 2) ? 8 : 0);
    *col = tig * 2 + (reg & 1);
}

}  // namespace

bool mma_fragment_coverage_ok() {
    {   // A: 32 lanes x 4 regs x 4 bytes = 512 = 16 x 32
        int seen[16][32] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 4; ++reg)
                for (int b = 0; b < 4; ++b) {
                    int r, k;
                    mma_a_map(lane, reg, b, &r, &k);
                    if (r < 0 || r >= 16 || k < 0 || k >= 32) return false;
                    if (++seen[r][k] != 1) return false;
                }
    }
    {   // B: 32 x 2 x 4 = 256 = 32 x 8
        int seen[32][8] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 2; ++reg)
                for (int b = 0; b < 4; ++b) {
                    int k, c;
                    mma_b_map(lane, reg, b, &k, &c);
                    if (k < 0 || k >= 32 || c < 0 || c >= 8) return false;
                    if (++seen[k][c] != 1) return false;
                }
    }
    {   // C/D: 32 x 4 = 128 = 16 x 8
        int seen[16][8] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 4; ++reg) {
                int r, c;
                mma_c_map(lane, reg, &r, &c);
                if (r < 0 || r >= 16 || c < 0 || c >= 8) return false;
                if (++seen[r][c] != 1) return false;
            }
    }
    return true;
}

void matvec_q4_k_q8_1_mma_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                               float* y, int in, int out, int ncol) {
    const int nb = in / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * Q4K_BLOCK_BYTES;
    // Row tiles of 16 and column tiles of 8 = exactly one MMA's M and N.
    for (int rt = 0; rt < out; rt += 16) {
        const int nr = std::min(16, out - rt);
        for (int ct = 0; ct < ncol; ct += 8) {
            const int nc = std::min(8, ncol - ct);
            std::vector<float> acc(static_cast<size_t>(nr) * nc, 0.0f);
            for (int b = 0; b < nb; ++b) {
                // Per (row,col) partials, reset each super-block, exactly as the
                // layout-agnostic reference does -- the fp32 parenthesisation and
                // the j/b accumulation order are part of the spec.
                std::vector<float> sd(static_cast<size_t>(nr) * nc, 0.0f);
                std::vector<float> sm(static_cast<size_t>(nr) * nc, 0.0f);
                for (int j = 0; j < 8; ++j) {
                    // ---- A tile, gathered through the fragment map ----
                    // sub-block j covers k' = j*32 .. j*32+31, i.e. exactly K=32.
                    int8_t A[16][32];
                    for (int lane = 0; lane < 32; ++lane)
                        for (int reg = 0; reg < 4; ++reg)
                            for (int bb = 0; bb < 4; ++bb) {
                                int r, k;
                                mma_a_map(lane, reg, bb, &r, &k);
                                const uint8_t* sb =
                                    w + static_cast<size_t>(rt + r) * row_bytes +
                                    static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
                                const uint8_t* qsj = sb + 16 + (j >> 1) * 32;
                                A[r][k] = static_cast<int8_t>((qsj[k] >> ((j & 1) * 4)) & 0xF);
                            }
                    // ---- B tile, gathered through the fragment map ----
                    int8_t B[32][8];
                    for (int lane = 0; lane < 32; ++lane)
                        for (int reg = 0; reg < 2; ++reg)
                            for (int bb = 0; bb < 4; ++bb) {
                                int k, c;
                                mma_b_map(lane, reg, bb, &k, &c);
                                if (ct + c < ncol)
                                    B[k][c] = qs[q8_index(b * 8 + j, ct + c, k, ncol)];
                            }
                    // ---- the MMA's own arithmetic: exact int32 dot per output ----
                    for (int r = 0; r < nr; ++r) {
                        const uint8_t* sb = w + static_cast<size_t>(rt + r) * row_bytes +
                                            static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
                        // d4/dmin are per (row, super-block) and are applied after
                        // the j loop, exactly as the reference does.
                        uint8_t sc, m;
                        get_scale_min_k4(j, sb + 4, &sc, &m);
                        for (int n = 0; n < nc; ++n) {
                            int32_t dot1 = 0;
                            for (int k = 0; k < 32; ++k) dot1 += A[r][k] * B[k][n];
                            const int32_t dot2 =
                                ds[static_cast<size_t>(b * 8 + j) * ncol + ct + n].sum_q;
                            const float d8 = ds[static_cast<size_t>(b * 8 + j) * ncol + ct + n].d;
                            const size_t o = static_cast<size_t>(r) * nc + n;
                            sd[o] += d8 * static_cast<float>(dot1 * static_cast<int32_t>(sc));
                            sm[o] += d8 * static_cast<float>(dot2 * static_cast<int32_t>(m));
                        }
                    }
                }
                for (int r = 0; r < nr; ++r) {
                    const uint8_t* sb = w + static_cast<size_t>(rt + r) * row_bytes +
                                        static_cast<size_t>(b) * Q4K_BLOCK_BYTES;
                    const float d4 = f16_to_f32(load_f16_bits(sb));
                    const float dmin = f16_to_f32(load_f16_bits(sb + 2));
                    for (int n = 0; n < nc; ++n) {
                        const size_t o = static_cast<size_t>(r) * nc + n;
                        acc[o] += d4 * sd[o] - dmin * sm[o];
                    }
                }
            }
            for (int r = 0; r < nr; ++r)
                for (int n = 0; n < nc; ++n)
                    y[static_cast<size_t>(ct + n) * out + rt + r] =
                        acc[static_cast<size_t>(r) * nc + n];
        }
    }
}

// ---------------------------------------------------------------- B13 step 2
// m16n8k16 int8: A is 16x16 (2 regs/thread), B is 16x8 (1 reg), C/D is 16x8
// int32 (4 regs). Still lane = gid*4 + tig.
namespace {

inline void mma16_a_map(int lane, int reg, int b, int* row, int* k) {
    const int gid = lane >> 2, tig = lane & 3;
    *row = gid + (reg ? 8 : 0);
    *k = tig * 4 + b;
}

inline void mma16_b_map(int lane, int b, int* k, int* col) {
    const int gid = lane >> 2, tig = lane & 3;
    *k = tig * 4 + b;
    *col = gid;
}

inline void mma16_c_map(int lane, int reg, int* row, int* col) {
    const int gid = lane >> 2, tig = lane & 3;
    *row = gid + ((reg >= 2) ? 8 : 0);
    *col = tig * 2 + (reg & 1);
}

// Q6_K value at (row, super-block b, nn, qq) with l in [0,32) -- the q6k_at
// formula, kept identical to gpu_quants.cuh and to matvec_q6_k_q8_1_host.
inline int q6k_value(const uint8_t* sb, int nn, int qq, int l) {
    const uint8_t* ql = sb + nn * 64;
    const uint8_t* qh = sb + 128 + nn * 32;
    const uint8_t byte = ql[l + (qq & 1) * 32];
    const int nib = (qq < 2) ? (byte & 0xF) : (byte >> 4);
    return (nib | (((qh[l] >> (2 * qq)) & 3) << 4)) - 32;
}

}  // namespace

bool mma16_fragment_coverage_ok() {
    {   // A: 32 lanes x 2 regs x 4 bytes = 256 = 16 x 16
        int seen[16][16] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 2; ++reg)
                for (int b = 0; b < 4; ++b) {
                    int r, k;
                    mma16_a_map(lane, reg, b, &r, &k);
                    if (r < 0 || r >= 16 || k < 0 || k >= 16) return false;
                    if (++seen[r][k] != 1) return false;
                }
    }
    {   // B: 32 x 1 x 4 = 128 = 16 x 8
        int seen[16][8] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int b = 0; b < 4; ++b) {
                int k, c;
                mma16_b_map(lane, b, &k, &c);
                if (k < 0 || k >= 16 || c < 0 || c >= 8) return false;
                if (++seen[k][c] != 1) return false;
            }
    }
    {   // C/D: 32 x 4 = 128 = 16 x 8
        int seen[16][8] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 4; ++reg) {
                int r, c;
                mma16_c_map(lane, reg, &r, &c);
                if (r < 0 || r >= 16 || c < 0 || c >= 8) return false;
                if (++seen[r][c] != 1) return false;
            }
    }
    return true;
}

void matvec_q6_k_q8_1_mma_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                               float* y, int in, int out, int ncol) {
    const int nb = in / QK_K;
    const size_t row_bytes = static_cast<size_t>(nb) * Q6K_BLOCK_BYTES;
    for (int rt = 0; rt < out; rt += 16) {
        const int nr = std::min(16, out - rt);
        for (int ct = 0; ct < ncol; ct += 8) {
            const int nc = std::min(8, ncol - ct);
            std::vector<float> acc(static_cast<size_t>(nr) * nc, 0.0f);
            for (int b = 0; b < nb; ++b) {
                // sd is reset per super-block and consumed by `acc += d6 * sd`;
                // the accumulation order (nn, qq, half) and the fp32
                // parenthesisation below are the spec (matvec_q6_k_q8_1_host).
                std::vector<float> sd(static_cast<size_t>(nr) * nc, 0.0f);
                for (int nn = 0; nn < 2; ++nn) {
                    for (int qq = 0; qq < 4; ++qq) {
                        const int g = nn * 4 + qq;
                        for (int half = 0; half < 2; ++half) {
                            // ---- A tile (16 rows x 16 k), through the map ----
                            // K=16 is exactly one half-block of Q6_K's scale.
                            int8_t A[16][16];
                            for (int lane = 0; lane < 32; ++lane)
                                for (int reg = 0; reg < 2; ++reg)
                                    for (int bb = 0; bb < 4; ++bb) {
                                        int r, k;
                                        mma16_a_map(lane, reg, bb, &r, &k);
                                        const uint8_t* sb =
                                            w + static_cast<size_t>(rt + r) * row_bytes +
                                            static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
                                        A[r][k] = static_cast<int8_t>(
                                            q6k_value(sb, nn, qq, half * 16 + k));
                                    }
                            // ---- B tile (16 k x 8 cols), through the map ----
                            int8_t B[16][8];
                            for (int lane = 0; lane < 32; ++lane)
                                for (int bb = 0; bb < 4; ++bb) {
                                    int k, c;
                                    mma16_b_map(lane, bb, &k, &c);
                                    if (ct + c < ncol)
                                        B[k][c] = qs[q8_index(b * 8 + g, ct + c,
                                                              half * 16 + k, ncol)];
                                }
                            // ---- exact int32 dot + the reference's epilogue ----
                            for (int r = 0; r < nr; ++r) {
                                const uint8_t* sb =
                                    w + static_cast<size_t>(rt + r) * row_bytes +
                                    static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
                                const int8_t* scp = reinterpret_cast<const int8_t*>(sb + 192) + nn * 8;
                                const int32_t scg = static_cast<int32_t>(scp[half + 2 * qq]);
                                for (int n = 0; n < nc; ++n) {
                                    int32_t dot = 0;
                                    for (int k = 0; k < 16; ++k) dot += A[r][k] * B[k][n];
                                    const float d8 =
                                        ds[static_cast<size_t>(b * 8 + g) * ncol + ct + n].d;
                                    const size_t o = static_cast<size_t>(r) * nc + n;
                                    sd[o] += d8 * static_cast<float>(dot * scg);
                                }
                            }
                        }
                    }
                }
                for (int r = 0; r < nr; ++r) {
                    const uint8_t* sb = w + static_cast<size_t>(rt + r) * row_bytes +
                                        static_cast<size_t>(b) * Q6K_BLOCK_BYTES;
                    const float d6 = f16_to_f32(load_f16_bits(sb + 208));
                    for (int n = 0; n < nc; ++n) {
                        const size_t o = static_cast<size_t>(r) * nc + n;
                        acc[o] += d6 * sd[o];
                    }
                }
            }
            for (int r = 0; r < nr; ++r)
                for (int n = 0; n < nc; ++n)
                    y[static_cast<size_t>(ct + n) * out + rt + r] =
                        acc[static_cast<size_t>(r) * nc + n];
        }
    }
}

}  // namespace ri
