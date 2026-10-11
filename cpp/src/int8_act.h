#pragma once

#include <cstdint>

// B11: int8 activations (Q8_1-aligned) x Q4_K/Q6_K. See docs/34 section 3.2.1
// for the semantics this file implements and cpp/tests/ref_q4_int8.py for the
// independent NumPy twin of it.
//
// Scope (docs/34 section 3.1): quantized weights only, prefill only. F16 weights
// and the decode path keep fp32 activations, so test_b4's 440/440 and the whole
// B7/B10 decode record are untouched by this file.
//
// This header is the *host reference*: it is what layer-1 of the cross-check
// ("our CUDA kernel vs an independent implementation") is measured against. It
// is not wired into the CPU engine backend.

namespace ri {

// Per-32-value scale record. 8 bytes.
//
// Deliberate difference from llama.cpp's block_q8_1: the offset term of Q4_K
// needs sum(qs) and llama.cpp's CUDA mmvq path recomputes it with a dp4a against
// 0x01010101 (vecdotq.cuh:520), so storing the exact integer is the same value
// with less work. The half `s` field is therefore not carried.
struct Q8_1Ds {
    float d;         // float(half(amax/127)); the half-rounded value is what the
                     // dot uses, exactly as llama.cpp CUDA does (it quantizes
                     // with the fp32 scale but stores/uses fp16)
    int32_t sum_q;   // exact sum of the block's 32 int8 values
};
static_assert(sizeof(Q8_1Ds) == 8, "Q8_1Ds must be 8 bytes");

// Activation layout, block-major: [k/32][ncol][32] quants plus a separate
// [k/32][ncol] scale array.
//
//   qs[block*ncol*32 + c*32 + lane]
//   ds[block*ncol + c]
//
// The *column-minor* order inside a block is what makes the CUDA gemv viable:
// with a column-major [ncol][k] layout one warp load touches one 128-byte line
// per column (16 lines), and B11 measured that as a 2.4x net SLOWDOWN versus the
// fp32 route despite ~2x fewer instructions. Packing a block's columns together
// turns the same load into 4 lines. `ncol` here is the allocated column count
// (16 = the prefill chunk width), which is what the kernel's tile assumes.
// k must be a multiple of 32.
void quantize_q8_1_host(const float* x, int ncol, int k, int8_t* qs, Q8_1Ds* ds);

// y[c][out] for c in [0, ncol). `w` is the raw quantized tensor with `in` values
// per row; `qs`/`ds` come from quantize_q8_1_host. Every fp32 operation is a
// separately-rounded multiply or add (see the -ffp-contract=off property on
// int8_act.cpp): the parenthesisation is part of the spec.
void matvec_q4_k_q8_1_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                           float* y, int in, int out, int ncol);
void matvec_q6_k_q8_1_host(const uint8_t* w, const int8_t* qs, const Q8_1Ds* ds,
                           float* y, int in, int out, int ncol);

}  // namespace ri
