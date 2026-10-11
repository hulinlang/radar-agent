#pragma once

#include <cstdint>
#include <cstring>

namespace ri {

// IEEE 754 half -> single, exact (including subnormals, inf, nan).
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        // Zero or subnormal: value = mant * 2^-24.
        const float v = mant == 0 ? 0.0f : static_cast<float>(mant) * 5.9604644775390625e-08f;
        return (h & 0x8000u) ? -v : v;
    }
    uint32_t bits;
    if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

}  // namespace ri
