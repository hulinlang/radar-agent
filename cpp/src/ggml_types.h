#pragma once

#include <cstdint>

namespace ri {

struct GgmlTypeInfo {
    const char* name;
    uint64_t block_size;  // elements per block
    uint64_t type_size;   // bytes per block
};

// Returns nullptr for an unknown type id (caller must treat as an error).
// Values follow ggml-common.h: block_size x type_size per block.
inline const GgmlTypeInfo* ggml_type_info(uint32_t type) {
    switch (type) {
        case 0:  { static const GgmlTypeInfo t{"F32",   1,   4}; return &t; }
        case 1:  { static const GgmlTypeInfo t{"F16",   1,   2}; return &t; }
        case 2:  { static const GgmlTypeInfo t{"Q4_0", 32,  18}; return &t; }
        case 3:  { static const GgmlTypeInfo t{"Q4_1", 32,  20}; return &t; }
        case 6:  { static const GgmlTypeInfo t{"Q5_0", 32,  22}; return &t; }
        case 7:  { static const GgmlTypeInfo t{"Q5_1", 32,  24}; return &t; }
        case 8:  { static const GgmlTypeInfo t{"Q8_0", 32,  34}; return &t; }
        case 9:  { static const GgmlTypeInfo t{"Q8_1", 32,  40}; return &t; }
        case 10: { static const GgmlTypeInfo t{"Q2_K", 256, 84}; return &t; }
        case 11: { static const GgmlTypeInfo t{"Q3_K", 256, 110}; return &t; }
        case 12: { static const GgmlTypeInfo t{"Q4_K", 256, 144}; return &t; }
        case 13: { static const GgmlTypeInfo t{"Q5_K", 256, 176}; return &t; }
        case 14: { static const GgmlTypeInfo t{"Q6_K", 256, 210}; return &t; }
        case 15: { static const GgmlTypeInfo t{"Q8_K", 256, 292}; return &t; }
        default: return nullptr;
    }
}

}  // namespace ri
