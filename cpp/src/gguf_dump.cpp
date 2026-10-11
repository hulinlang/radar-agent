// gguf-dump: canonical text dump of a GGUF file, byte-compared against
// tests/dump_gguf.py (an independent Python parser) as the B1 golden test.
//
// Canonical format (both sides must match byte for byte):
//   GGUF version=3 tensors=310 kv=31 alignment=32
//   KV <key> = <rendered> (<tag>)            scalar: tag in u8..f64
//   KV <key> = [<e0>,<e1>,...] (<etag>[], n=N)   arrays
//   TENSOR <i> <name> <TYPE> dims=[d0,d1] offset=<o> bytes=<b>
//   ACCOUNT header=24 kv=<n> infos=<n> padding=<n> data=<n> tensors_sum=<n> pad_inside=<n> total=<n> ok
//   HEAD <name> [<v0>,...,<v7>]              first 8 values, F32/F16 only
// Floats render as %.17g of the value widened to double; strings are raw
// bytes between double quotes (no escaping).

#include "ggml_types.h"
#include "gguf.h"
#include "f16.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace {

using namespace ri;
using ri::f16_to_f32;

const char* tag_of(ValueType t) {
    switch (t) {
        case ValueType::UINT8:   return "u8";
        case ValueType::INT8:    return "i8";
        case ValueType::UINT16:  return "u16";
        case ValueType::INT16:   return "i16";
        case ValueType::UINT32:  return "u32";
        case ValueType::INT32:   return "i32";
        case ValueType::FLOAT32: return "f32";
        case ValueType::BOOL:    return "bool";
        case ValueType::STRING:  return "str";
        case ValueType::ARRAY:   return "array";
        case ValueType::UINT64:  return "u64";
        case ValueType::INT64:   return "i64";
        case ValueType::FLOAT64: return "f64";
    }
    return "?";
}

void print_double(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    std::fputs(buf, stdout);
}

void print_scalar(const Value& v) {
    switch (v.type) {
        case ValueType::UINT8: case ValueType::UINT16:
        case ValueType::UINT32: case ValueType::UINT64:
            std::printf("%llu", static_cast<unsigned long long>(std::get<uint64_t>(v.data)));
            break;
        case ValueType::INT8: case ValueType::INT16:
        case ValueType::INT32: case ValueType::INT64:
            std::printf("%lld", static_cast<long long>(std::get<int64_t>(v.data)));
            break;
        case ValueType::FLOAT32: case ValueType::FLOAT64:
            print_double(std::get<double>(v.data));
            break;
        case ValueType::BOOL:
            std::fputs(std::get<bool>(v.data) ? "true" : "false", stdout);
            break;
        case ValueType::STRING: {
            const std::string& s = std::get<std::string>(v.data);
            std::fputc('"', stdout);
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fputc('"', stdout);
            break;
        }
        case ValueType::ARRAY:
            break;  // handled by caller
    }
}

void print_kv(const std::string& key, const Value& v) {
    std::printf("KV %s = ", key.c_str());
    if (v.type == ValueType::ARRAY) {
        const Array& a = std::get<Array>(v.data);
        ValueType et = a.empty() ? ValueType::UINT8 : a.front().type;
        std::fputc('[', stdout);
        for (size_t i = 0; i < a.size(); ++i) {
            if (i) std::fputc(',', stdout);
            print_scalar(a[i]);
        }
        std::fputc(']', stdout);
        std::printf(" (%s[], n=%zu)\n", tag_of(et), a.size());
    } else {
        print_scalar(v);
        std::printf(" (%s)\n", tag_of(v.type));
    }
}

void print_head(const GGUFFile& f, const char* name) {
    const TensorInfo* t = f.tensor(name);
    if (!t || (t->ggml_type != 0 && t->ggml_type != 1)) return;
    std::printf("HEAD %s [", name);
    const uint8_t* base = static_cast<const uint8_t*>(f.tensor_data(*t));
    for (int i = 0; i < 8; ++i) {
        if (i) std::fputc(',', stdout);
        if (t->ggml_type == 0) {
            float v;
            std::memcpy(&v, base + i * 4, 4);
            print_double(static_cast<double>(v));
        } else {
            uint16_t h;
            std::memcpy(&h, base + i * 2, 2);
            print_double(static_cast<double>(f16_to_f32(h)));
        }
    }
    std::fputs("]\n", stdout);
}

int run(const char* path) {
    GGUFFile f = GGUFFile::open(path);

    std::printf("GGUF version=%u tensors=%zu kv=%zu alignment=%u\n",
                f.version(), f.tensors().size(), f.kvs().size(), f.alignment());

    for (const auto& p : f.kvs()) print_kv(p.first, p.second);

    for (size_t i = 0; i < f.tensors().size(); ++i) {
        const TensorInfo& t = f.tensors()[i];
        const GgmlTypeInfo* ti = ggml_type_info(t.ggml_type);
        std::printf("TENSOR %zu %s %s dims=[", i, t.name.c_str(), ti->name);
        for (size_t d = 0; d < t.dims.size(); ++d) {
            if (d) std::fputc(',', stdout);
            std::printf("%llu", static_cast<unsigned long long>(t.dims[d]));
        }
        std::printf("] offset=%llu bytes=%llu\n",
                    static_cast<unsigned long long>(t.offset),
                    static_cast<unsigned long long>(t.n_bytes));
    }

    const uint64_t header = 24;
    const uint64_t data_bytes = f.file_size() - f.data_start();
    const uint64_t padding = f.data_start() - header - f.kv_bytes() - f.infos_bytes();
    uint64_t tensors_sum = 0;
    for (const TensorInfo& t : f.tensors()) tensors_sum += t.n_bytes;
    std::printf("ACCOUNT header=%llu kv=%llu infos=%llu padding=%llu data=%llu "
                "tensors_sum=%llu pad_inside=%llu total=%llu ok\n",
                static_cast<unsigned long long>(header),
                static_cast<unsigned long long>(f.kv_bytes()),
                static_cast<unsigned long long>(f.infos_bytes()),
                static_cast<unsigned long long>(padding),
                static_cast<unsigned long long>(data_bytes),
                static_cast<unsigned long long>(tensors_sum),
                static_cast<unsigned long long>(data_bytes - tensors_sum),
                static_cast<unsigned long long>(f.file_size()));

    print_head(f, "output_norm.weight");
    print_head(f, "token_embd.weight");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gguf-dump <file.gguf>\n");
        return 2;
    }
    try {
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gguf-dump: error: %s\n", e.what());
        return 1;
    }
}
