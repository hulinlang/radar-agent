#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ri {

enum class ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

struct Value;
using Array = std::vector<Value>;

// A metadata value. Integers are widened to 64-bit, floats to double;
// the original ValueType keeps the on-disk precision.
struct Value {
    ValueType type;
    std::variant<uint64_t, int64_t, double, bool, std::string, Array> data;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // fastest-varying dimension first (GGUF convention)
    uint32_t ggml_type = 0;
    uint64_t offset = 0;         // relative to the tensor data region
    uint64_t n_bytes = 0;
};

// A memory-mapped, read-only view of a GGUF file. Tensor data is never copied:
// tensor_data() returns a pointer into the mapping.
class GGUFFile {
public:
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;
    GGUFFile(GGUFFile&& other) noexcept;
    GGUFFile& operator=(GGUFFile&& other) noexcept;
    ~GGUFFile();

    // Throws std::runtime_error on any format violation.
    static GGUFFile open(const std::string& path);

    uint32_t version() const { return version_; }
    uint32_t alignment() const { return alignment_; }
    uint64_t file_size() const { return file_size_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t kv_bytes() const { return kv_bytes_; }
    uint64_t infos_bytes() const { return infos_bytes_; }

    const std::vector<std::pair<std::string, Value>>& kvs() const { return kvs_; }
    const Value* kv(const std::string& key) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* tensor(const std::string& name) const;

    const void* data_region() const { return data_region_; }
    const void* tensor_data(const TensorInfo& t) const {
        return static_cast<const uint8_t*>(data_region_) + t.offset;
    }

private:
    GGUFFile() = default;
    void unmap();
    void validate_accounting();

    const uint8_t* map_ = nullptr;
    uint64_t file_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif

    uint32_t version_ = 0;
    uint32_t alignment_ = 32;
    uint64_t data_start_ = 0;
    uint64_t kv_bytes_ = 0;
    uint64_t infos_bytes_ = 0;

    std::vector<std::pair<std::string, Value>> kvs_;
    std::vector<TensorInfo> tensors_;
    const uint8_t* data_region_ = nullptr;
};

}  // namespace ri
