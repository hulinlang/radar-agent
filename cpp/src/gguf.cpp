#include "gguf.h"

#include "ggml_types.h"

#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace ri {
namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gguf: " + msg);
}

uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

// Bounds-checked little-endian reader over the mapped bytes.
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    size_t remaining() const { return static_cast<size_t>(end - p); }

    void need(size_t n, const char* what) const {
        if (remaining() < n) fail(std::string("unexpected end of file while reading ") + what);
    }

    template <typename T>
    T read(const char* what) {
        need(sizeof(T), what);
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string read_string(const char* what) {
        const uint64_t len = read<uint64_t>(what);
        if (len > remaining()) fail(std::string("string length out of bounds while reading ") + what);
        std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
        p += len;
        return s;
    }
};

Value read_value(Cursor& c, ValueType t, const char* what) {
    switch (t) {
        case ValueType::UINT8:   return Value{t, uint64_t(c.read<uint8_t>(what))};
        case ValueType::INT8:    return Value{t, int64_t(c.read<int8_t>(what))};
        case ValueType::UINT16:  return Value{t, uint64_t(c.read<uint16_t>(what))};
        case ValueType::INT16:   return Value{t, int64_t(c.read<int16_t>(what))};
        case ValueType::UINT32:  return Value{t, uint64_t(c.read<uint32_t>(what))};
        case ValueType::INT32:   return Value{t, int64_t(c.read<int32_t>(what))};
        case ValueType::UINT64:  return Value{t, c.read<uint64_t>(what)};
        case ValueType::INT64:   return Value{t, int64_t(c.read<int64_t>(what))};
        case ValueType::FLOAT32: return Value{t, double(c.read<float>(what))};
        case ValueType::FLOAT64: return Value{t, c.read<double>(what)};
        case ValueType::BOOL:    return Value{t, c.read<uint8_t>(what) != 0};
        case ValueType::STRING:  return Value{t, c.read_string(what)};
        case ValueType::ARRAY: {
            const uint32_t et = c.read<uint32_t>(what);
            if (et > static_cast<uint32_t>(ValueType::FLOAT64) ||
                et == static_cast<uint32_t>(ValueType::ARRAY)) {
                fail("bad array element type");
            }
            const uint64_t n = c.read<uint64_t>(what);
            // Every element costs at least one byte on disk, so n cannot
            // exceed the remaining byte count; this caps corrupt allocations.
            if (n > c.remaining()) fail("array length out of bounds");
            Array a;
            a.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i) {
                a.push_back(read_value(c, static_cast<ValueType>(et), what));
            }
            return Value{t, std::move(a)};
        }
    }
    fail("unknown value type");
}

}  // namespace

GGUFFile::GGUFFile(GGUFFile&& o) noexcept { *this = std::move(o); }

GGUFFile& GGUFFile::operator=(GGUFFile&& o) noexcept {
    if (this != &o) {
        unmap();
        map_ = o.map_;             o.map_ = nullptr;
        file_size_ = o.file_size_; o.file_size_ = 0;
#if defined(_WIN32)
        file_handle_ = o.file_handle_;       o.file_handle_ = nullptr;
        mapping_handle_ = o.mapping_handle_; o.mapping_handle_ = nullptr;
#else
        fd_ = o.fd_;               o.fd_ = -1;
#endif
        version_ = o.version_;
        alignment_ = o.alignment_;
        data_start_ = o.data_start_;
        kv_bytes_ = o.kv_bytes_;
        infos_bytes_ = o.infos_bytes_;
        kvs_ = std::move(o.kvs_);
        tensors_ = std::move(o.tensors_);
        data_region_ = o.data_region_;
        o.data_region_ = nullptr;
    }
    return *this;
}

GGUFFile::~GGUFFile() { unmap(); }

void GGUFFile::unmap() {
    if (map_) {
#if defined(_WIN32)
        UnmapViewOfFile(map_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_) CloseHandle(file_handle_);
        mapping_handle_ = file_handle_ = nullptr;
#else
        munmap(const_cast<uint8_t*>(map_), file_size_);
        if (fd_ >= 0) close(fd_);
        fd_ = -1;
#endif
        map_ = nullptr;
    }
}

GGUFFile GGUFFile::open(const std::string& path) {
    GGUFFile f;

#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) fail("cannot open " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) { CloseHandle(fh); fail("cannot stat " + path); }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { CloseHandle(fh); fail("cannot create mapping for " + path); }
    void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) { CloseHandle(mh); CloseHandle(fh); fail("cannot map " + path); }
    f.file_handle_ = fh;
    f.mapping_handle_ = mh;
    f.map_ = static_cast<const uint8_t*>(view);
    f.file_size_ = static_cast<uint64_t>(sz.QuadPart);
#else
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) fail("cannot open " + path);
    struct stat st;
    if (fstat(f.fd_, &st) != 0) fail("cannot stat " + path);
    f.file_size_ = static_cast<uint64_t>(st.st_size);
    if (f.file_size_ == 0) fail("empty file " + path);
    void* view = mmap(nullptr, f.file_size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (view == MAP_FAILED) fail("cannot mmap " + path);
    f.map_ = static_cast<const uint8_t*>(view);
#endif

    Cursor c{f.map_, f.map_ + f.file_size_};

    char magic[4];
    c.need(4, "magic");
    std::memcpy(magic, c.p, 4);
    c.p += 4;
    if (std::memcmp(magic, "GGUF", 4) != 0) fail("bad magic (not a GGUF file)");

    f.version_ = c.read<uint32_t>("version");
    if (f.version_ != 2 && f.version_ != 3) fail("unsupported version " + std::to_string(f.version_));

    const uint64_t tensor_count = c.read<uint64_t>("tensor_count");
    const uint64_t kv_count = c.read<uint64_t>("metadata_kv_count");
    if (kv_count > c.remaining()) fail("metadata_kv_count out of bounds");
    if (tensor_count > c.remaining()) fail("tensor_count out of bounds");

    const uint64_t kv_begin = static_cast<uint64_t>(c.p - f.map_);
    f.kvs_.reserve(static_cast<size_t>(kv_count));
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.read_string("metadata key");
        const uint32_t vt = c.read<uint32_t>("metadata value type");
        if (vt > static_cast<uint32_t>(ValueType::FLOAT64)) fail("bad metadata value type");
        f.kvs_.emplace_back(std::move(key), read_value(c, static_cast<ValueType>(vt), "metadata value"));
    }
    f.kv_bytes_ = static_cast<uint64_t>(c.p - f.map_) - kv_begin;

    const uint64_t infos_begin = static_cast<uint64_t>(c.p - f.map_);
    f.tensors_.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = c.read_string("tensor name");
        const uint32_t n_dims = c.read<uint32_t>("tensor n_dims");
        if (n_dims > 8) fail("tensor n_dims out of range");
        t.dims.reserve(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.dims.push_back(c.read<uint64_t>("tensor dims"));
        t.ggml_type = c.read<uint32_t>("tensor type");
        t.offset = c.read<uint64_t>("tensor offset");

        const GgmlTypeInfo* ti = ggml_type_info(t.ggml_type);
        if (!ti) fail("tensor " + t.name + " has unknown ggml type " + std::to_string(t.ggml_type));
        uint64_t n_elems = 1;
        for (uint64_t d : t.dims) {
            if (d != 0 && n_elems > UINT64_MAX / d) fail("tensor " + t.name + " size overflow");
            n_elems *= d;
        }
        if (n_elems % ti->block_size != 0) {
            fail("tensor " + t.name + " element count not divisible by block size");
        }
        t.n_bytes = n_elems / ti->block_size * ti->type_size;
        f.tensors_.push_back(std::move(t));
    }
    f.infos_bytes_ = static_cast<uint64_t>(c.p - f.map_) - infos_begin;

    if (const Value* v = f.kv("general.alignment")) {
        if (v->type != ValueType::UINT32) fail("general.alignment must be uint32");
        f.alignment_ = static_cast<uint32_t>(std::get<uint64_t>(v->data));
        if (f.alignment_ == 0) fail("general.alignment must be nonzero");
    }

    f.data_start_ = align_up(static_cast<uint64_t>(c.p - f.map_), f.alignment_);
    if (f.data_start_ > f.file_size_) fail("data region starts beyond end of file");
    f.data_region_ = f.map_ + f.data_start_;

    f.validate_accounting();
    return f;
}

void GGUFFile::validate_accounting() {
    const uint64_t data_bytes = file_size_ - data_start_;
    uint64_t max_end = 0;
    for (const TensorInfo& t : tensors_) {
        if (t.offset % alignment_ != 0) {
            fail("tensor " + t.name + " offset is not aligned");
        }
        if (t.offset > data_bytes || t.n_bytes > data_bytes - t.offset) {
            fail("tensor " + t.name + " extends beyond end of file");
        }
        if (t.offset + t.n_bytes > max_end) max_end = t.offset + t.n_bytes;
    }
    if (max_end != data_bytes) {
        fail("file size accounting mismatch: tensors end at " + std::to_string(max_end) +
             " but data region is " + std::to_string(data_bytes) + " bytes");
    }
}

const Value* GGUFFile::kv(const std::string& key) const {
    for (const auto& p : kvs_) {
        if (p.first == key) return &p.second;
    }
    return nullptr;
}

const TensorInfo* GGUFFile::tensor(const std::string& name) const {
    for (const TensorInfo& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

}  // namespace ri
