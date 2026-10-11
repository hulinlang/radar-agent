#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ri {

class GGUFFile;

// Byte-level BPE tokenizer (gpt2 family, "qwen2" pre-tokenizer rules).
// All vocabulary data is read from the GGUF file -- nothing Qwen-specific
// is hard-coded. See docs/22 for the algorithm and design trade-offs.
class Tokenizer {
public:
    // Throws std::runtime_error on missing/unsupported tokenizer metadata.
    static Tokenizer from_gguf(const GGUFFile& f);

    std::vector<int32_t> encode(const std::string& utf8) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    // Debug/inspection: pre-tokenizer segments (original text slices).
    std::vector<std::string> pre_tokenize(const std::string& utf8) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t pad() const { return pad_; }
    bool is_special(int32_t id) const;
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    void bpe(const std::string& mapped, std::vector<int32_t>& out) const;

    std::vector<std::string> id_to_token_;      // mapped-unicode token strings
    std::vector<uint8_t> id_is_special_;        // token_type 3 (control) / 4 (user-defined)
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "A B" -> rank (lower = earlier)
    std::vector<std::pair<std::string, int32_t>> specials_;  // sorted by length desc
    int32_t bos_ = -1, eos_ = -1, pad_ = -1;
};

}  // namespace ri
