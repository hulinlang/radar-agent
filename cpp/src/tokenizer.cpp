#include "tokenizer.h"

#include "gguf.h"
#include "unicode_tables.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ri {
namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("tokenizer: " + msg);
}

std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the other 68
// bytes map to U+0100+n. Token strings in the GGUF vocab live in this
// mapped space, so encode must map input bytes before any vocab lookup.
struct ByteMap {
    std::string fwd[256];
    uint8_t rev[256];  // rev[mapped_cp - 0x100 for non-printable]; see rev_of()
    std::unordered_map<uint32_t, uint8_t> rev_map;

    ByteMap() {
        bool printable[256] = {};
        for (int b = '!'; b <= '~'; ++b) printable[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = printable[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
            fwd[b] = utf8_encode(cp);
            rev_map[cp] = static_cast<uint8_t>(b);
        }
    }
};

const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

// Decodes UTF-8 into codepoints plus the byte offset where each starts.
// Throws on malformed input (engine input is expected to be valid UTF-8).
std::vector<std::pair<uint32_t, size_t>> decode_utf8(const std::string& s, size_t* total) {
    std::vector<std::pair<uint32_t, size_t>> cps;
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<uint8_t>(s[i]);
        uint32_t cp;
        size_t len;
        if (b0 < 0x80) { cp = b0; len = 1; }
        else if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; len = 2; }
        else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; len = 3; }
        else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; len = 4; }
        else fail("malformed UTF-8 lead byte");
        if (i + len > s.size()) fail("truncated UTF-8 sequence");
        for (size_t k = 1; k < len; ++k) {
            const auto bk = static_cast<uint8_t>(s[i + k]);
            if ((bk & 0xC0) != 0x80) fail("malformed UTF-8 continuation byte");
            cp = (cp << 6) | (bk & 0x3F);
        }
        cps.emplace_back(cp, i);
        i += len;
    }
    *total = s.size();
    return cps;
}

std::vector<std::string> split_utf8_chars(const std::string& s) {
    size_t total;
    const auto cps = decode_utf8(s, &total);
    std::vector<std::string> out;
    for (size_t i = 0; i < cps.size(); ++i) {
        const size_t begin = cps[i].second;
        const size_t end = (i + 1 < cps.size()) ? cps[i + 1].second : total;
        out.push_back(s.substr(begin, end - begin));
    }
    return out;
}

bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// The seven qwen2 pre-tokenizer rules, applied in regex alternation order
// (first matching rule wins; each rule is greedy within itself).
// Returns the end index (exclusive) of the segment starting at cps[i].
size_t match_pretoken(const std::vector<std::pair<uint32_t, size_t>>& cps, size_t i) {
    const size_t n = cps.size();
    const uint32_t c0 = cps[i].first;

    // Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c0 == '\'') {
        auto letter = [&](size_t k) -> uint32_t {
            if (k >= n) return 0;
            uint32_t c = cps[k].first;
            return (c >= 'A' && c <= 'Z') ? c + 32 : c;
        };
        const uint32_t a = letter(i + 1), b = letter(i + 2);
        if (a == 's' || a == 't' || a == 'm') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }

    // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
    if (unicode::is_letter(c0)) {
        size_t j = i + 1;
        while (j < n && unicode::is_letter(cps[j].first)) ++j;
        return j;
    }
    if (!is_newline(c0) && !unicode::is_number(c0) && i + 1 < n &&
        unicode::is_letter(cps[i + 1].first)) {
        size_t j = i + 2;
        while (j < n && unicode::is_letter(cps[j].first)) ++j;
        return j;
    }

    // Rule 3: \p{N}{1,3}
    if (unicode::is_number(c0)) {
        size_t j = i;
        while (j < n && unicode::is_number(cps[j].first) && j - i < 3) ++j;
        return j;
    }

    // Rule 4: \ ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        size_t k = i;
        if (c0 == ' ') ++k;  // optional single ASCII space
        if (k < n) {
            const uint32_t ck = cps[k].first;
            if (!unicode::is_space(ck) && !unicode::is_letter(ck) && !unicode::is_number(ck)) {
                size_t j = k + 1;
                while (j < n) {
                    const uint32_t cj = cps[j].first;
                    if (unicode::is_space(cj) || unicode::is_letter(cj) || unicode::is_number(cj)) break;
                    ++j;
                }
                while (j < n && is_newline(cps[j].first)) ++j;
                return j;
            }
        }
    }

    // Rule 5: \s*[\r\n]+  (whitespace run, match ends at its last newline)
    if (unicode::is_space(c0)) {
        size_t j = i;
        while (j < n && unicode::is_space(cps[j].first)) ++j;
        size_t last_nl = n;
        for (size_t k = i; k < j; ++k) {
            if (is_newline(cps[k].first)) last_nl = k;
        }
        if (last_nl != n) return last_nl + 1;

        // Rule 6: \s+(?!\S)  -- longest prefix of the run followed by
        // whitespace-or-end; i.e. the whole run if it reaches the end,
        // otherwise the run minus its last character.
        if (j == n) return j;
        if (j - i >= 2) return j - 1;

        // Rule 7: \s+
        return j;
    }

    // Nothing matched (should not happen: any char is covered by rule 2/4/5-7
    // or rule 3). Emit one char to guarantee progress.
    return i + 1;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const GGUFFile& f) {
    const Value* pre = f.kv("tokenizer.ggml.pre");
    if (!pre || pre->type != ValueType::STRING) fail("missing tokenizer.ggml.pre");
    if (std::get<std::string>(pre->data) != "qwen2") {
        fail("unsupported pre-tokenizer '" + std::get<std::string>(pre->data) + "'");
    }

    const Value* tokens = f.kv("tokenizer.ggml.tokens");
    const Value* merges = f.kv("tokenizer.ggml.merges");
    const Value* types = f.kv("tokenizer.ggml.token_type");
    if (!tokens || tokens->type != ValueType::ARRAY) fail("missing tokenizer.ggml.tokens");
    if (!merges || merges->type != ValueType::ARRAY) fail("missing tokenizer.ggml.merges");
    if (!types || types->type != ValueType::ARRAY) fail("missing tokenizer.ggml.token_type");

    const Array& tok_arr = std::get<Array>(tokens->data);
    const Array& merge_arr = std::get<Array>(merges->data);
    const Array& type_arr = std::get<Array>(types->data);
    if (type_arr.size() != tok_arr.size()) fail("token_type length != tokens length");

    Tokenizer t;
    t.id_to_token_.reserve(tok_arr.size());
    t.id_is_special_.reserve(tok_arr.size());
    for (size_t i = 0; i < tok_arr.size(); ++i) {
        t.id_to_token_.push_back(std::get<std::string>(tok_arr[i].data));
        const int64_t ty = std::get<int64_t>(type_arr[i].data);
        t.id_is_special_.push_back(ty == 3 || ty == 4 ? 1 : 0);
    }
    t.token_to_id_.reserve(tok_arr.size() * 2);
    for (size_t i = 0; i < t.id_to_token_.size(); ++i) {
        t.token_to_id_.emplace(t.id_to_token_[i], static_cast<int32_t>(i));
    }
    t.merge_rank_.reserve(merge_arr.size() * 2);
    for (size_t i = 0; i < merge_arr.size(); ++i) {
        t.merge_rank_.emplace(std::get<std::string>(merge_arr[i].data), static_cast<int32_t>(i));
    }
    for (size_t i = 0; i < t.id_to_token_.size(); ++i) {
        if (t.id_is_special_[i]) t.specials_.emplace_back(t.id_to_token_[i], static_cast<int32_t>(i));
    }
    std::sort(t.specials_.begin(), t.specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    auto get_id = [&](const char* key) -> int32_t {
        const Value* v = f.kv(key);
        if (!v) fail(std::string("missing ") + key);
        if (v->type == ValueType::UINT32 || v->type == ValueType::UINT64) {
            return static_cast<int32_t>(std::get<uint64_t>(v->data));
        }
        return static_cast<int32_t>(std::get<int64_t>(v->data));
    };
    t.bos_ = get_id("tokenizer.ggml.bos_token_id");
    t.eos_ = get_id("tokenizer.ggml.eos_token_id");
    t.pad_ = get_id("tokenizer.ggml.padding_token_id");
    return t;
}

bool Tokenizer::is_special(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < id_is_special_.size() && id_is_special_[id];
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& utf8) const {
    size_t total;
    const auto cps = decode_utf8(utf8, &total);
    std::vector<std::string> out;
    size_t i = 0;
    while (i < cps.size()) {
        const size_t j = match_pretoken(cps, i);
        const size_t begin = cps[i].second;
        const size_t end = (j < cps.size()) ? cps[j].second : total;
        out.push_back(utf8.substr(begin, end - begin));
        i = j;
    }
    return out;
}

void Tokenizer::bpe(const std::string& mapped, std::vector<int32_t>& out) const {
    if (mapped.empty()) return;
    const auto whole = token_to_id_.find(mapped);
    if (whole != token_to_id_.end()) {
        out.push_back(whole->second);
        return;
    }

    std::vector<std::string> syms = split_utf8_chars(mapped);
    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        size_t best_i = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            const auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
                found = true;
            }
        }
        if (!found) break;
        const std::string left = syms[best_i], right = syms[best_i + 1];
        std::vector<std::string> next;
        next.reserve(syms.size());
        for (size_t i = 0; i < syms.size();) {
            if (i + 1 < syms.size() && syms[i] == left && syms[i + 1] == right) {
                next.push_back(left + right);
                i += 2;
            } else {
                next.push_back(syms[i]);
                ++i;
            }
        }
        syms.swap(next);
    }

    for (const std::string& s : syms) {
        const auto it = token_to_id_.find(s);
        if (it == token_to_id_.end()) fail("no vocab entry for BPE piece");
        out.push_back(it->second);
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& utf8) const {
    std::vector<int32_t> out;
    const size_t n = utf8.size();
    size_t i = 0, seg = 0;

    auto flush = [&](size_t end) {
        if (end <= seg) return;
        const std::string text = utf8.substr(seg, end - seg);
        for (const std::string& piece : pre_tokenize(text)) {
            std::string mapped;
            for (const unsigned char b : piece) mapped += byte_map().fwd[b];
            bpe(mapped, out);
        }
    };

    while (i < n) {
        int32_t hit = -1;
        size_t hlen = 0;
        for (const auto& sp : specials_) {
            if (i + sp.first.size() <= n && utf8.compare(i, sp.first.size(), sp.first) == 0) {
                hit = sp.second;
                hlen = sp.first.size();
                break;
            }
        }
        if (hit >= 0) {
            flush(i);
            out.push_back(hit);
            i += hlen;
            seg = i;
        } else {
            ++i;
        }
    }
    flush(n);
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (const int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) fail("id out of range");
        const std::string& tok = id_to_token_[id];
        if (id_is_special_[id]) {
            out += tok;
            continue;
        }
        for (const std::string& ch : split_utf8_chars(tok)) {
            size_t total;
            const auto cps = decode_utf8(ch, &total);
            const auto it = byte_map().rev_map.find(cps[0].first);
            if (it == byte_map().rev_map.end()) fail("unmappable character in token");
            out += static_cast<char>(it->second);
        }
    }
    return out;
}

}  // namespace ri
