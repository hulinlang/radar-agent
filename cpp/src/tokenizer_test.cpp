// tokenizer-test: B2 acceptance harness.
// usage: tokenizer-test <model.gguf> <golden.tsv>
// golden line format: hex(utf8 bytes) \t id,id,id
// Checks encode(text)==ids and decode(ids)==text for every line.

#include "gguf.h"
#include "tokenizer.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

std::string hex_decode(const std::string& h) {
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        out += static_cast<char>(nib(h[i]) * 16 + nib(h[i + 1]));
    }
    return out;
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> ids;
    size_t i = 0;
    while (i < s.size()) {
        char* end = nullptr;
        long v = std::strtol(s.c_str() + i, &end, 10);
        ids.push_back(static_cast<int32_t>(v));
        i = static_cast<size_t>(end - s.c_str());
        if (i < s.size() && s[i] == ',') ++i;
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tokenizer-test <model.gguf> <golden.tsv>\n");
        return 2;
    }
    try {
        ri::GGUFFile f = ri::GGUFFile::open(argv[1]);
        ri::Tokenizer tok = ri::Tokenizer::from_gguf(f);
        std::fprintf(stderr, "tokenizer loaded: vocab=%zu bos=%d eos=%d pad=%d\n",
                     tok.vocab_size(), tok.bos(), tok.eos(), tok.pad());

        FILE* fp = std::fopen(argv[2], "rb");
        if (!fp) {
            std::fprintf(stderr, "cannot open golden file %s\n", argv[2]);
            return 2;
        }
        int total = 0, enc_ok = 0, dec_ok = 0;
        std::string line;
        char buf[65536];
        while (std::fgets(buf, sizeof(buf), fp)) {
            line.assign(buf);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            const std::string text = hex_decode(line.substr(0, tab));
            const std::vector<int32_t> want = parse_ids(line.substr(tab + 1));
            ++total;

            const std::vector<int32_t> got = tok.encode(text);
            if (got == want) {
                ++enc_ok;
            } else {
                std::fprintf(stderr, "ENC MISMATCH line %d (text bytes=%zu): got %zu ids, want %zu\n",
                             total, text.size(), got.size(), want.size());
                const size_t m = got.size() < want.size() ? got.size() : want.size();
                for (size_t k = 0; k < m; ++k) {
                    if (got[k] != want[k]) {
                        std::fprintf(stderr, "  first diff at %zu: got %d want %d\n", k, got[k], want[k]);
                        break;
                    }
                }
            }
            if (tok.decode(want) == text) {
                ++dec_ok;
            } else {
                std::fprintf(stderr, "DEC MISMATCH line %d\n", total);
            }
        }
        std::fclose(fp);
        std::printf("B2 TOK: encode %d/%d ok, decode %d/%d ok\n", enc_ok, total, dec_ok, total);
        return (enc_ok == total && dec_ok == total) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tokenizer-test: error: %s\n", e.what());
        return 1;
    }
}
