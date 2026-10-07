// radar-serve: B9b minimal HTTP server for the radar-infer CUDA engine.
// Zero dependencies: raw Linux sockets, hand-rolled HTTP/1.1 parsing, and a
// custom plain-text payload (both ends are our code — no JSON needed).
//
//   radar-serve --model <gguf> [--port 8090] [--kv-cap 4096] [--host 127.0.0.1]
//
// Request:  POST /gen
//   MAXNEW <int>
//   [TEMP <f>] [TOPK <i>] [TOPP <f>] [SEED <u64>]
//   [BAN <id> <id> ...]
//   [RESET 1]            (drop session history before this request)
//   PROMPT
//   <raw utf-8 prompt bytes>
// Response body: "N <n_generated_tokens>\n<text>"  (generation stops at EOS
// or MAXNEW; the EOS token itself is never included).
//
// Session: the server keeps ONE conversation's KV state; a new prompt is
// LCP-matched against history at token level and only the suffix is
// prefilled (bitwise-identical to a full re-run by construction: same tokens
// at the same positions produce the same KV). Single-threaded, sequential
// requests only (ReAct is a serial chain).

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "gguf.h"
#include "gpu_model.h"
#include "model.h"  // ModelConfig (vocab)
#include "sampler.h"
#include "tokenizer.h"

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

struct Request {
    int max_new = 256;
    ri::Sampler sampler;
    std::vector<int32_t> bans;
    bool reset = false;
    std::string prompt;
};

class HttpConn {
public:
    explicit HttpConn(int fd) : fd_(fd) {}

    bool read_request(std::string& method, std::string& path, std::string& body) {
        buf_.clear();
        char tmp[16384];
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<size_t>(n));
            header_end = buf_.find("\r\n\r\n");
            if (buf_.size() > 1 << 20) fail("header too large");
        }
        const std::string head = buf_.substr(0, header_end);
        size_t content_length = 0;
        {
            size_t b = 0;
            while (b < head.size()) {
                const size_t e = head.find("\r\n", b);
                const std::string line = head.substr(b, e == std::string::npos ? e : e - b);
                if (line.rfind("Content-Length:", 0) == 0)
                    content_length = std::stoul(line.substr(15));
                if (e == std::string::npos) break;
                b = e + 2;
            }
        }
        const size_t need = header_end + 4 + content_length;
        while (buf_.size() < need) {
            const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf_.append(tmp, static_cast<size_t>(n));
        }
        body = buf_.substr(header_end + 4, content_length);
        // method + path
        const size_t sp1 = head.find(' ');
        const size_t sp2 = head.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) fail("bad request line");
        method = head.substr(0, sp1);
        path = head.substr(sp1 + 1, sp2 - sp1 - 1);
        return true;
    }

    void respond(int code, const char* status, const std::string& body) {
        std::string head = "HTTP/1.1 " + std::to_string(code) + " " + status +
                           "\r\nContent-Type: text/plain; charset=utf-8\r\n"
                           "Content-Length: " + std::to_string(body.size()) +
                           "\r\nConnection: close\r\n\r\n";
        const std::string out = head + body;
        size_t sent = 0;
        while (sent < out.size()) {
            const ssize_t n = ::send(fd_, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) return;
            sent += static_cast<size_t>(n);
        }
    }

private:
    int fd_;
    std::string buf_;
};

Request parse_gen_body(const std::string& body) {
    Request r;
    const size_t p = body.find("\nPROMPT\n");
    if (p == std::string::npos) fail("missing PROMPT marker");
    r.prompt = body.substr(p + 8);
    size_t b = 0;
    const size_t head_len = p + 1;
    while (b < head_len) {
        const size_t e = body.find('\n', b);
        const std::string line = body.substr(b, (e == std::string::npos ? head_len : e) - b);
        b = (e == std::string::npos) ? head_len : e + 1;
        if (line.empty()) continue;
        const size_t sp = line.find(' ');
        const std::string key = sp == std::string::npos ? line : line.substr(0, sp);
        const std::string val = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "MAXNEW") r.max_new = std::stoi(val);
        else if (key == "TEMP") r.sampler.temp = std::stof(val);
        else if (key == "TOPK") r.sampler.top_k = std::stoi(val);
        else if (key == "TOPP") r.sampler.top_p = std::stof(val);
        else if (key == "SEED") r.sampler.seed(std::stoull(val));
        else if (key == "RESET") r.reset = true;
        else if (key == "BAN") {
            size_t vb = 0;
            while (vb < val.size()) {
                const size_t ve = val.find(' ', vb);
                r.bans.push_back(std::stoi(val.substr(vb, ve == std::string::npos ? ve : ve - vb)));
                if (ve == std::string::npos) break;
                vb = ve + 1;
            }
        }
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, host = "127.0.0.1";
    int port = 8090;
    int kv_cap = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--model") model_path = need("--model");
        else if (a == "--port") port = std::stoi(need("--port"));
        else if (a == "--host") host = need("--host");
        else if (a == "--kv-cap") kv_cap = std::stoi(need("--kv-cap"));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty()) fail("--model required");

    try {
        ri::GGUFFile f = ri::GGUFFile::open(model_path);
        ri::Tokenizer tok = ri::Tokenizer::from_gguf(f);
        ri::GpuModel gpu = ri::GpuModel::load(f, kv_cap, 1);
        std::vector<float> logits(gpu.config().vocab), scratch;
        std::fprintf(stderr, "radar-serve: model loaded (vocab=%d, kv_cap=%d), listening on %s:%d\n",
                     gpu.config().vocab, kv_cap, host.c_str(), port);

        ::signal(SIGPIPE, SIG_IGN);
        const int srv = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) fail("socket() failed");
        int one = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) fail("bad --host");
        if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
            fail(std::string("bind() failed: ") + std::strerror(errno));
        if (::listen(srv, 16) != 0) fail("listen() failed");

        // session state (single conversation, LCP reuse)
        std::vector<int32_t> history;
        int pos = 0;

        while (true) {
            const int fd = ::accept(srv, nullptr, nullptr);
            if (fd < 0) continue;
            HttpConn conn(fd);
            std::string method, path, body;
            try {
                if (!conn.read_request(method, path, body)) { ::close(fd); continue; }
                if (method == "GET" && path == "/health") {
                    conn.respond(200, "OK", "ok\n");
                } else if (method == "POST" && path == "/gen") {
                    const Request req = parse_gen_body(body);
                    if (req.reset) {
                        history.clear();
                        pos = 0;
                    }
                    const std::vector<int32_t> ids = tok.encode(req.prompt);
                    if (ids.empty()) fail("prompt encodes to zero tokens");
                    if (static_cast<int>(ids.size()) + req.max_new + 8 > kv_cap)
                        fail("prompt + max_new exceeds kv_cap");

                    // LCP prefix reuse against session history. The clamp
                    // guarantees pos < ids.size(), so a suffix prefill always
                    // runs and yields fresh logits for the first prediction.
                    size_t lcp = 0;
                    while (lcp < ids.size() && lcp < history.size() && ids[lcp] == history[lcp])
                        ++lcp;
                    if (lcp >= ids.size()) lcp = ids.size() - 1;
                    if (static_cast<int>(lcp) < pos) pos = static_cast<int>(lcp);
                    const size_t reused = static_cast<size_t>(pos);

                    ri::Sampler sampler = req.sampler;  // per-request RNG state
                    gpu.prefill_batch(ids.data() + pos, static_cast<int>(ids.size()) - pos,
                                      logits.data(), 0, pos);
                    pos = static_cast<int>(ids.size());
                    int32_t next = ri::sample_from(sampler, req.bans,
                                                   logits.data(), logits.size(), scratch);

                    std::vector<int32_t> gen;
                    while (static_cast<int>(gen.size()) < req.max_new && next != tok.eos()) {
                        gen.push_back(next);
                        gpu.forward_logits(next, pos++, logits.data());
                        next = ri::sample_from(sampler, req.bans, logits.data(), logits.size(), scratch);
                    }
                    const bool stopped_by_eos = (next == tok.eos());
                    history = ids;
                    history.insert(history.end(), gen.begin(), gen.end());
                    pos = static_cast<int>(history.size());

                    const std::string out = "N " + std::to_string(gen.size()) + "\n" +
                                            "REUSED " + std::to_string(reused) + "\n" +
                                            tok.decode(gen);
                    std::fprintf(stderr, "[gen] prompt=%zu tok reused=%zu gen=%zu tok eos=%d\n",
                                 ids.size(), reused, gen.size(),
                                 static_cast<int>(stopped_by_eos));
                    conn.respond(200, "OK", out);
                } else {
                    conn.respond(404, "Not Found", "unknown route\n");
                }
            } catch (const std::exception& e) {
                conn.respond(400, "Bad Request", std::string("error: ") + e.what() + "\n");
            }
            ::close(fd);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "radar-serve: error: %s\n", e.what());
        return 1;
    }
}
