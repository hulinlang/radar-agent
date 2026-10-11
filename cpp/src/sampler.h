#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace ri {

// Host-side sampler (B9a): ban list -> temperature -> top_k -> top_p -> RNG
// draw. temp=0 is greedy with first-index tie-break (identical to the plain
// argmax loop). xorshift64* with a fixed default seed keeps runs reproducible.
struct Sampler {
    float temp = 0.0f;    // <=0 means greedy
    int top_k = 0;        // 0 means off
    float top_p = 1.0f;   // >=1 means off
    uint64_t rng = 0x9E3779B97F4A7C15ull;

    void seed(uint64_t s) { rng = s ? s : 0x9E3779B97F4A7C15ull; }

    static void apply_bans(std::vector<float>& logits, const int32_t* ban_ids, int n_ban) {
        for (int i = 0; i < n_ban; ++i) {
            const int32_t id = ban_ids[i];
            if (id >= 0 && static_cast<size_t>(id) < logits.size())
                logits[id] = -std::numeric_limits<float>::infinity();
        }
    }

    uint32_t next_u32() {
        uint64_t x = rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        rng = x;
        return static_cast<uint32_t>((x * 0x2545F4914F6CDD1Dull) >> 32);
    }

    float next_f32() { return static_cast<float>(next_u32()) * (1.0f / 4294967296.0f); }

    // Greedy: first strict max (same rule as the host argmax loop and
    // argmax_rows kernel).
    static int32_t argmax(const std::vector<float>& v) {
        int32_t best = 0;
        for (size_t i = 1; i < v.size(); ++i)
            if (v[i] > v[best]) best = static_cast<int32_t>(i);
        return best;
    }

    // Samples from logits (modified in place). Returns the token id.
    int32_t sample(std::vector<float>& logits) {
        if (temp <= 0.0f) return argmax(logits);

        // temperature + softmax
        const float inv_t = 1.0f / temp;
        float mx = -std::numeric_limits<float>::infinity();
        for (float v : logits) mx = std::max(mx, v);
        double sum = 0.0;
        for (float& v : logits) {
            v = (v == -std::numeric_limits<float>::infinity())
                    ? 0.0f
                    : std::exp((v - mx) * inv_t);
            sum += v;
        }
        if (sum <= 0.0) return argmax(logits);  // everything banned -> fall back
        const float inv_s = static_cast<float>(1.0 / sum);
        for (float& v : logits) v *= inv_s;

        // top_k / top_p over the descending order
        std::vector<int32_t> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        size_t keep = idx.size();
        if (top_k > 0 && static_cast<size_t>(top_k) < keep) keep = top_k;
        if (top_p < 1.0f) {
            double acc = 0.0;
            for (size_t i = 0; i < keep; ++i) {
                acc += logits[idx[i]];
                if (acc >= top_p) { keep = i + 1; break; }
            }
        }
        // draw within [0, keep)
        double r = next_f32();
        double acc = 0.0, kept_sum = 0.0;
        for (size_t i = 0; i < keep; ++i) kept_sum += logits[idx[i]];
        if (kept_sum <= 0.0) return idx[0];
        for (size_t i = 0; i < keep; ++i) {
            acc += logits[idx[i]] / kept_sum;
            if (r <= acc) return idx[i];
        }
        return idx[keep - 1];
    }
};

// B9a: ban list -> sampler (greedy when temp=0); destructive on `scratch`.
inline int32_t sample_from(ri::Sampler& sampler, const std::vector<int32_t>& bans,
                           const float* logits, size_t n, std::vector<float>& scratch) {
    scratch.assign(logits, logits + n);
    if (!bans.empty())
        ri::Sampler::apply_bans(scratch, bans.data(), static_cast<int>(bans.size()));
    return sampler.sample(scratch);
}

}  // namespace ri
