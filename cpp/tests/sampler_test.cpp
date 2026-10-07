// sampler-test: B9a unit checks for the host sampler (no model needed).
//   1. greedy = first strict max (tie -> lower index)
//   2. ban list: banned ids never selected; banned top -> second best
//   3. top_k=1 == greedy
//   4. seed reproducibility (same seed -> same 1000-draw sequence)
//   5. distribution sanity: biased logits, temp=1, 40k draws -> order and
//      rough proportions match softmax expectation

#include "../src/sampler.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int fails = 0;
void check(bool ok, const char* name) {
    std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
}

}  // namespace

int main() {
    using ri::Sampler;

    // 1. greedy first-max
    {
        std::vector<float> logits = {1.f, 5.f, 5.f, 2.f};
        Sampler s;
        check(s.sample(logits) == 1, "greedy first-max (tie -> lower index)");
    }

    // 2. ban list
    {
        std::vector<float> logits = {1.f, 5.f, 3.f};
        const int32_t ban[] = {1};
        Sampler s;
        Sampler::apply_bans(logits, ban, 1);
        check(s.sample(logits) == 2, "banned top -> second best");
        bool ban_ok = true;
        for (int i = 0; i < 50; ++i) {
            std::vector<float> l2 = {1.f, 5.f, 3.f};
            Sampler::apply_bans(l2, ban, 1);
            Sampler t;
            t.temp = 1.0f;
            if (t.sample(l2) == 1) ban_ok = false;
        }
        check(ban_ok, "ban holds under sampling");
    }

    // 3. top_k=1 == greedy even with temperature
    {
        std::vector<float> logits = {0.f, 2.f, 1.f, 3.f};
        Sampler s;
        s.temp = 1.0f;
        s.top_k = 1;
        check(s.sample(logits) == 3, "top_k=1 == greedy");
    }

    // 4. seed reproducibility
    {
        std::vector<float> base(100);
        for (int i = 0; i < 100; ++i) base[i] = std::sin(i * 0.3f);
        Sampler a, b;
        a.temp = b.temp = 0.9f;
        a.seed(42);
        b.seed(42);
        bool same = true;
        for (int i = 0; i < 1000; ++i) {
            std::vector<float> la = base, lb = base;
            if (a.sample(la) != b.sample(lb)) { same = false; break; }
        }
        check(same, "same seed -> same 1000 draws");
        Sampler c;
        c.temp = 0.9f;
        c.seed(43);
        bool diff = false;
        Sampler a2;
        a2.temp = 0.9f;
        a2.seed(42);
        for (int i = 0; i < 100; ++i) {
            std::vector<float> la = base, lc = base;
            if (a2.sample(la) != c.sample(lc)) { diff = true; break; }
        }
        check(diff, "different seed -> different draws");
    }

    // 5. distribution sanity: logits [2, 1, 0] -> softmax probs ~ [0.665, 0.245, 0.090]
    {
        const std::vector<float> base = {2.f, 1.f, 0.f};
        Sampler s;
        s.temp = 1.0f;
        s.seed(7);
        int cnt[3] = {0, 0, 0};
        const int N = 40000;
        for (int i = 0; i < N; ++i) {
            std::vector<float> l = base;
            ++cnt[s.sample(l)];
        }
        const double e0 = std::exp(2.0) / (std::exp(2.0) + std::exp(1.0) + 1.0);
        const double f0 = static_cast<double>(cnt[0]) / N;
        const bool order = cnt[0] > cnt[1] && cnt[1] > cnt[2];
        const bool close = std::fabs(f0 - e0) < 0.02;
        std::printf("  dist: [%d %d %d] -> %.3f %.3f %.3f (expect %.3f ...)\n",
                    cnt[0], cnt[1], cnt[2], f0, static_cast<double>(cnt[1]) / N,
                    static_cast<double>(cnt[2]) / N, e0);
        check(order && close, "sampling distribution ~ softmax");
    }

    std::printf("SAMPLER: %s\n", fails == 0 ? "all pass" : "FAILURES");
    return fails == 0 ? 0 : 1;
}
