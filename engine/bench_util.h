// bench_util.h — shared, CUDA-free helpers for the GuardCAN engine and benchmarks:
// CLI parsing, wall-clock timing, percentile statistics, the replayed input ring, a tiny JSON writer.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ae_model.h"

namespace bench {

// ------------------------------------------------------------------ CLI
struct Args {
    int argc; char** argv;
    bool has(const std::string& k) const {
        for (int i = 1; i < argc; ++i) if (k == argv[i]) return true;
        return false;
    }
    std::string str(const std::string& k, const std::string& def) const {
        for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return argv[i + 1];
        return def;
    }
    long num(const std::string& k, long def) const { return std::stol(str(k, std::to_string(def))); }
    std::vector<int> list(const std::string& k, const std::string& def) const {
        std::vector<int> v; std::stringstream ss(str(k, def)); std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
        return v;
    }
};

// ------------------------------------------------------------------ timing
inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------ statistics
// Nearest-rank percentiles: p-th percentile = the ceil(p/100 * n)-th smallest sample.
struct Pctl { size_t n = 0; double mean = 0, min = 0, p50 = 0, p90 = 0, p99 = 0, p999 = 0, max = 0; };

inline Pctl percentiles(std::vector<double> v) {
    Pctl r;
    if (v.empty()) return r;
    std::sort(v.begin(), v.end());
    auto rank = [&](double p) {
        size_t k = (size_t)std::ceil(p / 100.0 * (double)v.size());
        return v[std::min(v.size() - 1, k == 0 ? 0 : k - 1)];
    };
    double s = 0; for (double x : v) s += x;
    r.n = v.size(); r.mean = s / v.size(); r.min = v.front(); r.max = v.back();
    r.p50 = rank(50); r.p90 = rank(90); r.p99 = rank(99); r.p999 = rank(99.9);
    return r;
}

// ------------------------------------------------------------------ JSON (just enough to write results)
inline std::string jnum(double x) {
    if (!std::isfinite(x)) return "null";
    char b[64];
    if (x == std::floor(x) && std::fabs(x) < 1e15) std::snprintf(b, sizeof b, "%.0f", x);   // counts stay exact
    else std::snprintf(b, sizeof b, "%.6g", x);                                           // measurements: 6 sig. figs
    return b;
}
inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}
struct JObj {
    std::vector<std::pair<std::string, std::string>> kv;
    JObj& raw(const std::string& k, const std::string& v) { kv.emplace_back(k, v); return *this; }
    JObj& num(const std::string& k, double v) { return raw(k, jnum(v)); }
    JObj& str(const std::string& k, const std::string& v) { return raw(k, jstr(v)); }
    JObj& boolean(const std::string& k, bool v) { return raw(k, v ? "true" : "false"); }
    std::string dump(int ind = 0) const {
        std::string pad(ind + 2, ' '), o = "{\n";
        for (size_t i = 0; i < kv.size(); ++i)
            o += pad + jstr(kv[i].first) + ": " + kv[i].second + (i + 1 < kv.size() ? ",\n" : "\n");
        return o + std::string(ind, ' ') + "}";
    }
};
inline std::string jarr(const std::vector<std::string>& items, int ind = 0) {
    std::string pad(ind + 2, ' '), o = "[\n";
    for (size_t i = 0; i < items.size(); ++i) o += pad + items[i] + (i + 1 < items.size() ? ",\n" : "\n");
    return o + std::string(ind, ' ') + "]";
}
inline std::string pctl_json(const Pctl& p, int ind) {
    return JObj().num("n", (double)p.n).num("mean", p.mean).num("min", p.min).num("p50", p.p50)
        .num("p90", p.p90).num("p99", p.p99).num("p99_9", p.p999).num("max", p.max).dump(ind);
}
inline void write_file(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    f << s << "\n";
    std::printf("wrote %s\n", path.c_str());
}

// ------------------------------------------------------------------ platform
inline bool is_wsl() {
    std::ifstream f("/proc/version"); std::string s; std::getline(f, s);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s.find("microsoft") != std::string::npos;
}
inline std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo"); std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

// ------------------------------------------------------------------ input stream
// The benchmark stream replays the reference feature vectors (held-out test split) cyclically into a
// ring of `frames` frames. The MLP has no data-dependent control flow, so timing does not depend on
// which frames are replayed — but every score is still a real forward pass on real feature vectors.
inline void fill_ring(float* dst, size_t frames, const ae::Reference& ref) {
    for (size_t i = 0; i < frames; ++i)
        std::memcpy(dst + i * ae::D_IN, ref.X.data() + (i % ref.n) * ae::D_IN, sizeof(float) * ae::D_IN);
}

// Compare engine scores against the Python reference. Tolerance: |a-b| <= atol + rtol*|b|.
struct Check { double max_abs = 0, max_rel = 0; size_t bad = 0, flips = 0, n = 0; bool pass = true; };

inline Check compare(const float* got, const float* want, size_t n, float threshold,
                     double atol = 1e-6, double rtol = 1e-4) {
    Check c; c.n = n;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)got[i] - (double)want[i]);
        c.max_abs = std::max(c.max_abs, d);
        if (want[i] != 0.f) c.max_rel = std::max(c.max_rel, d / std::fabs((double)want[i]));
        if (d > atol + rtol * std::fabs((double)want[i])) ++c.bad;
        if ((got[i] > threshold) != (want[i] > threshold)) ++c.flips;
    }
    c.pass = (c.bad == 0);
    return c;
}
inline std::string check_json(const Check& c, int ind) {
    return JObj().num("frames", (double)c.n).num("max_abs_diff", c.max_abs).num("max_rel_diff", c.max_rel)
        .num("out_of_tolerance", (double)c.bad).num("decision_flips", (double)c.flips)
        .boolean("pass", c.pass).dump(ind);
}

}  // namespace bench
