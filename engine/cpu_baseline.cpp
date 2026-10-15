// cpu_baseline.cpp — single-thread CPU baseline for the speedup figure.
//
// Two single-thread implementations of the same forward pass, both compiled with -O3 -march=native:
//   scalar  frame at a time through ae::score() — literally the code the GPU kernel runs. The compiler
//           cannot vectorize its dot-product reductions without reassociating floating point, so this
//           is mostly scalar code. Reported for reference only.
//   simd    8 frames at a time with frames in the SIMD lanes (GCC/Clang vector extensions, no
//           intrinsics): each weight is broadcast and multiply-added into 8 independent frames, so it
//           vectorizes with NO reassociation (same per-frame summation order as ae::score). This is the
//           baseline the speedup figure uses — a fair, optimized single core, not a strawman.
// The CPU needs no staging copy: it scores frames straight out of the source ring and counts frames
// over threshold (the same consume step as the GPU benchmarks).
//
//   1. verify      both implementations vs the Python reference set; same tolerance as the GPU engine.
//   2. throughput  frames/s over a 1M-frame ring (88 MB, DRAM-resident), median of --reps runs.
//   3. latency     per-batch wall time (simd path) for the same batch sizes as the GPU latency benchmark.
//
//   cpu_baseline --weights models/ae_weights.npz --ref models/ae_reference.npz --json out.json
#include <cstdio>
#include <string>
#include <vector>

#include "bench_util.h"

#ifndef CPU_FLAGS
#define CPU_FLAGS "unknown"
#endif

using bench::JObj;

constexpr int TILE = 8;
// 8 frames, one per lane of a 256-bit vector (GCC/Clang vector extension, no intrinsics). Measured on
// the i7-14650HX: 8 lanes beat 16/32 (which GCC splits poorly on AVX2), and blocking OB outputs per
// pass gives OB independent FMA chains to hide FMA latency (OB=8 hidden, 11 for the 22-wide output).
typedef float vf __attribute__((vector_size(TILE * sizeof(float))));

template <int IN, int OUT, bool RELU, int OB>
static inline void layer(const float (&W)[OUT][IN], const float (&b)[OUT], const vf (&x)[IN], vf (&y)[OUT]) {
    static_assert(OUT % OB == 0, "output blocking must divide the layer width");
    const vf zero = {};
    for (int o = 0; o < OUT; o += OB) {
        vf acc[OB];
        for (int q = 0; q < OB; ++q) acc[q] = zero + b[o + q];
        for (int j = 0; j < IN; ++j) {                             // same summation order as ae::score
            const vf xj = x[j];
            for (int q = 0; q < OB; ++q) acc[q] += W[o + q][j] * xj;
        }
        for (int q = 0; q < OB; ++q) y[o + q] = RELU ? (acc[q] > zero ? acc[q] : zero) : acc[q];
    }
}

// Score TILE consecutive frames; frames live in the SIMD lanes.
static void score_tile(const ae::Model& m, const float* x, float* out) {
    vf xs[ae::D_IN];
    for (int j = 0; j < ae::D_IN; ++j)
        for (int f = 0; f < TILE; ++f) {
            const float v = x[f * ae::D_IN + j];
            xs[j][f] = m.scaled[j] ? (v - m.mean[j]) / m.std_[j] : v;
        }
    vf h1[ae::H1], h2[ae::H2], h3[ae::H1], y[ae::D_IN];
    layer<ae::D_IN, ae::H1, true, 8>(m.W0, m.b0, xs, h1);
    layer<ae::H1, ae::H2, true, 8>(m.W1, m.b1, h1, h2);
    layer<ae::H2, ae::H1, true, 8>(m.W2, m.b2, h2, h3);
    layer<ae::H1, ae::D_IN, false, 11>(m.W3, m.b3, h3, y);
    vf sse = {};
    for (int o = 0; o < ae::D_IN; ++o) { const vf d = y[o] - xs[o]; sse += d * d; }
    for (int f = 0; f < TILE; ++f) out[f] = sse[f] / (float)ae::D_IN;
}

static int score_scalar(const ae::Model& m, const float* x, float* out, int n) {
    int over = 0;
    for (int i = 0; i < n; ++i) {
        out[i] = ae::score(m, x + (size_t)i * ae::D_IN);
        over += out[i] > m.threshold;
    }
    return over;
}

static int score_simd(const ae::Model& m, const float* x, float* out, int n) {
    int i = 0, over = 0;
    for (; i + TILE <= n; i += TILE) score_tile(m, x + (size_t)i * ae::D_IN, out + i);
    for (; i < n; ++i) out[i] = ae::score(m, x + (size_t)i * ae::D_IN);      // tail (and B=1)
    for (int k = 0; k < n; ++k) over += out[k] > m.threshold;
    return over;
}

using ScoreFn = int (*)(const ae::Model&, const float*, float*, int);

int main(int argc, char** argv) {
    bench::Args a{argc, argv};
    const std::string weights = a.str("--weights", "../models/ae_weights.npz");
    const std::string refpath = a.str("--ref", "../models/ae_reference.npz");
    const std::string json_out = a.str("--json", "");
    const std::vector<int> lat_batch = a.list("--lat-batch", "1,64,1024,16384");
    const long iters = a.num("--iters", 20000);
    const long tp_frames = a.num("--tp-frames", 1L << 25);
    const int reps = (int)a.num("--reps", 3);
    const size_t ring_frames = (size_t)a.num("--ring", 1 << 20);

    const ae::Model model = ae::load_model(weights);
    const ae::Reference ref = ae::load_reference(refpath);
    volatile int sink = 0;
    std::printf("CPU baseline (1 thread) | %s | %s\n", bench::cpu_model().c_str(), CPU_FLAGS);

    std::vector<float> ring(ring_frames * ae::D_IN), out(ring_frames);
    bench::fill_ring(ring.data(), ring_frames, ref);

    struct Impl { const char* name; ScoreFn fn; bench::Check chk; std::vector<double> fps; double med = 0; };
    std::vector<Impl> impls = {{"scalar", score_scalar, {}, {}}, {"simd", score_simd, {}, {}}};
    bool ok = true;
    for (Impl& im : impls) {
        // 1. verify
        std::vector<float> got(ref.n);
        im.fn(model, ref.X.data(), got.data(), (int)ref.n);
        im.chk = bench::compare(got.data(), ref.err.data(), ref.n, model.threshold);
        std::printf("verify %-6s: %zu frames vs Python  max|diff|=%.3e  max rel=%.3e  out-of-tol=%zu  flips=%zu -> %s\n",
                    im.name, im.chk.n, im.chk.max_abs, im.chk.max_rel, im.chk.bad, im.chk.flips, im.chk.pass ? "PASS" : "FAIL");
        ok = ok && im.chk.pass;

        // 2. throughput
        const int chunk = 4096;
        auto pass = [&](long frames) {
            const double t0 = bench::now_us();
            for (long done = 0; done < frames; done += chunk) {
                const size_t off = (size_t)done % ring_frames;
                sink += im.fn(model, ring.data() + off * ae::D_IN, out.data() + off, chunk);
            }
            return frames / ((bench::now_us() - t0) / 1e6);
        };
        pass(1 << 20);                                                 // warm-up
        for (int r = 0; r < reps; ++r) im.fps.push_back(pass(tp_frames));
        std::vector<double> s = im.fps; std::sort(s.begin(), s.end());
        im.med = s[s.size() / 2];
        std::printf("throughput %-6s: median %.4g frames/s (min %.4g, max %.4g, %d reps x %ld frames) = %.1f ns/frame\n",
                    im.name, im.med, s.front(), s.back(), reps, tp_frames, 1e9 / im.med);
    }
    if (!ok) return 2;

    // 3. latency per batch (simd path — the faster one)
    std::vector<std::string> lat_rows;
    for (int B : lat_batch) {
        const size_t slots = ring_frames / B;
        std::vector<double> lat; lat.reserve(iters);
        for (long it = -200; it < iters; ++it) {
            const size_t off = ((it + 200) % slots) * B;
            const double t0 = bench::now_us();
            sink += score_simd(model, ring.data() + off * ae::D_IN, out.data() + off, B);
            const double t1 = bench::now_us();
            if (it >= 0) lat.push_back(t1 - t0);
        }
        const bench::Pctl p = bench::percentiles(lat);
        std::printf("latency simd B=%-6d p50 %9.2f  p99 %9.2f  p99.9 %9.2f  max %9.2f us\n", B, p.p50, p.p99, p.p999, p.max);
        lat_rows.push_back(JObj().num("batch", B).num("iters", (double)iters).raw("e2e_us", bench::pctl_json(p, 6)).dump(4));
    }

    if (!json_out.empty()) {
        std::vector<std::string> tp;
        for (const Impl& im : impls) {
            std::vector<std::string> r; for (double f : im.fps) r.push_back(bench::jnum(f));
            std::vector<double> s = im.fps; std::sort(s.begin(), s.end());
            tp.push_back(JObj().str("impl", im.name).num("frames_per_rep", (double)tp_frames).num("reps", reps)
                .num("frames_per_s_median", im.med).num("frames_per_s_min", s.front()).num("frames_per_s_max", s.back())
                .raw("frames_per_s_reps", bench::jarr(r, 6)).num("ns_per_frame_median", 1e9 / im.med)
                .raw("verify", bench::check_json(im.chk, 6)).dump(4));
        }
        JObj root;
        root.str("benchmark", "cpu_baseline").str("host_cpu", bench::cpu_model()).num("threads", 1)
            .str("platform", bench::is_wsl() ? "WSL2" : "native Linux")
            .str("compiler_flags", CPU_FLAGS).str("weights", weights).num("simd_tile_frames", TILE)
            .raw("throughput", bench::jarr(tp, 2))
            .str("latency_impl", "simd")
            .str("latency_definition", "wall time to score one batch of B frames (steady_clock), one thread")
            .str("percentile_method", "nearest-rank")
            .raw("latency", bench::jarr(lat_rows, 2));
        bench::write_file(json_out, root.dump());
    }
    return 0;
}
