// mem_strategy_bench.cu — how should CAN frames get from host memory to the GPU and back?
//
// The ingest path, per micro-batch of B frames, is the same for every strategy:
//   produce  the CPU writes the batch's feature vectors into the strategy's input buffer
//            (models the ingest thread / driver landing frames in host memory)
//   score    the fused autoencoder kernel (ae_kernel.cuh) produces one anomaly score per frame
//   consume  the CPU reads the scores and counts frames over the threshold (an alerting decision)
// Only the memory the buffers live in — and therefore how the bytes cross PCIe — changes:
//
//   pageable     malloc'd host memory; cudaMemcpyAsync H2D/D2H. The driver stages through its own
//                pinned bounce buffer (an extra CPU copy) and D2H to pageable memory is synchronous.
//   pinned       cudaHostAlloc page-locked memory; true async DMA H2D/D2H on the copy engine.
//   zero-copy    cudaHostAllocMapped; NO copies — the kernel reads the frames over PCIe directly and
//                writes scores straight into host memory.
//   zero-copy+WC same, input buffer write-combined (fast CPU streaming writes and GPU reads over PCIe;
//                CPU reads from it are slow, but the CPU never reads the input).
//   unified      cudaMallocManaged; the driver migrates pages. Buffers are attached to their stream
//                (cudaStreamAttachMemAsync) so the CPU may touch one slot while others are running —
//                required when concurrentManagedAccess == 0 (Windows / WSL2).
//   unified+prefetch  unified + explicit cudaMemPrefetchAsync both ways (needs concurrentManagedAccess;
//                reported as unsupported otherwise).
//
// Two measurements per strategy, both on the local GPU:
//   latency     closed loop, ONE batch in flight: host wall clock from produce start to consume end
//               (end-to-end), plus the produce / GPU / consume split and the CUDA-event GPU time.
//               p50 / p99 / p99.9 over --iters batches per cell, taken in --rounds interleaved blocks.
//   throughput  open loop, double-buffered waves of S streams (breadth-first issue: all inbound
//               transfers, then all kernels, then all outbound), CPU produce/consume of one wave
//               overlapping the GPU work of the other. Median frames/s over --reps interleaved runs.
// Cells are interleaved in a shuffled (seeded) order every round so drift and GPU power-state history
// affect all strategies alike; the GPU clock / P-state is recorded for every cell.
//
//   mem_strategy_bench --weights models/ae_weights.npz --ref models/ae_reference.npz --json out.json
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "ae_kernel.cuh"
#include "gpu_monitor.h"

using bench::JObj;

enum class Mem { Pageable, Pinned, ZeroCopy, ZeroCopyWC, Unified, UnifiedPrefetch };
struct StratInfo { Mem m; const char* key; const char* desc; };
static const StratInfo kStrategies[] = {
    {Mem::Pageable, "pageable", "malloc host buffers + cudaMemcpyAsync (driver-staged)"},
    {Mem::Pinned, "pinned", "cudaHostAlloc page-locked buffers + async DMA copies"},
    {Mem::ZeroCopy, "zero-copy", "cudaHostAllocMapped; kernel reads/writes host memory over PCIe, no copies"},
    {Mem::ZeroCopyWC, "zero-copy+wc", "mapped + write-combined input buffer; no copies"},
    {Mem::Unified, "unified", "cudaMallocManaged, stream-attached; driver migrates pages"},
    {Mem::UnifiedPrefetch, "unified+prefetch", "cudaMallocManaged + cudaMemPrefetchAsync to device and back"},
};

static int g_dev = 0;
static mon::GpuMonitor* g_mon = nullptr;     // samples clocks / P-state during every measured config

// One in-flight micro-batch: host-visible buffers, the pointers the kernel uses, its stream.
struct Slot {
    Mem m; int B;
    float *h_in = nullptr, *h_out = nullptr;      // CPU writes h_in, reads h_out
    float *d_in = nullptr, *d_out = nullptr;      // kernel reads d_in, writes d_out
    cudaStream_t st{};
    cudaEvent_t e0{}, e1{};

    Slot(Mem mem, int batch) : m(mem), B(batch) {
        const size_t bin = sizeof(float) * B * ae::D_IN, bout = sizeof(float) * B;
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        switch (m) {
        case Mem::Pageable:
            h_in = (float*)std::aligned_alloc(64, round64(bin)); h_out = (float*)std::aligned_alloc(64, round64(bout));
            std::memset(h_in, 0, bin); std::memset(h_out, 0, bout);           // fault the pages in up front
            CUDA_CHECK(cudaMalloc(&d_in, bin)); CUDA_CHECK(cudaMalloc(&d_out, bout));
            break;
        case Mem::Pinned:
            CUDA_CHECK(cudaHostAlloc(&h_in, bin, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_out, bout, cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc(&d_in, bin)); CUDA_CHECK(cudaMalloc(&d_out, bout));
            break;
        case Mem::ZeroCopy:
        case Mem::ZeroCopyWC:
            CUDA_CHECK(cudaHostAlloc(&h_in, bin, cudaHostAllocMapped |
                                     (m == Mem::ZeroCopyWC ? cudaHostAllocWriteCombined : 0)));
            CUDA_CHECK(cudaHostAlloc(&h_out, bout, cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_in, h_in, 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_out, h_out, 0));
            break;
        case Mem::Unified:
        case Mem::UnifiedPrefetch:
            CUDA_CHECK(cudaMallocManaged(&h_in, bin, cudaMemAttachGlobal));
            CUDA_CHECK(cudaMallocManaged(&h_out, bout, cudaMemAttachGlobal));
            CUDA_CHECK(cudaStreamAttachMemAsync(st, h_in, 0, cudaMemAttachSingle));
            CUDA_CHECK(cudaStreamAttachMemAsync(st, h_out, 0, cudaMemAttachSingle));
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::memset(h_in, 0, bin); std::memset(h_out, 0, bout);
            d_in = h_in; d_out = h_out;
            break;
        }
    }
    ~Slot() {
        switch (m) {
        case Mem::Pageable: std::free(h_in); std::free(h_out); cudaFree(d_in); cudaFree(d_out); break;
        case Mem::Pinned: cudaFreeHost(h_in); cudaFreeHost(h_out); cudaFree(d_in); cudaFree(d_out); break;
        case Mem::ZeroCopy: case Mem::ZeroCopyWC: cudaFreeHost(h_in); cudaFreeHost(h_out); break;
        case Mem::Unified: case Mem::UnifiedPrefetch: cudaFree(h_in); cudaFree(h_out); break;
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(st);
    }
    static size_t round64(size_t n) { return (n + 63) / 64 * 64; }

    // ---- the three GPU-side phases (enqueue only)
    void inbound(int n) {
        if (m == Mem::Pageable || m == Mem::Pinned)
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, sizeof(float) * n * ae::D_IN, cudaMemcpyHostToDevice, st));
        else if (m == Mem::UnifiedPrefetch)
            CUDA_CHECK(gpu::prefetch(h_in, sizeof(float) * n * ae::D_IN, false, g_dev, st));
    }
    void kernel(int n) { gpu::launch(d_in, d_out, n, st); }
    void outbound(int n) {
        if (m == Mem::Pageable || m == Mem::Pinned)
            CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost, st));
        else if (m == Mem::UnifiedPrefetch)
            CUDA_CHECK(gpu::prefetch(h_out, sizeof(float) * n, true, g_dev, st));
    }
    // ---- host-side phases
    void produce(const float* src, int n) { std::memcpy(h_in, src, sizeof(float) * n * ae::D_IN); }
    int consume(int n, float thr) const { int c = 0; for (int i = 0; i < n; ++i) c += h_out[i] > thr; return c; }
};

static std::string supported(Mem m) {
    if ((m == Mem::ZeroCopy || m == Mem::ZeroCopyWC) && !gpu::attr(cudaDevAttrCanMapHostMemory))
        return "device cannot map host memory";
    if ((m == Mem::Unified || m == Mem::UnifiedPrefetch) && !gpu::attr(cudaDevAttrManagedMemory))
        return "device does not support managed memory";
    if (m == Mem::UnifiedPrefetch && !gpu::attr(cudaDevAttrConcurrentManagedAccess))
        return "cudaMemPrefetchAsync needs concurrentManagedAccess=1; this platform reports 0 (Windows/WSL2 managed memory)";
    return "";
}

// ------------------------------------------------------------------ interleaved scheduling
// Every (strategy, batch) cell is measured in several blocks, and each round visits the cells in a
// freshly shuffled order (fixed seed). Slow drift — thermals, background load, and above all the GPU's
// power state, which depends on what ran just before — then lands on every cell alike instead of
// biasing whichever strategy happened to run first or last.
static std::vector<size_t> shuffled(size_t n, unsigned seed) {
    std::vector<size_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = i;
    std::shuffle(v.begin(), v.end(), std::mt19937(seed));
    return v;
}

// ------------------------------------------------------------------ latency (closed loop)
struct LatCell {
    const StratInfo* si; int B;
    Slot* slot = nullptr;
    size_t pos = 0;                                              // position in the source ring (in batches)
    std::vector<double> e2e, prod, gpuw, cons, gev;
    std::vector<mon::Sample> clk;
    bench::Check chk; bool checked = false;
};

static void latency_block(LatCell& c, long iters, long warm, const float* src, size_t ring_frames,
                          const ae::Model& model, bool& ok) {
    Slot& s = *c.slot;
    const int B = c.B;
    const size_t slots = ring_frames / B;
    volatile int sink = 0;
    for (long it = -warm; it < iters; ++it) {
        if (it == 0 && g_mon) g_mon->start();
        const float* in = src + (c.pos++ % slots) * B * ae::D_IN;
        const double t0 = bench::now_us();
        s.produce(in, B);
        const double t1 = bench::now_us();
        CUDA_CHECK(cudaEventRecord(s.e0, s.st));
        s.inbound(B); s.kernel(B); s.outbound(B);
        CUDA_CHECK(cudaEventRecord(s.e1, s.st));
        CUDA_CHECK(cudaStreamSynchronize(s.st));
        const double t2 = bench::now_us();
        sink += s.consume(B, model.threshold);
        const double t3 = bench::now_us();
        if (!c.checked) {                                // correctness: GPU scores vs the CPU forward pass
            std::vector<float> want(B);
            for (int i = 0; i < B; ++i) want[i] = ae::score(model, in + (size_t)i * ae::D_IN);
            c.chk = bench::compare(s.h_out, want.data(), B, model.threshold);
            c.checked = true;
            if (!c.chk.pass) ok = false;
        }
        if (it < 0) continue;
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s.e0, s.e1));
        c.e2e.push_back(t3 - t0); c.prod.push_back(t1 - t0); c.gpuw.push_back(t2 - t1); c.cons.push_back(t3 - t2);
        c.gev.push_back(ms * 1000.0);
    }
    if (g_mon) { auto v = g_mon->stop(); c.clk.insert(c.clk.end(), v.begin(), v.end()); }
}

static std::string latency_json(const LatCell& c, int rounds) {
    const mon::Summary clk = mon::summarize(c.clk);
    const bench::Pctl p = bench::percentiles(c.e2e);
    std::printf("  latency    %-17s B=%-6d e2e p50 %9.1f  p99 %9.1f  p99.9 %9.1f  max %9.1f us  %5.0f MHz %-3s [%s]\n",
                c.si->key, c.B, p.p50, p.p99, p.p999, p.max, clk.sm, clk.pstate.c_str(),
                c.chk.pass ? "scores OK" : "SCORE MISMATCH");
    return JObj().str("strategy", c.si->key).num("batch", c.B).num("iters", (double)c.e2e.size()).num("blocks", rounds)
        .raw("e2e_us", bench::pctl_json(p, 6))
        .raw("produce_us", bench::pctl_json(bench::percentiles(c.prod), 6))
        .raw("gpu_wall_us", bench::pctl_json(bench::percentiles(c.gpuw), 6))
        .raw("consume_us", bench::pctl_json(bench::percentiles(c.cons), 6))
        .raw("gpu_event_us", bench::pctl_json(bench::percentiles(c.gev), 6))
        .num("frames_per_s_closed_loop", c.B / (p.mean / 1e6))
        .raw("check_vs_cpu_forward", bench::check_json(c.chk, 6))
        .raw("gpu_clocks", mon::summary_json(clk, 6)).dump(4);
}

// ------------------------------------------------------------------ throughput (open loop)
static double run_throughput_once(std::vector<Slot*>& slots, int S, int B, long waves, const float* src,
                                  size_t ring_frames, float thr) {
    const size_t ring_slots = ring_frames / B;
    volatile int sink = 0;
    long k = 0;                                                    // global batch counter (input position)
    const double t0 = bench::now_us();
    for (long w = 0; w < waves + 2; ++w) {
        Slot** set = slots.data() + (w % 2) * S;                   // double-buffered: set A / set B
        if (w >= 2)                                                // retire this set's previous wave
            for (int i = 0; i < S; ++i) { CUDA_CHECK(cudaStreamSynchronize(set[i]->st)); sink += set[i]->consume(B, thr); }
        if (w >= waves) continue;                                  // drain
        for (int i = 0; i < S; ++i, ++k) set[i]->produce(src + (k % ring_slots) * B * ae::D_IN, B);
        for (int i = 0; i < S; ++i) set[i]->inbound(B);
        for (int i = 0; i < S; ++i) set[i]->kernel(B);
        for (int i = 0; i < S; ++i) set[i]->outbound(B);
    }
    const double t1 = bench::now_us();
    return (double)waves * S * B / ((t1 - t0) / 1e6);
}

struct TpCell { const StratInfo* si; int B; long waves = 0; std::vector<double> fps; std::vector<mon::Sample> clk; };

// One rep of one cell: fresh slots, short warm-up, one timed run.
static void throughput_rep(TpCell& c, int S, const float* src, size_t ring_frames, float thr) {
    std::vector<Slot*> slots;
    for (int i = 0; i < 2 * S; ++i) slots.push_back(new Slot(c.si->m, c.B));
    run_throughput_once(slots, S, c.B, std::min(c.waves, 8L), src, ring_frames, thr);       // warm-up
    if (g_mon) g_mon->start();
    c.fps.push_back(run_throughput_once(slots, S, c.B, c.waves, src, ring_frames, thr));
    if (g_mon) { auto v = g_mon->stop(); c.clk.insert(c.clk.end(), v.begin(), v.end()); }
    for (Slot* s : slots) delete s;
}

static std::string throughput_json(const TpCell& c, int S) {
    const mon::Summary clk = mon::summarize(c.clk);
    std::vector<double> sorted = c.fps; std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    std::printf("  throughput %-17s B=%-6d S=%d  median %.4g frames/s  (min %.4g, max %.4g, %zu reps)  %5.0f MHz %s\n",
                c.si->key, c.B, S, med, sorted.front(), sorted.back(), c.fps.size(), clk.sm, clk.pstate.c_str());
    std::vector<std::string> reps_json; for (double f : c.fps) reps_json.push_back(bench::jnum(f));
    return JObj().str("strategy", c.si->key).num("batch", c.B).num("streams_per_wave", S).num("waves_in_flight", 2)
        .num("frames_per_rep", (double)c.waves * S * c.B).num("reps", (double)c.fps.size())
        .num("frames_per_s_median", med).num("frames_per_s_min", sorted.front()).num("frames_per_s_max", sorted.back())
        .raw("frames_per_s_reps", bench::jarr(reps_json, 6))
        .num("input_gb_per_s_median", med * sizeof(float) * ae::D_IN / 1e9)
        .raw("gpu_clocks", mon::summary_json(clk, 6)).dump(4);
}

int main(int argc, char** argv) {
    bench::Args a{argc, argv};
    const std::string weights = a.str("--weights", "../models/ae_weights.npz");
    const std::string refpath = a.str("--ref", "../models/ae_reference.npz");
    const std::string json_out = a.str("--json", "");
    const std::vector<int> lat_batch = a.list("--lat-batch", "1,64,1024,16384");
    const std::vector<int> tp_batch = a.list("--tp-batch", "1024,4096,16384,65536");
    const long iters = a.num("--iters", 20000);
    const int rounds = (int)a.num("--rounds", 10);
    const int S = (int)a.num("--streams", 4);
    const long tp_frames = a.num("--tp-frames", 1L << 26);
    const int reps = (int)a.num("--reps", 3);
    const size_t ring_frames = (size_t)a.num("--ring", 1 << 20);
    const std::string only = a.str("--only", "");

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CUDA_CHECK(cudaGetDevice(&g_dev));
    mon::GpuMonitor monitor(50, !a.has("--no-monitor"));
    if (monitor.available()) g_mon = &monitor;
    const ae::Model model = ae::load_model(weights);
    const ae::Reference ref = ae::load_reference(refpath);
    gpu::upload_model(model);

    // Source ring: ordinary (pageable) memory the producer copies frames out of.
    std::vector<float> src(ring_frames * ae::D_IN);
    bench::fill_ring(src.data(), ring_frames, ref);

    std::printf("GuardCAN memory-strategy benchmark | asyncEngineCount=%d concurrentManagedAccess=%d | %s\n",
                gpu::attr(cudaDevAttrAsyncEngineCount), gpu::attr(cudaDevAttrConcurrentManagedAccess),
                bench::is_wsl() ? "WSL2" : "native Linux");
    std::vector<const StratInfo*> strats;
    std::vector<std::string> skipped;
    for (const StratInfo& si : kStrategies) {
        if (!only.empty() && only != si.key) continue;
        const std::string why = supported(si.m);
        if (!why.empty()) {
            std::printf("  skip       %-17s %s\n", si.key, why.c_str());
            skipped.push_back(JObj().str("strategy", si.key).str("reason", why).dump(4));
        } else {
            strats.push_back(&si);
        }
    }
    bool ok = true;

    // ---- latency: `rounds` blocks of iters/rounds batches per cell, cells shuffled every round
    std::vector<LatCell> lat;
    for (const StratInfo* si : strats) for (int B : lat_batch) { lat.push_back({si, B}); lat.back().slot = new Slot(si->m, B); }
    const long per_block = std::max(1L, iters / rounds);
    for (int r = 0; r < rounds; ++r) {
        for (size_t i : shuffled(lat.size(), 42u + r)) {
            nvtxRangePushA(lat[i].si->key);
            latency_block(lat[i], per_block, 50, src.data(), ring_frames, model, ok);
            nvtxRangePop();
        }
        std::printf("  latency round %d/%d done\n", r + 1, rounds); std::fflush(stdout);
    }
    std::vector<std::string> lat_rows;
    for (LatCell& c : lat) { lat_rows.push_back(latency_json(c, rounds)); delete c.slot; }

    // ---- throughput: `reps` rounds, cells shuffled every round
    std::vector<TpCell> tp;
    for (const StratInfo* si : strats) for (int B : tp_batch) {
        tp.push_back({si, B});
        tp.back().waves = std::max(4L, tp_frames / ((long)S * B));
    }
    for (int r = 0; r < reps; ++r) {
        for (size_t i : shuffled(tp.size(), 1042u + r))
            throughput_rep(tp[i], S, src.data(), ring_frames, model.threshold);
        std::printf("  throughput round %d/%d done\n", r + 1, reps); std::fflush(stdout);
    }
    std::vector<std::string> tp_rows;
    for (const TpCell& c : tp) tp_rows.push_back(throughput_json(c, S));

    if (!json_out.empty()) {
        std::vector<std::string> strat;
        for (const StratInfo& si : kStrategies) strat.push_back(JObj().str("strategy", si.key).str("description", si.desc).dump(4));
        JObj root;
        root.str("benchmark", "mem_strategy").raw("device", gpu::device_json(2))
            .str("weights", weights).num("bytes_per_frame_in", sizeof(float) * ae::D_IN).num("bytes_per_frame_out", sizeof(float))
            .str("input", "reference feature vectors (held-out test split) replayed from a pageable source ring")
            .str("latency_definition", "closed loop, one batch in flight; host steady_clock from produce start to consume end")
            .str("throughput_definition", "open loop, 2 waves x streams_per_wave batches in flight, breadth-first issue; frames / wall clock incl. produce+consume")
            .str("scheduling", "interleaved: latency in `blocks` blocks per cell and throughput reps, each round visiting all cells in a shuffled order (seeded)")
            .str("percentile_method", "nearest-rank")
            .str("gpu_clocks_note", "NVML samples every 50 ms during each measured block; clocks are driver-managed (not locked)")
            .raw("strategies", bench::jarr(strat, 2))
            .raw("latency", bench::jarr(lat_rows, 2))
            .raw("throughput", bench::jarr(tp_rows, 2))
            .raw("skipped", bench::jarr(skipped, 2));
        bench::write_file(json_out, root.dump());
    }
    return ok ? 0 : 2;
}
