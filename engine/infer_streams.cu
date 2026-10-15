// infer_streams.cu — GuardCAN stream-concurrent, batched GPU inference engine.
//
// Scores a stream of CAN feature vectors with the Week-1 autoencoder (22->16->8->16->22) using a
// hand-written fused kernel (ae_kernel.cuh) and an S-stream pipeline. Each micro-batch of B frames is
//     H2D copy (pinned) -> score kernel -> D2H copy
// on stream k % S. The output is one reconstruction error (anomaly score) per frame.
//
// Issue order (--order) decides how much of that actually overlaps:
//   depth   batch by batch: H2D, kernel, D2H of batch k, then batch k+1 ...
//   breadth in waves of S batches: all S H2Ds, then all S kernels, then all S D2Hs (default).
// With many hardware queues (native Linux, Hyper-Q) both overlap. When the driver funnels streams
// through few queues and every copy<->compute hand-off costs a cross-engine sync (measured on WSL2),
// depth-first serializes and pays that sync per batch; breadth-first pays it once per wave.
//
// What it measures (all on the local GPU, nothing estimated):
//   1. verify  — the Python model's held-out test split through the engine; every score must match
//                the PyTorch/sklearn output within |a-b| <= 1e-6 + 1e-4|b|. Exits non-zero if not.
//   2. sweep   — issue order x streams x micro-batch grid: throughput (frames/s, wall clock) and
//                per-batch latency p50/p99/p99.9 from CUDA events (H2D start -> D2H end, GPU timeline).
//                Reps are interleaved across configurations in a shuffled order; GPU clocks recorded.
//   3. kernel  — device-resident kernel-only throughput, no PCIe: per-launch "sync" timing and a
//                "sustained" back-to-back run after a clock ramp (the kernel's ceiling).
//
//   infer_streams --weights models/ae_weights.npz --ref models/ae_reference.npz --json out.json
//   infer_streams ... --profile        # one short config with NVTX ranges, for nsys
//   infer_streams ... --kernel-only    # verify + the kernel-only measurements, no sweep
//   infer_streams ... --verify-only    # verify only (writes the correctness record with --json)
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "ae_kernel.cuh"
#include "gpu_monitor.h"

using bench::JObj;

static mon::GpuMonitor* g_mon = nullptr;     // samples clocks / P-state during every measured config

enum class Order { Depth, Breadth };
static const char* order_name(Order o) { return o == Order::Depth ? "depth-first" : "breadth-first"; }

// Per-stream device buffers + the round-robin enqueue. Host buffers belong to the caller (pinned).
struct Engine {
    int S, B;
    Order order;
    std::vector<cudaStream_t> st;
    std::vector<float*> d_in, d_out;

    Engine(int streams, int batch, Order o) : S(streams), B(batch), order(o), st(streams), d_in(streams), d_out(streams) {
        for (int s = 0; s < S; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&d_in[s], sizeof(float) * B * ae::D_IN));
            CUDA_CHECK(cudaMalloc(&d_out[s], sizeof(float) * B));
        }
    }
    ~Engine() {
        for (int s = 0; s < S; ++s) { cudaFree(d_in[s]); cudaFree(d_out[s]); cudaStreamDestroy(st[s]); }
    }

    // One micro-batch: n <= B frames from h_in, scores to h_out. e0/e1 (optional) bracket it.
    struct Batch { const float* h_in; float* h_out; int n; cudaEvent_t e0, e1; };

    void h2d(long k, const Batch& b) {
        const int s = (int)(k % S);
        if (b.e0) CUDA_CHECK(cudaEventRecord(b.e0, st[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s], b.h_in, sizeof(float) * b.n * ae::D_IN, cudaMemcpyHostToDevice, st[s]));
    }
    void kernel(long k, const Batch& b) { const int s = (int)(k % S); gpu::launch(d_in[s], d_out[s], b.n, st[s]); }
    void d2h(long k, const Batch& b) {
        const int s = (int)(k % S);
        CUDA_CHECK(cudaMemcpyAsync(b.h_out, d_out[s], sizeof(float) * b.n, cudaMemcpyDeviceToHost, st[s]));
        if (b.e1) CUDA_CHECK(cudaEventRecord(b.e1, st[s]));
    }

    // Enqueue batches [k0, k0 + count), count <= S, in the engine's issue order.
    void enqueue(long k0, const Batch* bs, int count) {
        if (order == Order::Depth) {
            for (int i = 0; i < count; ++i) { h2d(k0 + i, bs[i]); kernel(k0 + i, bs[i]); d2h(k0 + i, bs[i]); }
        } else {
            for (int i = 0; i < count; ++i) h2d(k0 + i, bs[i]);
            for (int i = 0; i < count; ++i) kernel(k0 + i, bs[i]);
            for (int i = 0; i < count; ++i) d2h(k0 + i, bs[i]);
        }
    }
    void sync() { for (auto s : st) CUDA_CHECK(cudaStreamSynchronize(s)); }

    // Score n frames end-to-end (used by verify).
    void score(const float* h_in, float* h_out, size_t n) {
        std::vector<Batch> wave;
        long k0 = 0;
        for (size_t off = 0; off < n; off += B) {
            wave.push_back({h_in + off * ae::D_IN, h_out + off, (int)std::min<size_t>(B, n - off), nullptr, nullptr});
            if ((int)wave.size() == S || off + B >= n) { enqueue(k0, wave.data(), (int)wave.size()); k0 += wave.size(); wave.clear(); }
        }
        sync();
    }
};

struct SweepResult {
    Order order; int S, B; long batches;
    std::vector<double> fps, fps_timed;     // one per rep
    std::vector<double> lat;                // per-batch latency samples, pooled over reps
    std::vector<mon::Sample> clk;           // GPU clocks / P-state while it ran
};

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// One rep of one (order, S, B) configuration: `batches` micro-batches through the pipeline, reading a
// pinned ring. Two passes: throughput WITHOUT per-batch events (recording them costs host time on WDDM
// and lowers throughput measurably), then the same batches WITH events for the latency distribution.
// Reps of all configurations are interleaved in a shuffled order by main(), so drift and GPU power-state
// history are spread over every configuration instead of biasing the ones that ran first or last.
static void run_config_rep(SweepResult& r, const float* h_ring, float* h_out, size_t ring_frames,
                           std::vector<cudaEvent_t>& ev0, std::vector<cudaEvent_t>& ev1, bool nvtx) {
    const int S = r.S, B = r.B;
    const long batches = r.batches;
    Engine eng(S, B, r.order);
    const size_t slots = ring_frames / B;                     // ring holds whole batches only
    auto run = [&](long nb, bool timed) {
        std::vector<Engine::Batch> wave(S);
        for (long k0 = 0; k0 < nb; k0 += S) {
            const int cnt = (int)std::min<long>(S, nb - k0);
            for (int i = 0; i < cnt; ++i) {
                const long k = k0 + i; const size_t slot = k % slots;
                wave[i] = {h_ring + slot * B * ae::D_IN, h_out + slot * B, B,
                           timed ? ev0[k] : nullptr, timed ? ev1[k] : nullptr};
            }
            eng.enqueue(k0, wave.data(), cnt);
        }
        eng.sync();
    };
    run(std::max(4L * S, batches / 8), false);                // warm-up (GPU clocks, driver queues)

    char label[96]; std::snprintf(label, sizeof label, "%s S=%d B=%d", order_name(r.order), S, B);
    if (nvtx) nvtxRangePushA(label);
    if (g_mon) g_mon->start();
    const double t0 = bench::now_us();
    run(batches, false);
    const double t1 = bench::now_us();
    run(batches, true);
    const double t2 = bench::now_us();
    if (g_mon) { auto v = g_mon->stop(); r.clk.insert(r.clk.end(), v.begin(), v.end()); }
    if (nvtx) nvtxRangePop();
    r.fps.push_back((double)batches * B / ((t1 - t0) / 1e6));
    r.fps_timed.push_back((double)batches * B / ((t2 - t1) / 1e6));
    for (long k = 0; k < batches; ++k) {
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0[k], ev1[k]));
        r.lat.push_back(ms * 1000.0);
    }
}

int main(int argc, char** argv) {
    bench::Args a{argc, argv};
    const std::string weights = a.str("--weights", "../models/ae_weights.npz");
    const std::string refpath = a.str("--ref", "../models/ae_reference.npz");
    const std::string json_out = a.str("--json", "");
    const bool profile = a.has("--profile");
    std::vector<int> streams = a.list("--streams", "1,2,4,8");
    std::vector<int> batchs = a.list("--batch", "256,1024,4096,16384,65536");
    std::vector<Order> orders;
    const std::string ord = a.str("--order", "both");
    if (ord == "depth" || ord == "both") orders.push_back(Order::Depth);
    if (ord == "breadth" || ord == "both") orders.push_back(Order::Breadth);
    // Batches per config: enough frames for a stable throughput figure, clamped so small batches stay
    // fast and large batches still give thousands of latency samples per rep.
    const long target_frames = a.num("--target-frames", 1L << 27);
    const long min_batches = a.num("--min-batches", 2000), max_batches = a.num("--batches", 20000);
    int reps = (int)a.num("--reps", 3);
    const size_t ring_frames = (size_t)a.num("--ring", 1 << 20);
    if (profile) { streams = {4}; batchs = {4096}; reps = 1; }
    if (a.has("--kernel-only")) { streams.clear(); batchs.clear(); }       // verify + kernel-only measurements
    auto batches_for = [&](int B) {
        if (profile) return a.num("--batches", 200);
        return std::min(max_batches, std::max(min_batches, target_frames / B));   // --batches is a hard cap (event pool)
    };
    const long batches = profile ? batches_for(0) : max_batches;          // event pool size

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    mon::GpuMonitor monitor(50, !a.has("--no-monitor"));
    if (monitor.available()) g_mon = &monitor;
    const ae::Model model = ae::load_model(weights);
    const ae::Reference ref = ae::load_reference(refpath);
    gpu::upload_model(model);
    std::printf("GuardCAN engine | weights=%s | %d params | threshold=%.6f\n", weights.c_str(), ae::N_PARAMS, model.threshold);

    // ---------------------------------------------------------------- 1. verify against Python
    nvtxRangePushA("verify");
    float *h_ref, *h_got;
    CUDA_CHECK(cudaHostAlloc(&h_ref, sizeof(float) * ref.n * ae::D_IN, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_got, sizeof(float) * ref.n, cudaHostAllocDefault));
    std::memcpy(h_ref, ref.X.data(), sizeof(float) * ref.n * ae::D_IN);
    { Engine eng(4, 4096, Order::Breadth); eng.score(h_ref, h_got, ref.n); }
    const bench::Check chk = bench::compare(h_got, ref.err.data(), ref.n, model.threshold);
    nvtxRangePop();
    std::printf("verify: %zu frames vs Python  max|diff|=%.3e  max rel=%.3e  out-of-tol=%zu  decision flips=%zu  -> %s\n",
                chk.n, chk.max_abs, chk.max_rel, chk.bad, chk.flips, chk.pass ? "PASS" : "FAIL");
    if (!chk.pass) return 2;
    if (a.has("--verify-only")) {                     // correctness record for the current models/ only
        if (!json_out.empty())
            bench::write_file(json_out, JObj().str("benchmark", "verify").raw("device", gpu::device_json(2))
                .str("weights", weights).str("reference", refpath).num("model_params", ae::N_PARAMS)
                .num("threshold", model.threshold)
                .str("tolerance", "|gpu - python| <= 1e-6 + 1e-4 * |python|")
                .raw("verify", bench::check_json(chk, 2)).dump());
        cudaFreeHost(h_ref); cudaFreeHost(h_got);
        return 0;
    }

    // ---------------------------------------------------------------- 2. streams x batch sweep
    float *h_ring, *h_out;
    CUDA_CHECK(cudaHostAlloc(&h_ring, sizeof(float) * ring_frames * ae::D_IN, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out, sizeof(float) * ring_frames, cudaHostAllocDefault));
    bench::fill_ring(h_ring, ring_frames, ref);

    std::vector<cudaEvent_t> ev0(batches), ev1(batches);
    for (long k = 0; k < batches; ++k) { CUDA_CHECK(cudaEventCreate(&ev0[k])); CUDA_CHECK(cudaEventCreate(&ev1[k])); }

    std::printf("\n%-14s %-8s %-8s %10s %14s %12s %12s %12s %12s %14s\n", "order", "streams", "batch", "batches",
                "frames/s", "lat p50 us", "p99 us", "p99.9 us", "max us", "SM MHz/Pstate");
    std::vector<SweepResult> cfgs;
    for (Order order : orders) for (int B : batchs) for (int S : streams) {
        if ((size_t)B * S > ring_frames) continue;
        if (order == Order::Breadth && S == 1 && orders.size() > 1) continue;   // identical to depth-first
        cfgs.push_back({order, S, B, batches_for(B)});
    }
    for (int rep = 0; rep < reps; ++rep) {
        std::vector<size_t> idx(cfgs.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        if (!profile) std::shuffle(idx.begin(), idx.end(), std::mt19937(42u + rep));
        for (size_t i : idx) run_config_rep(cfgs[i], h_ring, h_out, ring_frames, ev0, ev1, profile);
        if (!profile && !cfgs.empty()) { std::printf("  sweep round %d/%d done\n", rep + 1, reps); std::fflush(stdout); }
    }
    std::vector<std::string> rows;
    for (const SweepResult& r : cfgs) {
        const double fps_med = median(r.fps);
        const bench::Pctl lat = bench::percentiles(r.lat);
        const mon::Summary clk = mon::summarize(r.clk);
        std::printf("%-14s %-8d %-8d %10ld %14.4g %12.1f %12.1f %12.1f %12.1f %9.0f %s\n", order_name(r.order), r.S, r.B,
                    r.batches, fps_med, lat.p50, lat.p99, lat.p999, lat.max, clk.sm, clk.pstate.c_str());
        std::vector<std::string> fr; for (double f : r.fps) fr.push_back(bench::jnum(f));
        rows.push_back(JObj().str("order", order_name(r.order)).num("streams", r.S).num("batch", r.B)
            .num("batches_per_rep", (double)r.batches).num("reps", (double)r.fps.size())
            .num("frames_per_rep", (double)r.batches * r.B)
            .num("frames_per_s_median", fps_med)
            .num("frames_per_s_min", *std::min_element(r.fps.begin(), r.fps.end()))
            .num("frames_per_s_max", *std::max_element(r.fps.begin(), r.fps.end()))
            .raw("frames_per_s_reps", bench::jarr(fr, 6))
            .num("h2d_gb_per_s_median", fps_med * sizeof(float) * ae::D_IN / 1e9)
            .num("frames_per_s_with_events_median", median(r.fps_timed))
            .raw("batch_latency_us", bench::pctl_json(lat, 6))
            .raw("gpu_clocks", mon::summary_json(clk, 6)).dump(4));
    }
    for (long k = 0; k < batches; ++k) { cudaEventDestroy(ev0[k]); cudaEventDestroy(ev1[k]); }

    // ---------------------------------------------------------------- 3. kernel-only (device-resident, no PCIe)
    //   sync       each launch timed in isolation (record, launch, record, wait): what a request/response
    //              caller sees. The GPU idles between launches, so a driver-managed clock may stay low.
    //   sustained  >= 300 ms of back-to-back launches to let the clock ramp, then back-to-back launches
    //              with an event between each: the throughput ceiling of the kernel itself.
    JObj kernel_sync, kernel_sus;
    if (!profile) {
        const int n = (int)ring_frames, launches = 200;
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * n * ae::D_IN));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(d_in, h_ring, sizeof(float) * n * ae::D_IN, cudaMemcpyHostToDevice));
        std::vector<cudaEvent_t> kev(launches + 1);
        for (auto& e : kev) CUDA_CHECK(cudaEventCreate(&e));
        auto report = [&](const char* mode, std::vector<double> per, JObj& out) {
            const mon::Summary clk = g_mon ? mon::summarize(g_mon->stop()) : mon::Summary{};
            const bench::Pctl p = bench::percentiles(per);
            const double fps = n / (p.p50 / 1e6), gbs = fps * sizeof(float) * (ae::D_IN + 1) / 1e9;
            std::printf("kernel-only %-9s (%d frames/launch, %d launches): p50 %.1f us -> %.4g frames/s (%.1f GB/s DRAM) at %.0f MHz %s\n",
                        mode, n, launches, p.p50, fps, gbs, clk.sm, clk.pstate.c_str());
            out.str("mode", mode).num("frames_per_launch", n).num("launches", launches).raw("launch_us", bench::pctl_json(p, 4))
                .num("frames_per_s_at_p50", fps).num("dram_gb_per_s_at_p50", gbs).raw("gpu_clocks", mon::summary_json(clk, 4));
        };
        std::printf("\n");
        {   // sync
            for (int i = 0; i < 10; ++i) gpu::launch(d_in, d_out, n, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> per;
            if (g_mon) g_mon->start();
            for (int i = 0; i < launches; ++i) {
                CUDA_CHECK(cudaEventRecord(kev[0])); gpu::launch(d_in, d_out, n, 0); CUDA_CHECK(cudaEventRecord(kev[1]));
                CUDA_CHECK(cudaEventSynchronize(kev[1]));
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, kev[0], kev[1])); per.push_back(ms * 1000.0);
            }
            report("sync", per, kernel_sync);
        }
        {   // sustained
            const double ramp0 = bench::now_us();
            while (bench::now_us() - ramp0 < 300000.0) {
                for (int i = 0; i < 50; ++i) gpu::launch(d_in, d_out, n, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            if (g_mon) g_mon->start();
            CUDA_CHECK(cudaEventRecord(kev[0]));
            for (int i = 0; i < launches; ++i) { gpu::launch(d_in, d_out, n, 0); CUDA_CHECK(cudaEventRecord(kev[i + 1])); }
            CUDA_CHECK(cudaEventSynchronize(kev[launches]));
            std::vector<double> per;
            for (int i = 0; i < launches; ++i) {
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, kev[i], kev[i + 1])); per.push_back(ms * 1000.0);
            }
            report("sustained", per, kernel_sus);
        }
        for (auto& e : kev) cudaEventDestroy(e);
        cudaFree(d_in); cudaFree(d_out);
    }

    if (!json_out.empty()) {
        JObj root;
        root.str("benchmark", "infer_streams").raw("device", gpu::device_json(2))
            .str("weights", weights).num("model_params", ae::N_PARAMS).num("threads_per_block", gpu::TPB)
            .str("input", "reference feature vectors (held-out test split) replayed cyclically through a pinned ring")
            .num("ring_frames", (double)ring_frames).num("bytes_per_frame_in", sizeof(float) * ae::D_IN)
            .num("bytes_per_frame_out", sizeof(float))
            .str("latency_definition", "per micro-batch, CUDA events on the batch's stream: before H2D -> after D2H")
            .str("throughput_definition", "frames / wall clock of the event-free pass, median over reps (latency comes from a second, instrumented pass; samples pooled over reps)")
            .str("scheduling", "interleaved: each rep runs every configuration once, in a shuffled (seeded) order")
            .str("percentile_method", "nearest-rank")
            .str("gpu_clocks_note", "NVML samples every 50 ms during each config; clocks are driver-managed (not locked)")
            .raw("verify", bench::check_json(chk, 2))
            .raw("sweep", bench::jarr(rows, 2));
        if (!profile) root.raw("kernel_only", kernel_sync.dump(2)).raw("kernel_only_sustained", kernel_sus.dump(2));
        bench::write_file(json_out, root.dump());
    }
    cudaFreeHost(h_ref); cudaFreeHost(h_got); cudaFreeHost(h_ring); cudaFreeHost(h_out);
    return 0;
}
