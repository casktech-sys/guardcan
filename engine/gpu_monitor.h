// gpu_monitor.h — background sampler of the GPU's clocks and power state (P-state) via NVML.
//
// Why: a laptop GPU's driver drops to an idle P-state (e.g. P8, a few hundred MHz) when the GPU is
// mostly waiting — which is exactly what a latency-bound pipeline looks like. The same code then runs
// several times slower. Every benchmark row records the clocks it actually ran at, so a number is
// never reported without the power state that produced it.
//
// NVML is dlopen'd (libnvidia-ml.so.1 ships with the driver, including WSL2 and Colab), so there is
// no header or link-time dependency; if it is missing, the monitor reports "unavailable".
#pragma once

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bench_util.h"

namespace mon {

struct Sample { unsigned sm = 0, mem = 0, util = 0; int pstate = -1; bool util_ok = false; };

class GpuMonitor {
    using FnVoid = int (*)();
    using FnHandle = int (*)(unsigned, void**);
    using FnClock = int (*)(void*, int, unsigned*);     // NVML_CLOCK_SM = 1, NVML_CLOCK_MEM = 2
    using FnPstate = int (*)(void*, int*);
    struct Util { unsigned gpu, memory; };
    using FnUtil = int (*)(void*, Util*);

    void* lib_ = nullptr;
    void* dev_ = nullptr;
    FnClock clock_ = nullptr;
    FnPstate pstate_ = nullptr;
    FnUtil util_ = nullptr;
    int period_ms_;
    std::thread th_;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::vector<Sample> samples_;

    void sample() {
        Sample s;
        clock_(dev_, 1, &s.sm);
        clock_(dev_, 2, &s.mem);
        pstate_(dev_, &s.pstate);
        Util u{};
        if (util_ && util_(dev_, &u) == 0) { s.util = u.gpu; s.util_ok = true; }
        std::lock_guard<std::mutex> g(mu_);
        samples_.push_back(s);
    }

public:
    explicit GpuMonitor(int period_ms = 50, bool enabled = true) : period_ms_(period_ms) {
        if (!enabled) return;
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib_) return;
        auto init = (FnVoid)dlsym(lib_, "nvmlInit_v2");
        auto get = (FnHandle)dlsym(lib_, "nvmlDeviceGetHandleByIndex_v2");
        clock_ = (FnClock)dlsym(lib_, "nvmlDeviceGetClockInfo");
        pstate_ = (FnPstate)dlsym(lib_, "nvmlDeviceGetPerformanceState");
        util_ = (FnUtil)dlsym(lib_, "nvmlDeviceGetUtilizationRates");
        if (!init || !get || !clock_ || !pstate_ || init() != 0 || get(0, &dev_) != 0) dev_ = nullptr;
    }
    ~GpuMonitor() {
        stop();
        if (lib_) {
            auto shutdown = (FnVoid)dlsym(lib_, "nvmlShutdown");
            if (shutdown && dev_) shutdown();
            dlclose(lib_);
        }
    }
    bool available() const { return dev_ != nullptr; }

    void start() {
        if (!dev_) return;
        { std::lock_guard<std::mutex> g(mu_); samples_.clear(); }
        running_ = true;
        th_ = std::thread([this] {
            while (running_) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
            }
        });
    }
    std::vector<Sample> stop() {
        if (running_) { running_ = false; th_.join(); sample(); }
        std::lock_guard<std::mutex> g(mu_);
        return samples_;
    }
};

inline double median_u(std::vector<unsigned> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Summary of one measurement window: median clocks, the P-state it spent most samples in, utilization.
struct Summary { size_t n = 0; double sm = 0, mem = 0, util = -1; std::string pstate = "n/a"; double pstate_frac = 0; };

inline Summary summarize(const std::vector<Sample>& s) {
    Summary r;
    r.n = s.size();
    if (s.empty()) return r;
    std::vector<unsigned> sm, mem, util;
    std::map<int, int> ps;
    for (const Sample& x : s) {
        sm.push_back(x.sm); mem.push_back(x.mem); ps[x.pstate]++;
        if (x.util_ok) util.push_back(x.util);
    }
    r.sm = median_u(sm); r.mem = median_u(mem);
    if (!util.empty()) r.util = median_u(util);
    auto top = std::max_element(ps.begin(), ps.end(), [](auto& a, auto& b) { return a.second < b.second; });
    r.pstate = "P" + std::to_string(top->first);
    r.pstate_frac = (double)top->second / s.size();
    return r;
}

inline std::string summary_json(const Summary& m, int ind) {
    if (m.n == 0) return "null";
    bench::JObj o;
    o.num("samples", (double)m.n).num("sm_clock_mhz_median", m.sm).num("mem_clock_mhz_median", m.mem)
        .str("pstate_mode", m.pstate).num("pstate_mode_fraction", m.pstate_frac);
    if (m.util >= 0) o.num("gpu_util_pct_median", m.util);
    return o.dump(ind);
}

}  // namespace mon
