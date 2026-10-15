// ae_kernel.cuh — the fused GuardCAN scoring kernel + CUDA helpers shared by the engine binaries.
//
// One thread scores one frame. Per block:
//   1. the block's frames (TPB x 22 floats, contiguous) are loaded COALESCED into shared memory —
//      float4 when 16-byte aligned. This matters most for zero-copy, where the "global" pointer is
//      host memory across PCIe and wide, contiguous requests are the only way to get bandwidth.
//   2. each thread runs ae::score() (scaler + 4 layers + MSE) entirely in registers; the weights
//      live in __constant__ memory and are broadcast to the warp.
//   3. one float per frame is written back (coalesced).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "ae_model.h"
#include "bench_util.h"

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),    \
                         __FILE__, __LINE__, cudaGetErrorString(err_));                     \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace gpu {

constexpr int TPB = 128;                    // frames (threads) per block

__constant__ ae::Model c_model;

__global__ void __launch_bounds__(TPB) score_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ __align__(16) float tile[TPB * ae::D_IN];
    const int base = blockIdx.x * TPB;
    const int nv = min(TPB, n - base);
    const float* src = x + (size_t)base * ae::D_IN;
    const int nf = nv * ae::D_IN;

    if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        const int n4 = nf >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(src);
        float4* t4 = reinterpret_cast<float4*>(tile);
        for (int i = threadIdx.x; i < n4; i += TPB) t4[i] = s4[i];
        for (int i = (n4 << 2) + threadIdx.x; i < nf; i += TPB) tile[i] = src[i];
    } else {
        for (int i = threadIdx.x; i < nf; i += TPB) tile[i] = src[i];
    }
    __syncthreads();

    if (threadIdx.x < nv) out[base + threadIdx.x] = ae::score(c_model, tile + threadIdx.x * ae::D_IN);
}

inline void upload_model(const ae::Model& m) { CUDA_CHECK(cudaMemcpyToSymbol(c_model, &m, sizeof(ae::Model))); }

inline void launch(const float* d_in, float* d_out, int n, cudaStream_t s) {
    if (n <= 0) return;
    score_kernel<<<(n + TPB - 1) / TPB, TPB, 0, s>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
}

// cudaMemPrefetchAsync changed signature in CUDA 13 (cudaMemLocation instead of a device id).
inline cudaError_t prefetch(const void* p, size_t bytes, bool to_host, int dev, cudaStream_t s) {
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = to_host ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    loc.id = to_host ? 0 : dev;
    return cudaMemPrefetchAsync(p, bytes, loc, 0, s);
#else
    return cudaMemPrefetchAsync(p, bytes, to_host ? cudaCpuDeviceId : dev, s);
#endif
}

inline int attr(cudaDeviceAttr a, int dev = 0) { int v = 0; cudaDeviceGetAttribute(&v, a, dev); return v; }

// Everything a reader needs to judge the numbers: which GPU, which driver/runtime, which platform.
inline std::string device_json(int ind) {
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    return bench::JObj()
        .str("gpu", p.name)
        .str("compute_capability", std::to_string(p.major) + "." + std::to_string(p.minor))
        .num("sm_count", p.multiProcessorCount)
        .num("global_mem_gb", p.totalGlobalMem / 1e9)
        .num("cuda_driver_version", drv)
        .num("cuda_runtime_version", rt)
        .num("async_engine_count", attr(cudaDevAttrAsyncEngineCount))
        .num("concurrent_managed_access", attr(cudaDevAttrConcurrentManagedAccess))
        .num("pageable_memory_access", attr(cudaDevAttrPageableMemoryAccess))
        .num("can_map_host_memory", attr(cudaDevAttrCanMapHostMemory))
        .str("host_cpu", bench::cpu_model())
        .str("platform", bench::is_wsl() ? "WSL2 (Linux on Windows, GPU via WDDM paravirtualization)" : "native Linux")
        .dump(ind);
}

}  // namespace gpu
