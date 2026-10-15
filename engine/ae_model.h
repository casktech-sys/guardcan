// ae_model.h — the GuardCAN autoencoder as a plain struct + ONE forward pass shared by CPU and GPU.
//
// Architecture (fixed at compile time so the GPU kernel keeps everything in registers):
//   x[22] -> standardize timing columns -> 16 (ReLU) -> 8 (ReLU) -> 16 (ReLU) -> 22 (identity)
//   score = mean((recon - x_scaled)^2)   == model.reconstruction_error() in Python
//
// The struct is POD so the GPU engine can copy it verbatim into __constant__ memory: every thread
// reads the same weight at the same time, which is the constant cache's broadcast case.
#pragma once

#include <cmath>
#include <string>
#include <stdexcept>

#include "npz.h"

#ifdef __CUDACC__
#define AE_HD __host__ __device__ __forceinline__
#else
#define AE_HD inline
#endif
#if defined(__CUDA_ARCH__)
#define AE_UNROLL _Pragma("unroll")
#elif defined(__CUDACC__)
#define AE_UNROLL            /* host side of a .cu build: not the perf path */
#elif defined(__GNUC__)
#define AE_UNROLL _Pragma("GCC unroll 32")
#else
#define AE_UNROLL
#endif

namespace ae {

constexpr int D_IN = 22, H1 = 16, H2 = 8;   // 22 -> 16 -> 8 -> 16 -> 22
constexpr int N_PARAMS = (D_IN * H1 + H1) + (H1 * H2 + H2) + (H2 * H1 + H1) + (H1 * D_IN + D_IN);

struct Model {
    float W0[H1][D_IN], b0[H1];      // weights are [out][in], exactly as exported
    float W1[H2][H1],   b1[H2];
    float W2[H1][H2],   b2[H1];
    float W3[D_IN][H1], b3[D_IN];
    float mean[D_IN], std_[D_IN];    // feature scaler; only columns with scaled[j] != 0 are used
    int   scaled[D_IN];
    float threshold;                 // anomaly threshold (score > threshold -> anomalous)
};

// Score one frame of RAW features (the scaler is applied here, fused into the forward pass).
AE_HD float score(const Model& m, const float* x) {
    float xs[D_IN];
AE_UNROLL
    for (int j = 0; j < D_IN; ++j) xs[j] = m.scaled[j] ? (x[j] - m.mean[j]) / m.std_[j] : x[j];

    float h1[H1];
AE_UNROLL
    for (int o = 0; o < H1; ++o) {
        float a = m.b0[o];
AE_UNROLL
        for (int j = 0; j < D_IN; ++j) a += m.W0[o][j] * xs[j];
        h1[o] = a > 0.f ? a : 0.f;
    }
    float h2[H2];
AE_UNROLL
    for (int o = 0; o < H2; ++o) {
        float a = m.b1[o];
AE_UNROLL
        for (int j = 0; j < H1; ++j) a += m.W1[o][j] * h1[j];
        h2[o] = a > 0.f ? a : 0.f;
    }
    float h3[H1];
AE_UNROLL
    for (int o = 0; o < H1; ++o) {
        float a = m.b2[o];
AE_UNROLL
        for (int j = 0; j < H2; ++j) a += m.W2[o][j] * h2[j];
        h3[o] = a > 0.f ? a : 0.f;
    }
    float sse = 0.f;
AE_UNROLL
    for (int o = 0; o < D_IN; ++o) {
        float a = m.b3[o];
AE_UNROLL
        for (int j = 0; j < H1; ++j) a += m.W3[o][j] * h3[j];
        const float d = a - xs[o];
        sse += d * d;
    }
    return sse / (float)D_IN;
}

// Load the .npz written by model/export_weights.py and validate its shapes.
inline Model load_model(const std::string& path) {
    const npz::Archive a = npz::load(path);
    const std::vector<float> dims = npz::get(a, "dims").to_float();
    const int want[5] = {D_IN, H1, H2, H1, D_IN};
    if (dims.size() != 5) throw std::runtime_error("ae: expected 4 layers in " + path);
    for (int i = 0; i < 5; ++i)
        if ((int)dims[i] != want[i]) throw std::runtime_error("ae: layer dims do not match 22-16-8-16-22");

    Model m{};
    auto fill = [&](const char* key, float* dst, size_t n) {
        const std::vector<float> v = npz::get(a, key).to_float();
        if (v.size() != n) throw std::runtime_error(std::string("ae: bad size for ") + key);
        for (size_t i = 0; i < n; ++i) dst[i] = v[i];
    };
    fill("W0", &m.W0[0][0], H1 * D_IN); fill("b0", m.b0, H1);
    fill("W1", &m.W1[0][0], H2 * H1);   fill("b1", m.b1, H2);
    fill("W2", &m.W2[0][0], H1 * H2);   fill("b2", m.b2, H1);
    fill("W3", &m.W3[0][0], D_IN * H1); fill("b3", m.b3, D_IN);

    const std::vector<float> idx = npz::get(a, "scaler_idx").to_float();
    const std::vector<float> mu = npz::get(a, "scaler_mean").to_float();
    const std::vector<float> sd = npz::get(a, "scaler_std").to_float();
    for (int j = 0; j < D_IN; ++j) { m.mean[j] = 0.f; m.std_[j] = 1.f; m.scaled[j] = 0; }
    for (size_t k = 0; k < idx.size(); ++k) {
        const int j = (int)idx[k];
        if (j < 0 || j >= D_IN) throw std::runtime_error("ae: scaler index out of range");
        m.mean[j] = mu[k]; m.std_[j] = sd[k]; m.scaled[j] = 1;
    }
    m.threshold = npz::get(a, "threshold").to_float().at(0);
    return m;
}

// Reference set written by export_weights.py --data: raw test-split features + Python scores.
struct Reference {
    std::vector<float> X;       // [n, D_IN]
    std::vector<float> err;     // [n]
    size_t n = 0;
};

inline Reference load_reference(const std::string& path) {
    const npz::Archive a = npz::load(path);
    const npz::Array& X = npz::get(a, "X");
    if (X.shape.size() != 2 || X.shape[1] != (size_t)D_IN) throw std::runtime_error("ae: reference X must be [n, 22]");
    Reference r;
    r.X = X.to_float();
    r.err = npz::get(a, "err").to_float();
    r.n = X.shape[0];
    if (r.err.size() != r.n) throw std::runtime_error("ae: reference err length mismatch");
    return r;
}

}  // namespace ae
