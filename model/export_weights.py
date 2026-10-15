"""Export the trained autoencoder + feature scaler to a plain .npz the CUDA engine can read.

The engine (engine/*.cu) has no Python, joblib, or PyTorch dependency: it reads this .npz directly
(np.savez -> uncompressed zip of .npy arrays). Layout, all row-major:

  W0..W3        float32 [out, in]   layer weights (nn.Linear convention; sklearn coefs_ transposed)
  b0..b3        float32 [out]       biases
  dims          int32   [5]         layer widths, e.g. 22,16,8,16,22 (ReLU on hidden, identity out)
  scaler_idx    int32   [k]         feature columns the scaler standardizes (the timing columns)
  scaler_mean   float32 [k]
  scaler_std    float32 [k]
  threshold     float32 [1]         anomaly threshold from ae_meta.json

With --data, it also writes a reference set (ae_reference.npz) so the engine can verify itself:

  X             float32 [n, 22]     RAW features (unscaled) of the held-out test split
  err           float32 [n]         the Python model's reconstruction error for those rows
  is_attack     int8    [n]         ground truth (for reference only)

Before writing, the exported arrays are re-run through a NumPy forward pass and checked against the
live model's reconstruction_error(), so a transposition / ordering mistake fails here, not on the GPU.
"""
import os
import json
import argparse
import numpy as np

from ingest.features import FeatureScaler, N_FEATURES
from model.autoencoder import load_autoencoder


def extract_layers(ae):
    """-> list of (W [out, in], b [out]) float32, in forward order, for either backend."""
    if ae.backend == "torch":
        linears = [m for m in ae.net if m.__class__.__name__ == "Linear"]
        return [(m.weight.detach().cpu().numpy().astype(np.float32),
                 m.bias.detach().cpu().numpy().astype(np.float32)) for m in linears]
    if ae.backend == "sklearn":
        m = ae.model
        if m.activation != "relu" or m.out_activation_ != "identity":
            raise ValueError(f"engine expects relu/identity, got {m.activation}/{m.out_activation_}")
        return [(np.ascontiguousarray(W.T, dtype=np.float32), b.astype(np.float32))
                for W, b in zip(m.coefs_, m.intercepts_)]
    raise ValueError(f"unknown backend {ae.backend!r}")


def numpy_forward(layers, scaler_idx, mean, std, X):
    """Reference forward pass from the exported arrays only (what the CUDA kernel computes)."""
    x = np.asarray(X, dtype=np.float32).copy()
    x[:, scaler_idx] = (x[:, scaler_idx] - mean) / std
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return ((h - x) ** 2).mean(axis=1).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export GuardCAN autoencoder weights for the CUDA engine.")
    ap.add_argument("--models", default="models", help="dir with autoencoder.joblib, scaler.joblib, ae_meta.json")
    ap.add_argument("--out", default=None, help="weights .npz (default: <models>/ae_weights.npz)")
    ap.add_argument("--data", default=None,
                    help="dataset dir; if given, also write a reference set for engine verification")
    ap.add_argument("--ref-out", default=None, help="reference .npz (default: <models>/ae_reference.npz)")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.models, "ae_weights.npz")
    ref_out = args.ref_out or os.path.join(args.models, "ae_reference.npz")

    ae = load_autoencoder(os.path.join(args.models, "autoencoder.joblib"))
    scaler = FeatureScaler.load(os.path.join(args.models, "scaler.joblib"))
    meta = json.load(open(os.path.join(args.models, "ae_meta.json")))

    layers = extract_layers(ae)
    dims = [layers[0][0].shape[1]] + [W.shape[0] for W, _ in layers]
    if dims[0] != N_FEATURES or dims[-1] != N_FEATURES:
        raise ValueError(f"unexpected layer dims {dims}")
    for (W, b), (d_in, d_out) in zip(layers, zip(dims[:-1], dims[1:])):
        assert W.shape == (d_out, d_in) and b.shape == (d_out,), (W.shape, b.shape, d_in, d_out)

    idx = np.asarray(scaler.dt_idx, dtype=np.int32)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    std = np.asarray(scaler.std_, dtype=np.float32)

    arrays = {"dims": np.asarray(dims, dtype=np.int32), "scaler_idx": idx,
              "scaler_mean": mean, "scaler_std": std,
              "threshold": np.asarray([meta["threshold"]], dtype=np.float32)}
    for i, (W, b) in enumerate(layers):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b

    # Self-check: the exported arrays must reproduce the live model. Uniform probes put the timing columns
    # hundreds of std devs out (errors ~1e3), so judge float32 rounding relatively, as the CUDA engine does.
    rng = np.random.default_rng(42)
    probe = rng.random((256, N_FEATURES), dtype=np.float32)
    want = ae.reconstruction_error(scaler.transform(probe))
    got = numpy_forward(layers, idx, mean, std, probe)
    absd = np.abs(want - got)
    rel = float(np.max(absd / np.maximum(np.abs(want), 1e-12)))
    print(f"[export] backend={ae.backend}  dims={dims}  numpy-vs-model max|diff|={absd.max():.3e}  "
          f"max rel={rel:.3e}  (probe errors up to {np.abs(want).max():.3g})")
    if np.any(absd > 1e-6 + 1e-4 * np.abs(want)):
        raise RuntimeError("exported weights do not reproduce the model")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **arrays)                     # uncompressed on purpose: the engine reads it raw
    n_params = sum(W.size + b.size for W, b in layers)
    print(f"[export] wrote {out}  ({n_params} parameters, threshold={meta['threshold']:.6f})")

    if args.data:
        from ingest.can_loader import load_dataset
        from ingest.features import build_feature_matrix
        from model.common import split_indices

        df = load_dataset(args.data)
        X, _ = build_feature_matrix(df)
        labels = df["label"].to_numpy()
        _, test_idx = split_indices(labels)
        Xte = np.ascontiguousarray(X[test_idx], dtype=np.float32)
        err = ae.reconstruction_error(scaler.transform(Xte)).astype(np.float32)
        is_attack = (labels[test_idx] != "normal").astype(np.int8)
        np.savez(ref_out, X=Xte, err=err, is_attack=is_attack)
        print(f"[export] wrote {ref_out}  ({len(Xte):,} test-split frames from {args.data})")


if __name__ == "__main__":
    main()
