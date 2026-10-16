"""Train the unsupervised anomaly detector on NORMAL traffic only.

The reconstruction error on normal traffic sets the detection threshold (a high percentile).
At eval time, messages whose error exceeds the threshold are flagged as anomalous — including
attack types the model never saw.

With PyTorch, the model and optimiser are checkpointed to <out>/ae_checkpoint.pt every 20 epochs, so a run that is
killed (e.g. the machine shuts down) continues with --resume instead of starting over.
"""
import os
import json
import time
import argparse
import numpy as np

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler, N_FEATURES
from model.common import SPLITS, make_split, peak_memory
from model.autoencoder import build_autoencoder


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train GuardCAN autoencoder (unsupervised).")
    ap.add_argument("--data", required=True, help="dataset dir (e.g. data/car-hacking or data/synthetic)")
    ap.add_argument("--out", default="models", help="output dir for model artifacts")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "sklearn"])
    # 200: on the synthetic set the torch AE is still improving at 30 epochs (train MSE 0.0182 vs 0.0158 at 200,
    # detection F1 0.39 vs 0.43). For HCRL see model/epoch_study.py and bench/hcrl/epoch_study.json.
    # sklearn's MLPRegressor treats this as max_iter and stops earlier once its loss converges.
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="normal-error percentile used as the anomaly threshold")
    ap.add_argument("--split", default="random", choices=SPLITS,
                    help="random: stratified rows (default); time: last 30%% of every capture is test")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="torch: save model + optimiser to <out>/ae_checkpoint.pt every N epochs (0 = never)")
    ap.add_argument("--resume", action="store_true",
                    help="torch: continue from <out>/ae_checkpoint.pt if it exists (same data, split and settings)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[autoencoder] loading {args.data}")
    t0 = time.perf_counter()
    df = load_dataset(args.data)
    X, names = build_feature_matrix(df)
    labels = df["label"].to_numpy()
    load_s = time.perf_counter() - t0

    train_idx, _ = make_split(df, args.split)
    is_normal = labels == "normal"
    train_normal = train_idx[is_normal[train_idx]]
    print(f"[autoencoder] {len(df):,} messages; training on {len(train_normal):,} normal frames")

    scaler = FeatureScaler().fit(X[train_normal])
    Xtn = scaler.transform(X[train_normal])

    ae, backend = build_autoencoder(args.backend, N_FEATURES, epochs=args.epochs)
    print(f"[autoencoder] backend = {backend}")
    t0 = time.perf_counter()
    resumed = {}
    if backend == "torch":
        ae.fit(Xtn, checkpoint=os.path.join(args.out, "ae_checkpoint.pt"), checkpoint_every=args.checkpoint_every,
               resume=args.resume)
        train_s = ae.prior_train_seconds + time.perf_counter() - t0       # all sessions of a resumed run
        if ae.start_epoch:
            resumed = {"resumed_from_epoch": ae.start_epoch}
    else:
        if args.resume:
            print("[autoencoder] --resume applies to the torch backend only (scikit-learn trains in one call)")
        ae.fit(Xtn)
        train_s = time.perf_counter() - t0
    print(f"[autoencoder] trained in {train_s:.1f} s ({load_s:.1f} s to load + build features)")

    err = ae.reconstruction_error(Xtn)
    threshold = float(np.percentile(err, args.percentile))
    print(f"[autoencoder] threshold (p{args.percentile}) = {threshold:.6f}")

    ae.save(os.path.join(args.out, "autoencoder.joblib"))
    scaler.save(os.path.join(args.out, "scaler.joblib"))
    with open(os.path.join(args.out, "ae_meta.json"), "w") as f:
        json.dump({"threshold": threshold, "percentile": args.percentile,
                   "backend": backend, "feature_names": names, "seed": 42, "split": args.split,
                   "epochs": args.epochs, "train_frames": int(len(train_normal)),
                   "train_seconds": round(train_s, 1), "load_seconds": round(load_s, 1),
                   **resumed, **peak_memory()}, f, indent=2)
    print(f"[autoencoder] saved to {args.out}/")


if __name__ == "__main__":
    main()
