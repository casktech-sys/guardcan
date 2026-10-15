"""Train the unsupervised anomaly detector on NORMAL traffic only.

The reconstruction error on normal traffic sets the detection threshold (a high percentile).
At eval time, messages whose error exceeds the threshold are flagged as anomalous — including
attack types the model never saw.
"""
import os
import json
import argparse
import numpy as np

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler, N_FEATURES
from model.common import split_indices
from model.autoencoder import build_autoencoder


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train GuardCAN autoencoder (unsupervised).")
    ap.add_argument("--data", required=True, help="dataset dir (e.g. data/car-hacking or data/synthetic)")
    ap.add_argument("--out", default="models", help="output dir for model artifacts")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "sklearn"])
    # 200: at 30 epochs the torch AE is undertrained (synthetic set: detection F1 0.36 vs 0.90 at 200).
    # sklearn's MLPRegressor treats this as max_iter and stops earlier once its loss converges.
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="normal-error percentile used as the anomaly threshold")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[autoencoder] loading {args.data}")
    df = load_dataset(args.data)
    X, names = build_feature_matrix(df)
    labels = df["label"].to_numpy()

    train_idx, _ = split_indices(labels)
    is_normal = labels == "normal"
    train_normal = train_idx[is_normal[train_idx]]
    print(f"[autoencoder] {len(df):,} messages; training on {len(train_normal):,} normal frames")

    scaler = FeatureScaler().fit(X[train_normal])
    Xtn = scaler.transform(X[train_normal])

    ae, backend = build_autoencoder(args.backend, N_FEATURES, epochs=args.epochs)
    print(f"[autoencoder] backend = {backend}")
    ae.fit(Xtn)

    err = ae.reconstruction_error(Xtn)
    threshold = float(np.percentile(err, args.percentile))
    print(f"[autoencoder] threshold (p{args.percentile}) = {threshold:.6f}")

    ae.save(os.path.join(args.out, "autoencoder.joblib"))
    scaler.save(os.path.join(args.out, "scaler.joblib"))
    with open(os.path.join(args.out, "ae_meta.json"), "w") as f:
        json.dump({"threshold": threshold, "percentile": args.percentile,
                   "backend": backend, "feature_names": names, "seed": 42}, f, indent=2)
    print(f"[autoencoder] saved to {args.out}/")


if __name__ == "__main__":
    main()
