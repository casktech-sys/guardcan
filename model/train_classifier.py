"""Train the supervised attack-type classifier (normal / dos / fuzzy / gear / rpm / ...).

Used for the SOC console's class breakdown and the per-class F1 table. Trains on the shared
train split; a quick report on the held-out test split is printed for sanity (the authoritative
evaluation is in eval.py, on the same test split).
"""
import os
import json
import argparse
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler
from model.common import split_indices


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train GuardCAN attack-type classifier (supervised).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[classifier] loading {args.data}")
    df = load_dataset(args.data)
    X, names = build_feature_matrix(df)
    labels = df["label"].to_numpy()

    train_idx, test_idx = split_indices(labels)
    scaler = FeatureScaler().fit(X[train_idx])
    Xtr, ytr = scaler.transform(X[train_idx]), labels[train_idx]
    Xte, yte = scaler.transform(X[test_idx]), labels[test_idx]

    classes = sorted(np.unique(labels).tolist())
    print(f"[classifier] classes = {classes}; training on {len(ytr):,} rows")
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                        max_iter=args.epochs, random_state=42)
    clf.fit(Xtr, ytr)

    print("[classifier] held-out test report:")
    print(classification_report(yte, clf.predict(Xte), zero_division=0))

    import joblib
    joblib.dump(clf, os.path.join(args.out, "classifier.joblib"))
    scaler.save(os.path.join(args.out, "clf_scaler.joblib"))
    with open(os.path.join(args.out, "clf_classes.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"[classifier] saved to {args.out}/")


if __name__ == "__main__":
    main()
