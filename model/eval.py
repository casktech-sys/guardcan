"""Evaluate both models on the shared held-out test split and write bench/metrics.json.

  * Detection (autoencoder): binary attack-vs-normal from reconstruction error > threshold,
    plus ROC-AUC using the raw error as the score.
  * Classification (supervised): per-class precision / recall / F1 and a confusion matrix.
"""
import os
import json
import argparse
import numpy as np

from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             classification_report, confusion_matrix)

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler
from model.common import split_indices
from model.autoencoder import load_autoencoder


def _bar(title):
    print("\n" + title + "\n" + "-" * len(title))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate GuardCAN models.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="bench")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.data)
    X, names = build_feature_matrix(df)
    labels = df["label"].to_numpy()
    _, test_idx = split_indices(labels)
    y_label = labels[test_idx]
    metrics = {}

    # ---- detection (autoencoder) ----
    ae = load_autoencoder(os.path.join(args.models, "autoencoder.joblib"))
    scaler = FeatureScaler.load(os.path.join(args.models, "scaler.joblib"))
    meta = json.load(open(os.path.join(args.models, "ae_meta.json")))
    thr = meta["threshold"]
    err = ae.reconstruction_error(scaler.transform(X[test_idx]))
    y_true = (y_label != "normal").astype(int)
    y_pred = (err > thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  pos_label=1, zero_division=0)
    auc = float(roc_auc_score(y_true, err)) if len(np.unique(y_true)) > 1 else float("nan")
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    metrics["detection"] = dict(precision=p, recall=r, f1=f1, roc_auc=auc,
                                threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn, backend=meta["backend"])

    _bar("Detection (autoencoder, attack vs normal)")
    print(f"  backend={meta['backend']}  threshold={thr:.6f}")
    print(f"  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ---- classification (supervised) ----
    import joblib
    clf = joblib.load(os.path.join(args.models, "classifier.joblib"))
    clf_scaler = FeatureScaler.load(os.path.join(args.models, "clf_scaler.joblib"))
    classes = json.load(open(os.path.join(args.models, "clf_classes.json")))["classes"]
    pred = clf.predict(clf_scaler.transform(X[test_idx]))

    rep = classification_report(y_label, pred, labels=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_label, pred, labels=classes)
    metrics["classification"] = {"report": rep, "confusion": {"labels": classes, "matrix": cm.tolist()}}

    _bar("Attack-type classification — per-class F1")
    print(f"  {'class':<10}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for c in classes:
        row = rep[c]
        print(f"  {c:<10}{row['precision']:>10.3f}{row['recall']:>9.3f}"
              f"{row['f1-score']:>8.3f}{int(row['support']):>9}")
    print(f"  {'accuracy':<10}{'':>10}{'':>9}{rep['accuracy']:>8.3f}{len(y_label):>9}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[eval] wrote {args.out}/metrics.json")

    _plots(args.out, err, y_true, cm, classes)


def _plots(out, err, y_true, cm, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return
    try:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, err)
            plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
            plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
            plt.title("Detection ROC"); plt.tight_layout()
            plt.savefig(os.path.join(out, "roc.png"), dpi=110); plt.close()
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="BuGn")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, fraction=0.046); plt.tight_layout()
        plt.savefig(os.path.join(out, "confusion.png"), dpi=110); plt.close()
        print(f"[eval] wrote {out}/roc.png and {out}/confusion.png")
    except Exception as e:
        print(f"[eval] plot skipped: {e}")


if __name__ == "__main__":
    main()
