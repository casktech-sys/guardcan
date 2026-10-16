"""Evaluate both models on the shared held-out test split and write bench/metrics.json.

  * Detection (autoencoder): binary attack-vs-normal from reconstruction error > threshold,
    plus ROC-AUC using the raw error as the score, recall per attack type, and the false
    positives on normal traffic broken down by capture.
  * Classification (supervised): per-class precision / recall / F1 and a confusion matrix.

The split (random rows or per-capture time holdout) is the one the models were trained with.
"""
import os
import json
import argparse
import numpy as np

from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             classification_report, confusion_matrix)

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler
from model.common import SPLITS, make_split
from model.autoencoder import load_autoencoder


def _bar(title):
    print("\n" + title + "\n" + "-" * len(title))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate GuardCAN models.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--split", default=None, choices=SPLITS,
                    help="default: the split recorded in the models' ae_meta.json")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(args.models, "ae_meta.json")))
    clf_meta = json.load(open(os.path.join(args.models, "clf_classes.json")))
    split = args.split or meta.get("split", "random")
    for name, m in (("autoencoder", meta), ("classifier", clf_meta)):
        if m.get("split", "random") != split:
            raise SystemExit(f"[eval] the {name} was trained with split={m.get('split', 'random')!r}, not {split!r}")

    df = load_dataset(args.data)
    X, names = build_feature_matrix(df)
    labels = df["label"].to_numpy()
    _, test_idx = make_split(df, split)
    y_label = labels[test_idx]
    rec = df["recording"].to_numpy()[test_idx]
    metrics = {"data": {"path": args.data, "split": split, "frames": int(len(df)), "test_frames": int(len(test_idx)),
                        "test_support": {c: int(n) for c, n in zip(*np.unique(y_label, return_counts=True))}}}

    # ---- detection (autoencoder) ----
    ae = load_autoencoder(os.path.join(args.models, "autoencoder.joblib"))
    scaler = FeatureScaler.load(os.path.join(args.models, "scaler.joblib"))
    thr = meta["threshold"]
    err = ae.reconstruction_error(scaler.transform(X[test_idx]))
    y_true = (y_label != "normal").astype(int)
    y_pred = (err > thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  pos_label=1, zero_division=0)
    auc = float(roc_auc_score(y_true, err)) if len(np.unique(y_true)) > 1 else float("nan")
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    normal = y_true == 0
    per_type = {}
    for c in sorted(set(y_label) - {"normal"}):
        m = y_label == c
        per_type[c] = dict(frames=int(m.sum()), detected=int(y_pred[m].sum()), recall=float(y_pred[m].mean()),
                           roc_auc_vs_normal=float(roc_auc_score(np.r_[np.zeros(normal.sum()), np.ones(m.sum())],
                                                                 np.r_[err[normal], err[m]])))
    fp_by_rec = {}
    for g in sorted(set(rec[normal])):
        m = normal & (rec == g)
        fp_by_rec[g] = dict(normal_frames=int(m.sum()), false_positives=int(y_pred[m].sum()),
                            fp_rate=float(y_pred[m].mean()))
    metrics["detection"] = dict(precision=p, recall=r, f1=f1, roc_auc=auc,
                                threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn, backend=meta["backend"],
                                fp_rate=fp / max(fp + tn, 1), per_attack_type=per_type,
                                false_positives_by_recording=fp_by_rec,
                                training={k: meta[k] for k in ("epochs", "train_frames", "train_seconds", "load_seconds",
                                                               "peak_host_rss_mb", "peak_gpu_allocated_mb") if k in meta})

    _bar("Detection (autoencoder, attack vs normal)")
    print(f"  backend={meta['backend']}  threshold={thr:.6f}  split={split}")
    print(f"  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  (false-positive rate on normal frames {fp / max(fp + tn, 1):.4%})")
    print(f"  {'attack type':<12}{'frames':>10}{'detected':>10}{'recall':>8}{'AUC vs normal':>15}")
    for c, d in per_type.items():
        print(f"  {c:<12}{d['frames']:>10}{d['detected']:>10}{d['recall']:>8.3f}{d['roc_auc_vs_normal']:>15.4f}")
    print(f"  {'normal frames in':<26}{'frames':>10}{'false pos.':>11}{'rate':>9}")
    for g, d in fp_by_rec.items():
        print(f"  {g:<26}{d['normal_frames']:>10}{d['false_positives']:>11}{d['fp_rate']:>9.4%}")

    # ---- classification (supervised) ----
    import joblib
    clf = joblib.load(os.path.join(args.models, "classifier.joblib"))
    clf_scaler = FeatureScaler.load(os.path.join(args.models, "clf_scaler.joblib"))
    classes = json.load(open(os.path.join(args.models, "clf_classes.json")))["classes"]
    pred = clf.predict(clf_scaler.transform(X[test_idx]))

    rep = classification_report(y_label, pred, labels=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_label, pred, labels=classes)
    metrics["classification"] = {"report": rep, "confusion": {"labels": classes, "matrix": cm.tolist()},
                                 "training": {k: clf_meta[k] for k in ("n_iter", "max_iter", "train_frames",
                                              "train_seconds", "load_seconds", "peak_host_rss_mb") if k in clf_meta}}

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

    _plots(args.out, err, y_true, y_label, thr, cm, classes, f"{os.path.basename(os.path.normpath(args.data))}, "
           f"{split} split, {len(test_idx):,} test frames")


# Reference palette (dataviz skill, light mode), as in engine/plot_results.py. Colour follows the attack type.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plots(out, err, y_true, y_label, thr, cm, classes, subtitle):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from sklearn.metrics import roc_curve
    except Exception:
        return
    try:
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                             "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
                             "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                             "axes.spines.right": False, "legend.frameon": False})
        if len(np.unique(y_true)) > 1:
            # one curve per attack type against all normal test frames; log FPR so the low-FP region is visible
            fig, ax = plt.subplots(figsize=(7.6, 4.4))
            normal = y_label == "normal"
            types = sorted(set(y_label) - {"normal"})
            floor = 1.0 / max(normal.sum(), 1)
            for i, c in enumerate(types):
                m = normal | (y_label == c)
                fpr, tpr, _ = roc_curve((y_label[m] == c).astype(int), err[m])
                op_fpr = float((err[normal] > thr).mean()); op_tpr = float((err[y_label == c] > thr).mean())
                ax.plot(np.maximum(fpr, floor), tpr, color=SLOTS[i % len(SLOTS)], lw=2,
                        label=f"{c} (AUC {roc_auc_score((y_label[m] == c).astype(int), err[m]):.3f})")
                ax.plot(max(op_fpr, floor), op_tpr, "o", ms=8, color=SLOTS[i % len(SLOTS)],
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.set_xscale("log"); ax.set_xlim(floor, 1); ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.set_xlabel("false-positive rate on normal frames (log)"); ax.set_ylabel("attack frames detected")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="dot = operating\nthreshold",
                      title_fontsize=8, alignment="left")
            fig.suptitle("Detection ROC per attack type", x=0.02, ha="left", fontweight="bold", fontsize=11)
            ax.set_title(subtitle, loc="left", fontsize=8.5, color=INK2)
            fig.tight_layout(); fig.savefig(os.path.join(out, "roc.png"), dpi=130); plt.close(fig)
        # colour = share of the true class (row-normalised), text = raw count
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.imshow(share, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if share[i, j] > 0.5 else INK)
        ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        fig.suptitle("Attack-type classifier: confusion matrix", x=0.02, ha="left", fontweight="bold", fontsize=11)
        ax.set_title(subtitle + "\ncell shade = share of the true class", loc="left", fontsize=8.5, color=INK2)
        fig.tight_layout(); fig.savefig(os.path.join(out, "confusion.png"), dpi=130); plt.close(fig)
        print(f"[eval] wrote {out}/roc.png and {out}/confusion.png")
    except Exception as e:
        print(f"[eval] plot skipped: {e}")


if __name__ == "__main__":
    main()
