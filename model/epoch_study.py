"""How many autoencoder epochs does a dataset need? Measured on a validation set, not guessed.

Trains the PyTorch autoencoder exactly as train_autoencoder.py does (same features, scaler, model, optimiser,
batch, seed) but on 90% of the training split's normal frames. After selected epochs it scores a validation
set that never touches the test split:
  * the held-out 10% of the training split's normal frames (validation loss, false-positive rate), and
  * every attack frame of the training split (the autoencoder trains on normal traffic only).
The threshold at each checkpoint is the p99.5 of the fitted frames' error, as in train_autoencoder.py.
Precision / F1 here use the validation class balance, so compare their trend, not their level, with eval.py.

    python -m model.epoch_study --data data/car-hacking --epochs 200 --out bench/hcrl/epoch_study.json

The JSON is rewritten after every checkpoint, so a long run can be watched and a partial run keeps its data.
"""
import os
import json
import time
import argparse
import numpy as np

from sklearn.metrics import roc_auc_score

from ingest.can_loader import load_dataset
from ingest.features import build_feature_matrix, FeatureScaler, N_FEATURES
from model.common import SPLITS, make_split, peak_memory
from model.autoencoder import TorchAutoencoder

CHECKPOINTS = "1,2,3,5,10,15,20,30,40,50,75,100,125,150,175,200"


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)                       # never leave a half-written file behind


def main(argv=None):
    ap = argparse.ArgumentParser(description="GuardCAN autoencoder: validation metrics vs training epochs.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--checkpoints", default=CHECKPOINTS, help="comma-separated epochs to evaluate")
    ap.add_argument("--split", default="random", choices=SPLITS)
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--val-frac", type=float, default=0.10, help="share of training normals held out")
    ap.add_argument("--out", default="bench/hcrl/epoch_study.json")
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from an existing --out JSON")
    args = ap.parse_args(argv)
    if args.plot_only:
        return plot(json.load(open(args.out)), os.path.splitext(args.out)[0] + ".png")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",") if int(c) <= args.epochs} | {args.epochs})

    df = load_dataset(args.data)
    X, _ = build_feature_matrix(df)
    labels = df["label"].to_numpy()
    train_idx, _ = make_split(df, args.split)
    del df
    is_normal = labels == "normal"
    tr_normal, tr_attack = train_idx[is_normal[train_idx]], train_idx[~is_normal[train_idx]]
    rng = np.random.default_rng(42)
    held = rng.random(len(tr_normal)) < args.val_frac
    fit_idx, val_normal = tr_normal[~held], tr_normal[held]

    scaler = FeatureScaler().fit(X[fit_idx])
    Xfit = scaler.transform(X[fit_idx])
    Xvn, Xva = scaler.transform(X[val_normal]), scaler.transform(X[tr_attack])
    va_label = labels[tr_attack]
    del X
    types = sorted(set(va_label))
    print(f"[epoch_study] fit on {len(fit_idx):,} normal frames; validate on {len(val_normal):,} held-out normal "
          f"+ {len(tr_attack):,} attack frames ({', '.join(f'{t} {int((va_label == t).sum()):,}' for t in types)})")

    ae = TorchAutoencoder(N_FEATURES, epochs=args.epochs)
    import torch
    result = {"data": args.data, "split": args.split, "percentile": args.percentile, "epochs": args.epochs,
              "fit_frames": int(len(fit_idx)), "val_normal_frames": int(len(val_normal)),
              "val_attack_frames": {t: int((va_label == t).sum()) for t in types},
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "status": "running", "train_mse": [], "epoch_seconds": [],
              "checkpoints": []}
    clock = {"t": time.perf_counter()}

    def on_epoch(ep, mse):
        result["epoch_seconds"].append(round(time.perf_counter() - clock["t"], 2))
        result["train_mse"].append(float(mse))
        if ep in checkpoints:
            thr = float(np.percentile(ae.reconstruction_error(Xfit), args.percentile))
            evn, eva = ae.reconstruction_error(Xvn), ae.reconstruction_error(Xva)
            fp, tp = int((evn > thr).sum()), int((eva > thr).sum())
            prec = tp / max(tp + fp, 1); rec = tp / len(eva)
            auc = float(roc_auc_score(np.r_[np.zeros(len(evn)), np.ones(len(eva))], np.r_[evn, eva]))
            cp = {"epoch": ep, "train_seconds": round(sum(result["epoch_seconds"]), 1), "threshold": thr,
                  "val_loss_normal": float(evn.mean()), "fp_rate": fp / len(evn), "recall": rec,
                  "precision": prec, "f1": 2 * prec * rec / max(prec + rec, 1e-12), "roc_auc": auc,
                  "recall_per_type": {t: float((eva[va_label == t] > thr).mean()) for t in types}}
            result["checkpoints"].append(cp)
            _write(args.out, result)
            print(f"[epoch_study] epoch {ep:>3}  {cp['train_seconds']:>7.0f}s  val_loss={cp['val_loss_normal']:.5f}  "
                  f"recall={rec:.4f}  fp_rate={cp['fp_rate']:.4%}  auc={auc:.4f}  "
                  + "  ".join(f"{t}={v:.3f}" for t, v in cp["recall_per_type"].items()), flush=True)
        clock["t"] = time.perf_counter()        # evaluation time is not training time

    ae.fit(Xfit, on_epoch=on_epoch)
    result.update(status="done", **peak_memory())
    _write(args.out, result)
    plot(result, os.path.splitext(args.out)[0] + ".png")
    print(f"[epoch_study] wrote {args.out}")


def plot(result, path):
    """Small multiples (one measure per panel, shared epoch axis): validation loss, recall + AUC, FP rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model.eval import SURFACE, INK, INK2, MUTED, GRID, SLOTS
    cps = result["checkpoints"]
    ep = [c["epoch"] for c in cps]
    panels = [("validation loss (normal frames, MSE)", [("val loss", [c["val_loss_normal"] for c in cps])]),
              ("attack frames detected (recall)",
               [(t, [c["recall_per_type"][t] for c in cps]) for t in cps[0]["recall_per_type"]]
               + [("all attacks", [c["recall"] for c in cps])]),
              ("false-positive rate on held-out normal", [("FP rate", [c["fp_rate"] for c in cps])])]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, (title, series) in zip(axes, panels):
        for i, (name, ys) in enumerate(series):
            color = INK2 if name == "all attacks" else SLOTS[i % len(SLOTS)]
            ax.plot(ep, ys, color=color, lw=2, marker="o", ms=4, markeredgecolor=SURFACE, label=name,
                    ls="--" if name == "all attacks" else "-")
        ax.set_xscale("log"); ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_xticks([t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= max(ep)])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("epoch (log)")
        if len(series) > 1:
            ax.legend(fontsize=8)
    axes[2].set_ylim(0, 1.5 * max(c["fp_rate"] for c in cps))     # from 0: the threshold pins it near 0.5 %
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=2))
    fig.suptitle(f"Autoencoder epochs on {os.path.basename(os.path.normpath(result['data']))}: validation metrics "
                 f"({result['fit_frames']:,} fitted frames, {result['split']} split, {result['device']})",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[epoch_study] wrote {path}")


if __name__ == "__main__":
    main()
