"""Shared helpers so the standalone scripts agree on the train/test split."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("random", "time")


def split_indices(labels, seed=42, test_frac=0.30):
    """Deterministic stratified split -> (train_idx, test_idx), both sorted.

    Every script calls this with the same seed, so the autoencoder, the classifier, and
    eval all use the same held-out test rows (no train/test leakage across scripts).
    """
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    return np.sort(tr), np.sort(te)


def time_split_indices(recordings, test_frac=0.30):
    """Per-capture time holdout -> (train_idx, test_idx): the last test_frac of every recording is test.

    Rows must be in time order (load_dataset sorts them). Unlike the random row split, no test frame has
    a neighbour from the same moment of the same capture in the training set.
    """
    rec = pd.Series(recordings).reset_index(drop=True)
    g = rec.groupby(rec, sort=False, observed=True)
    pos, size = g.cumcount().to_numpy(), g.transform("size").to_numpy()
    test = pos >= np.floor((1.0 - test_frac) * size)
    idx = np.arange(len(rec))
    return idx[~test], idx[test]


def make_split(df, mode="random", seed=42, test_frac=0.30):
    """(train_idx, test_idx) for a load_dataset() frame: 'random' (stratified rows) or 'time' (per capture)."""
    if mode == "random":
        return split_indices(df["label"].to_numpy(), seed, test_frac)
    if mode == "time":
        return time_split_indices(df["recording"], test_frac)
    raise ValueError(f"unknown split {mode!r}; expected one of {SPLITS}")


def peak_memory():
    """Peak memory of this process so far, in MiB: host RSS and, if PyTorch used CUDA, GPU allocations."""
    out = {}
    try:
        import resource
        out["peak_host_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # KiB on Linux
    except Exception:
        pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        out["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out
