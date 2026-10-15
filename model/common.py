"""Shared helpers so the standalone scripts agree on the train/test split."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(labels, seed=42, test_frac=0.30):
    """Deterministic stratified split -> (train_idx, test_idx), both sorted.

    Every script calls this with the same seed, so the autoencoder, the classifier, and
    eval all use the same held-out test rows (no train/test leakage across scripts).
    """
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    return np.sort(tr), np.sort(te)
