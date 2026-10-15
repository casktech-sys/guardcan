"""Per-message feature engineering for CAN intrusion detection.

Each CAN message becomes a fixed-length vector:
  * 11 bits of the arbitration ID            -> flags never-before-seen / foreign IDs
  * DLC / 8                                   -> length anomalies
  * 8 payload bytes / 255                     -> payload anomalies (fuzzing, spoofed values)
  * inter-arrival time (global)               -> bus-level timing (DoS flooding)
  * inter-arrival time (same ID)              -> per-ID cadence (replay, masquerade doubling)

The two timing features are standardized with a FeatureScaler fitted on NORMAL traffic and
reused at eval time, so train/eval stay consistent.

Note: pandas here for clarity. For line-rate / large files, the same groupby + diff maps
directly onto RAPIDS cuDF (drop-in `import cudf as pd`) — that is the Week-2 HPC path.
"""
import numpy as np
import joblib

ID_BITS = 11
FEATURE_NAMES = (
    [f"idbit{i}" for i in range(ID_BITS)]
    + ["dlc"]
    + [f"b{i}" for i in range(8)]
    + ["dt_global", "dt_same"]
)
N_FEATURES = len(FEATURE_NAMES)                 # 22
_DT_IDX = [FEATURE_NAMES.index("dt_global"), FEATURE_NAMES.index("dt_same")]


def build_feature_matrix(df):
    """DataFrame (from can_loader) -> (X float32 [n, 22], feature_names).

    Timing columns are log1p-transformed but NOT yet standardized (that is the scaler's job).
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)

    cid = df["can_id"].to_numpy(dtype=np.int64)
    bits = np.zeros((n, ID_BITS), dtype=np.float32)
    for i in range(ID_BITS):
        bits[:, i] = ((cid >> i) & 1).astype(np.float32)   # lower 11 bits (standard CAN ID)

    dlc = (df["dlc"].to_numpy(dtype=np.float32) / 8.0).reshape(-1, 1)
    payload = df[[f"b{i}" for i in range(8)]].to_numpy(dtype=np.float32) / 255.0

    ts = df["timestamp"].to_numpy(dtype=np.float64)
    dt_global = np.clip(np.diff(ts, prepend=ts[0]), 0, None)

    dt_same = df.groupby("can_id")["timestamp"].diff().to_numpy(dtype=np.float64)
    dt_same = np.nan_to_num(dt_same, nan=0.0)
    dt_same = np.clip(dt_same, 0, None)

    dt_global = np.log1p(dt_global).astype(np.float32).reshape(-1, 1)
    dt_same = np.log1p(dt_same).astype(np.float32).reshape(-1, 1)

    X = np.concatenate([bits, dlc, payload, dt_global, dt_same], axis=1).astype(np.float32)
    assert X.shape[1] == N_FEATURES, (X.shape, N_FEATURES)
    return X, FEATURE_NAMES


class FeatureScaler:
    """Standardize the timing columns only (bits/bytes/dlc are already in [0,1])."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.dt_idx = _DT_IDX

    def fit(self, X):
        cols = X[:, self.dt_idx]
        self.mean_ = cols.mean(axis=0)
        self.std_ = cols.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1e-6
        return self

    def transform(self, X):
        X = X.copy()
        X[:, self.dt_idx] = (X[:, self.dt_idx] - self.mean_) / self.std_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        joblib.dump({"mean": self.mean_, "std": self.std_, "dt_idx": self.dt_idx}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        s = cls()
        s.mean_, s.std_, s.dt_idx = d["mean"], d["std"], d["dt_idx"]
        return s
