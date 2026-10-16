"""Per-message feature engineering for CAN intrusion detection.

Each CAN message becomes a fixed-length vector:
  * 11 bits of the arbitration ID            -> flags never-before-seen / foreign IDs
  * DLC / 8                                   -> length anomalies
  * 8 payload bytes / 255                     -> payload anomalies (fuzzing, spoofed values)
  * inter-arrival time (global)               -> bus-level timing (DoS flooding)
  * inter-arrival time (same ID)              -> per-ID cadence (replay, masquerade doubling)

The two timing features are standardized with a FeatureScaler fitted on NORMAL traffic and
reused at eval time, so train/eval stay consistent. Both are computed within each capture (the loader's
`recording` column): the dataset files are separate recordings, and a diff across the merged stream
would interleave frames from different captures.

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
    X = np.empty((n, N_FEATURES), dtype=np.float32)          # filled column by column (no temporaries)

    cid = df["can_id"].to_numpy(dtype=np.int64)
    for i in range(ID_BITS):
        X[:, i] = (cid >> i) & 1                                # lower 11 bits (standard CAN ID)

    X[:, ID_BITS] = df["dlc"].to_numpy(dtype=np.float32) / 8.0
    for i in range(8):
        X[:, ID_BITS + 1 + i] = df[f"b{i}"].to_numpy(dtype=np.float32) / 255.0

    if "recording" not in df:
        df["recording"] = ""                                 # a single capture
    dt_global = df.groupby("recording", sort=False, observed=True)["timestamp"].diff().to_numpy(dtype=np.float64)
    dt_global = np.clip(np.nan_to_num(dt_global, nan=0.0), 0, None)   # first frame of a capture -> 0

    dt_same = df.groupby(["recording", "can_id"], sort=False, observed=True)["timestamp"].diff()
    dt_same = np.clip(np.nan_to_num(dt_same.to_numpy(dtype=np.float64), nan=0.0), 0, None)

    X[:, _DT_IDX[0]] = np.log1p(dt_global)
    X[:, _DT_IDX[1]] = np.log1p(dt_same)
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
