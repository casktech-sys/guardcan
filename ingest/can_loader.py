"""Load CAN traffic from the HCRL Car-Hacking dataset (or the synthetic stand-in).

Handles the dataset's two shapes:
  * attack CSVs  -> comma-separated, ragged (flag R/T is the last field)
  * normal_run   -> either CSV (synthetic) or the real "Timestamp: .. ID: .. DLC: .. b b b" text

Public API:
  load_file(path, attack_label)   -> DataFrame for one file
  load_dataset(path)              -> DataFrame for a whole directory, labelled
Columns: timestamp, can_id, dlc, b0..b7, label, is_attack, recording

`recording` names the capture a frame came from (the file name; a file that goes silent for more than
MAX_GAP_S is split into "<name>", "<name>#1", ...). The files are separate captures whose time ranges
can overlap, so timing features must be computed per recording, never over the merged stream.
"""
import os
import glob
import numpy as np
import pandas as pd

BYTE_COLS = [f"b{i}" for i in range(8)]
COLUMNS = ["timestamp", "can_id", "dlc"] + BYTE_COLS + ["label", "is_attack", "recording"]
FRAME_KEY = ["timestamp", "can_id", "dlc"] + BYTE_COLS + ["is_attack"]   # identifies one captured frame
MAX_GAP_S = 1.0     # a longer silence ends a capture (normal traffic has a frame every ~0.25 ms)

# filename hint -> attack label for injected (flag T) rows
_LABEL_HINTS = [
    ("normal", "normal"), ("dos", "dos"), ("fuzz", "fuzzy"),
    ("gear", "gear"), ("rpm", "rpm"), ("spoof", "spoof"),
]


def _label_from_name(fname):
    low = os.path.basename(fname).lower()
    for hint, label in _LABEL_HINTS:
        if hint in low:
            return label
    return "attack"


def _parse_text_line(line):
    """Real normal_run_data.txt: 'Timestamp: <t>  ID: <hex>  <x>  DLC: <n>  b b ...'."""
    toks = line.replace(",", " ").split()
    try:
        ts = float(toks[toks.index("Timestamp:") + 1])
        cid = int(toks[toks.index("ID:") + 1], 16)
        di = toks.index("DLC:")
        dlc = int(toks[di + 1])
        data = [int(x, 16) for x in toks[di + 2: di + 2 + dlc]]
    except (ValueError, IndexError):
        return None
    return ts, cid, dlc, data, "R"


def _parse_csv_line(line):
    """Ragged CSV: ts, id(hex), dlc, data..., flag.  Flag is the last token."""
    toks = [t.strip() for t in line.split(",") if t.strip() != ""]
    if len(toks) < 4:
        return None
    try:
        ts = float(toks[0])
        cid = int(toks[1], 16)
        dlc = int(toks[2])
        data = [int(x, 16) for x in toks[3: 3 + dlc]]
    except ValueError:
        return None
    flag = toks[3 + dlc] if len(toks) > 3 + dlc else "R"
    flag = flag if flag in ("R", "T") else "R"
    return ts, cid, dlc, data, flag


def _split_at_gaps(df, name):
    """Name each frame's capture: <name>, then <name>#1, #2, ... after every gap > MAX_GAP_S."""
    seg = np.concatenate([[0], np.cumsum(np.diff(df["timestamp"].to_numpy()) > MAX_GAP_S)])
    cats = [name] + [f"{name}#{k}" for k in range(1, int(seg[-1]) + 1)]
    df["recording"] = pd.Categorical.from_codes(seg.astype(np.int32), categories=cats)
    return df


_CHUNK = 1 << 19        # records parsed into Python lists before being packed into compact arrays


def load_file(path, attack_label=None):
    """One dataset file -> DataFrame with compact dtypes (the full HCRL set is 17.5M frames, so
    payload bytes are uint8 and label/recording are categorical rather than Python objects)."""
    if attack_label is None:
        attack_label = _label_from_name(path)
    chunks, ts, cid, dlc, data, atk = [], [], [], [], [], []

    def pack():
        if ts:
            b = np.array(data, dtype=np.int64)
            ok = ((b >= 0) & (b <= 0xFF)).all(axis=1)          # a byte outside 00..ff = corrupt row
            chunks.append((np.array(ts, np.float64)[ok], np.array(cid, np.int32)[ok],
                           np.array(dlc, np.int16)[ok], b[ok].astype(np.uint8), np.array(atk, bool)[ok]))
            for lst in (ts, cid, dlc, data, atk):
                lst.clear()

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _parse_text_line(line) if "Timestamp:" in line else _parse_csv_line(line)
            if rec is None:
                continue
            t, c, n, d, flag = rec
            ts.append(t); cid.append(c); dlc.append(n); atk.append(flag == "T")
            data.append((d + [0] * 8)[:8])                   # pad to 8 bytes
            if len(ts) >= _CHUNK:
                pack()
    pack()
    if not chunks:
        return pd.DataFrame(columns=COLUMNS)
    t, c, n, b, a = (np.concatenate(col) for col in zip(*chunks))
    cats = ["normal"] if attack_label == "normal" else ["normal", attack_label]
    label = pd.Categorical.from_codes(a.astype(np.int8) if len(cats) == 2 else np.zeros(len(a), np.int8), cats)
    df = pd.DataFrame({"timestamp": t, "can_id": c, "dlc": n, **{k: b[:, i] for i, k in enumerate(BYTE_COLS)},
                       "label": label, "is_attack": a})
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return _split_at_gaps(df, os.path.splitext(os.path.basename(path))[0])


def load_dataset(path, dedup=True):
    """Load a directory of dataset files into one labelled DataFrame, sorted by time.

    With dedup, a frame that also appears in an earlier file (same timestamp, ID, DLC, payload and
    flag) is kept only once. The published HCRL attack CSVs all end with the same 891,068-frame
    attack-free block, which would otherwise be counted four times and straddle the train/test split.
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")) +
                       glob.glob(os.path.join(path, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .csv/.txt dataset files found in {path}")
    frames = [load_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    for col in ("label", "recording"):                      # per-file categories -> one categorical
        df[col] = df[col].astype("category")
    if dedup:
        dup = df.duplicated(subset=FRAME_KEY, keep="first").to_numpy()
        if dup.any():
            src = df.loc[dup, "recording"].value_counts()
            print(f"[loader] dropped {int(dup.sum()):,} frames already present in an earlier file: "
                  + ", ".join(f"{r} {n:,}" for r, n in sorted(src.items()) if n))
            df = df[~dup]
            for col in ("label", "recording"):
                df[col] = df[col].cat.remove_unused_categories()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    df = load_dataset(d)
    print(f"Loaded {len(df):,} messages from {d}")
    print(df["label"].value_counts())
    print(df["recording"].value_counts().sort_index())
