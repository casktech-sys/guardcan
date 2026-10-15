"""Load CAN traffic from the HCRL Car-Hacking dataset (or the synthetic stand-in).

Handles the dataset's two shapes:
  * attack CSVs  -> comma-separated, ragged (flag R/T is the last field)
  * normal_run   -> either CSV (synthetic) or the real "Timestamp: .. ID: .. DLC: .. b b b" text

Public API:
  load_file(path, attack_label)   -> DataFrame for one file
  load_dataset(path)              -> DataFrame for a whole directory, labelled
Columns: timestamp, can_id, dlc, b0..b7, label, is_attack
"""
import os
import glob
import pandas as pd

BYTE_COLS = [f"b{i}" for i in range(8)]
COLUMNS = ["timestamp", "can_id", "dlc"] + BYTE_COLS + ["label", "is_attack"]

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


def load_file(path, attack_label=None):
    if attack_label is None:
        attack_label = _label_from_name(path)
    recs = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _parse_text_line(line) if "Timestamp:" in line else _parse_csv_line(line)
            if rec is None:
                continue
            ts, cid, dlc, data, flag = rec
            data = (data + [0] * 8)[:8]                      # pad to 8 bytes
            is_attack = (flag == "T")
            label = attack_label if is_attack else "normal"
            recs.append([ts, cid, dlc, *data, label, is_attack])
    if not recs:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(recs, columns=COLUMNS)


def load_dataset(path):
    """Load a directory of dataset files into one labelled DataFrame, sorted by time."""
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.csv")) +
                       glob.glob(os.path.join(path, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .csv/.txt dataset files found in {path}")
    frames = [load_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    df = load_dataset(d)
    print(f"Loaded {len(df):,} messages from {d}")
    print(df["label"].value_counts())
