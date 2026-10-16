"""Fit the windowed detector (per-ID count envelopes) on attack-free training windows; check it on validation.

    python -m model.train_windowed --data data/car-hacking --out models/hcrl-time

Time split only. The detector reads consecutive windows of a capture, so its test set has to be a contiguous
stretch of time: the last 30 % of every recording (model/common.py). The random split holds out scattered frames
instead. Every 100 ms window would then contain training and test frames alike, so no window is held out.

Within each recording's training part, the first 80 % of its time fits the envelope and the last 20 % is
validation. Only attack-free windows (no injected frame) are fitted. The alert rule's K, N and QUIET are fixed in
model/windowed.py. Validation shows how the fixed detector does on traffic it was not fitted on, attack-free
(false events) and attacks (episodes detected), before the test split is touched. It writes windowed.joblib and
windowed_meta.json into --out.

--holdout normal_run_data is a cross-day check: the separate attack-free capture (recorded on another day) is
left out of fitting and validation entirely and is test from its first frame to its last.
"""
import os
import json
import time
import argparse
import numpy as np

from ingest.can_loader import load_dataset
from ingest.windows import WindowIndex
from model.common import SPLITS, make_split, peak_memory
from model.windowed import (WindowedDetector, raise_events, attack_episodes, assess, summarize, time_segments,
                            PARTS)


def load_windowed_inputs(data, holdout=None):
    """Load a dataset and the time split -> (df, dict of per-frame arrays used by the windowed detector).
    With holdout (a recording name), all of that recording is test, so nothing of it is fitted or validated on."""
    df = load_dataset(data)
    _, test_idx = make_split(df, "time")
    test = np.zeros(len(df), dtype=bool); test[test_idx] = True
    if holdout:
        if holdout not in df["recording"].cat.categories:
            raise SystemExit(f"[windowed] --holdout {holdout!r} is not a recording of {data}")
        test |= (df["recording"] == holdout).to_numpy()
    lab = df["label"]
    a = {"t": df["timestamp"].to_numpy(np.float64), "can_id": df["can_id"].to_numpy(np.int64),
         "rec": df["recording"].cat.codes.to_numpy(np.int64), "is_attack": df["is_attack"].to_numpy(bool),
         "label": np.asarray(lab.cat.categories)[lab.cat.codes.to_numpy()], "test": test,
         "rec_names": [str(c) for c in df["recording"].cat.categories]}
    return df, a


def print_summary(title, s):
    print(f"\n{title}\n" + "-" * len(title))
    print(f"  {'attack':<8}{'episodes':>9}{'detected':>9}{'latency p50':>13}{'p99':>8}{'max':>8}  (ms)")
    for lab, d in s["per_attack"].items():
        L = d["latency_ms"] or {}
        print(f"  {lab:<8}{d['episodes']:>9}{d['detected']:>9}{L.get('p50', np.nan):>13.0f}{L.get('p99', np.nan):>8.0f}"
              f"{L.get('max', np.nan):>8.0f}")
    print(f"  {'false alarms in':<18}{'events':>7}{'attack-free s':>15}{'per hour':>10}{'95% upper':>11}")
    for r, d in s["false_alarms"].items():
        print(f"  {r:<18}{d['false_alarms']:>7}{d['attack_free_s']:>15.1f}{d['per_hour']:>10.1f}"
              f"{d['per_hour_95_upper']:>11.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit GuardCAN's windowed (event-level) detector.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/hcrl-time")
    ap.add_argument("--split", default="time", choices=SPLITS, help="only 'time' is meaningful (see module doc)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="share of each recording's training time used for validation")
    ap.add_argument("--holdout", default=None,
                    help="cross-day check: a recording (e.g. normal_run_data) held out whole, as test only")
    args = ap.parse_args(argv)
    if args.split != "time":
        raise SystemExit("[windowed] needs the time split: under the random split every window mixes training and "
                         "test frames, so no window is held out")
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    df, a = load_windowed_inputs(args.data, args.holdout)
    del df
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    seg, segs = time_segments(a["rec"], a["t"], a["test"], args.val_frac)
    win = WindowIndex(a["t"], seg)
    part = win.seg % 4
    attack_w = win.any(a["is_attack"])
    fit_use = (part == 0) & ~attack_w
    det = WindowedDetector().fit(a["can_id"], win, fit_use)
    fit_s = time.perf_counter() - t0
    names = det.key_names()
    print(f"[windowed] {win.n_windows:,} windows of {det.width * 1e3:.0f} ms; envelope fitted on {det.fit_windows:,} "
          f"attack-free windows; {len(det.known_ids)} known IDs")
    print("  envelope (max frames per window): " + ", ".join(f"{n} {h}" for n, h in zip(names, det.hi)))

    # validation: the fitted detector on the val segments only
    counts = det.counts(a["can_id"], win)
    anom, score = det.anomalous(counts), det.score(counts)
    val_w = part == 1
    clean_val = val_w & ~attack_w
    ev = raise_events(anom & val_w[:, None], win, det.k, det.n, det.quiet, score)
    episodes = attack_episodes(a["t"], a["is_attack"], a["rec"], a["label"])
    ep, sg = assess(ev["seg"], ev["raise_t"], episodes, segs, det.width)
    val = summarize(ep, sg, episodes, segs, a["rec_names"], PARTS.index("val"))
    val["attack_free_windows"] = int(clean_val.sum())
    val["attack_free_windows_above_envelope"] = int(anom[clean_val].any(axis=1).sum())
    val["attack_free_cells_above_envelope"] = int(anom[clean_val].sum())
    above = anom[clean_val].sum(axis=0)
    val["attack_free_cells_above_envelope_by_key"] = {names[k]: int(n) for k, n in enumerate(above) if n}
    print_summary("Validation (last 20 % of each recording's training time)", val)
    print(f"  attack-free validation windows: {val['attack_free_windows']:,}, above the envelope on any ID: "
          f"{val['attack_free_windows_above_envelope']} ({val['attack_free_cells_above_envelope_by_key']})")

    det.save(os.path.join(args.out, "windowed.joblib"))
    meta = {"data": args.data, "split": "time", "val_frac": args.val_frac, "holdout": args.holdout,
            "window_ms": det.width * 1e3,
            "rule": {"k": det.k, "n": det.n, "quiet": det.quiet},
            "fit_windows": det.fit_windows, "envelope": dict(zip(names, det.hi.tolist())),
            "segments": {f"{a['rec_names'][r]}/{PARTS[p]}": [float(t0_), float(t1_)]
                         for r, p, t0_, t1_ in zip(segs["rec"], segs["part"], segs["t0"], segs["t1"])},
            "validation": val, "load_seconds": round(load_s, 1), "fit_seconds": round(fit_s, 1), **peak_memory()}
    with open(os.path.join(args.out, "windowed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[windowed] saved to {args.out}/windowed.joblib and windowed_meta.json")


if __name__ == "__main__":
    main()
