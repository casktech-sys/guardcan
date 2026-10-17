"""Export replay data for the V-SOC dashboard (dashboard/index.html) from the HCRL time split's test part.

    python -m dashboard.export_replay --data data/car-hacking --models models/hcrl-time \
        --metrics bench/hcrl/windowed --out dashboard/replay_data.js

Nothing is fitted or tuned here. The script loads the saved time-split models (model/train_windowed.py,
model/train_autoencoder.py --split time), runs both detectors on the test part exactly as model/eval_windowed.py
does, and stops unless it gets the same windowed events and the same per-frame false alarms as --metrics/metrics.json.
It then cuts five replay scenarios out of the test part (the last 30 % of each recording):

  dos, fuzzy, gear, rpm  one per attack capture, ~60 s of recorded traffic, whole attack episodes only. It starts
                         SETTLE after the end of the episode before the first one that lies wholly in the test part
                         (by then the windowed event of that earlier episode has closed) and ends SETTLE after the last
                         episode that still fits. HCRL injects a 3.4-6.0 s burst every ~9 s, so a scenario holds 6-7
                         episodes with ~3 s of attack-free traffic between them.
  normal                 the first 90 s of the test part of the separate attack-free capture (normal_run_data).

Per scenario it writes aggregates only, no frames or payloads:
  * frames per key and 100 ms window (keys = the 27 CAN IDs of normal training traffic + one shared unknown-ID key;
    the windows are the ones the detector used), and how many of those frames were injected (ground truth);
  * the attack episodes (ground-truth injection intervals) and, per episode, the latency of both detectors;
  * the windowed detector's events: raise time, key, matched episode, duration, peak excess over the ceiling, and,
    for events on the unknown-ID key, which IDs were behind it;
  * the per-frame autoencoder's alarms: flagged frames per key and window, and how many of them were injected.
Plus the learned ceiling per key and the headline numbers from metrics.json (whole test part, all 358 episodes).
The output is a JS file (window.GUARDCAN_REPLAY = {...}) rather than JSON, so the page also works when opened from
disk, where browsers block fetch() of local files.
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np

if __package__ in (None, ""):                        # run as a file: python dashboard/export_replay.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.features import build_feature_matrix, FeatureScaler
from ingest.windows import WindowIndex
from model.common import make_split, peak_memory
from model.autoencoder import load_autoencoder
from model.windowed import WindowedDetector, raise_events, attack_episodes, assess, time_segments, PARTS
from model.train_windowed import load_windowed_inputs

TEST = PARTS.index("test")
ATTACK_S, NORMAL_S = 60.0, 90.0                      # scenario lengths
ATTACKS = {"dos": "DoS flood", "fuzzy": "Fuzzing", "gear": "Gear spoofing", "rpm": "RPM spoofing"}
NORMAL_REC = "normal_run_data"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export recorded HCRL traffic and detector output for the dashboard.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models/hcrl-time")
    ap.add_argument("--metrics", default="bench/hcrl/windowed", help="directory of eval_windowed's metrics.json")
    ap.add_argument("--out", default="dashboard/replay_data.js")
    args = ap.parse_args(argv)

    det = WindowedDetector.load(os.path.join(args.models, "windowed.joblib"))
    wmeta = json.load(open(os.path.join(args.models, "windowed_meta.json")))
    if wmeta.get("holdout"):
        raise SystemExit(f"[export] {args.models} holds out {wmeta['holdout']}; use the plain time-split models")
    metrics = json.load(open(os.path.join(args.metrics, "metrics.json")))
    cross = os.path.join(args.metrics, "cross_day", "metrics.json")
    cross = json.load(open(cross)) if os.path.exists(cross) else None

    df, a = load_windowed_inputs(args.data)
    seg, segs = time_segments(a["rec"], a["t"], a["test"], wmeta["val_frac"])
    win = WindowIndex(a["t"], seg, det.width)
    test_w = (win.seg % 4) == TEST
    episodes = attack_episodes(a["t"], a["is_attack"], a["rec"], a["label"])
    key = det.keys(a["can_id"])

    # ---- windowed detector, as in eval_windowed
    counts = det.counts(a["can_id"], win)
    anom, score = det.anomalous(counts), det.score(counts)
    ev = raise_events(anom & test_w[:, None], win, det.k, det.n, det.quiet, score)
    ep, _ = assess(ev["seg"], ev["raise_t"], episodes, segs, det.width)
    del anom, score
    checks = {"windowed_events": _check_events(ev, segs, a, det, metrics)}
    injected = win.count(key, det.n_keys, a["is_attack"])

    # ---- per-frame autoencoder (time split): every flagged test frame is an alarm, as in eval_windowed
    meta = json.load(open(os.path.join(args.models, "ae_meta.json")))
    if meta.get("split") != "time":
        raise SystemExit(f"[export] the autoencoder in {args.models} was trained with split={meta.get('split')!r}")
    X, _ = build_feature_matrix(df)
    _, test_idx = make_split(df, "time")
    assert np.array_equal(np.flatnonzero(a["test"]), test_idx)
    del df
    ae = load_autoencoder(os.path.join(args.models, "autoencoder.joblib"))
    scaler = FeatureScaler.load(os.path.join(args.models, "scaler.joblib"))
    err = ae.reconstruction_error(scaler.transform(X[test_idx]))
    del X
    fl = test_idx[err > meta["threshold"]]
    ep_pf, sg_pf = assess(seg[fl], a["t"][fl], episodes, segs, det.width)
    checks["per_frame_false_alarms"] = _check_per_frame(sg_pf, segs, a, metrics)
    flagged = np.zeros(len(a["t"]), dtype=bool); flagged[fl] = True
    pf_all = win.count(key, det.n_keys, flagged)
    pf_inj = win.count(key, det.n_keys, flagged & a["is_attack"])
    gear_test = a["test"] & a["is_attack"] & (a["label"] == "gear")
    gear = {"injected_test_frames": int(gear_test.sum()), "flagged": int((gear_test & flagged).sum())}

    # ---- scenarios
    settle = (det.quiet + 2) * det.width             # an event closes `quiet` windows after its last anomalous one
    ctx = dict(a=a, win=win, segs=segs, det=det, episodes=episodes, ep=ep, ep_pf=ep_pf, ev=ev, key=key,
               counts=counts, injected=injected, pf_all=pf_all, pf_inj=pf_inj)
    scenarios = []
    for lab, title in ATTACKS.items():
        code, t_lo, t_hi = _attack_span(lab, episodes, ep, segs, settle)
        scenarios.append(_scenario(lab, title, code, t_lo, t_hi, ctx))
    r = a["rec_names"].index(NORMAL_REC)
    code = 4 * r + TEST
    t_lo = float(segs["t0"][np.flatnonzero(segs["code"] == code)[0]])
    scenarios.append(_scenario("normal", "Attack-free drive", code, t_lo, t_lo + NORMAL_S, ctx))

    out = {"generated_by": "dashboard/export_replay.py",
           "sources": {"data": "HCRL Car-Hacking dataset (data/car-hacking), time split: last 30 % of each recording",
                       "windowed_model": os.path.join(args.models, "windowed.joblib"),
                       "per_frame_model": os.path.join(args.models, "autoencoder.joblib"),
                       "metrics": os.path.join(args.metrics, "metrics.json")},
           "window_ms": round(det.width * 1e3, 3), "rule": {"k": det.k, "n": det.n, "quiet": det.quiet},
           "keys": det.key_names(), "ceiling": det.hi.tolist(), "ae_threshold": meta["threshold"],
           "headline": _headline(metrics, cross, gear),
           "scenarios": scenarios, "checks": checks}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("// Generated by dashboard/export_replay.py from recorded HCRL traffic and the saved detectors. "
                "Do not edit.\nwindow.GUARDCAN_REPLAY = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print(f"\n[export] wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB)")
    print(f"  {'scenario':<8}{'span (s into test)':>20}{'frames':>9}{'injected':>10}{'episodes':>9}{'events':>7}"
          f"{'false':>6}{'per-frame alarms':>17}{'on legit frames':>16}")
    for s in scenarios:
        n = s["totals"]
        print(f"  {s['id']:<8}{s['t_into_test']:>11.1f} – {s['t_into_test'] + s['duration_s']:<7.1f}"
              f"{n['frames']:>9}{n['injected']:>10}{n['episodes']:>9}{n['events']:>7}{n['false_events']:>6}"
              f"{n['pf_alarms']:>17}{n['pf_alarms_legit']:>16}")
    print(f"  {peak_memory()}")


def _check_events(ev, segs, a, det, metrics):
    """The windowed events must be the ones eval_windowed scored: same recording, key, raise time, windows."""
    codes, names = np.asarray(segs["code"]), det.key_names()
    mine = []
    for i in range(len(ev["raise_t"])):
        t0 = float(segs["t0"][np.flatnonzero(codes == ev["seg"][i])[0]])
        mine.append((a["rec_names"][ev["seg"][i] // 4], names[ev["key"][i]],
                     round(float(ev["raise_t"][i]) - t0, 3), int(ev["anomalous_windows"][i])))
    ref = [(e["recording"], e["key"], e["raise_s_into_test"], e["anomalous_windows"])
           for e in metrics["detectors"]["windowed"]["events"]]
    if mine != ref:
        raise SystemExit(f"[export] windowed events differ from metrics.json ({len(mine)} vs {len(ref)}); "
                         "rerun model.eval_windowed with these models first")
    print(f"[export] windowed detector: {len(mine)} test events, identical to metrics.json")
    return {"events": len(mine), "identical_to_metrics_json": True}


def _check_per_frame(sg, segs, a, metrics):
    """The per-frame alarms must give the false-alarm counts eval_windowed reported, per recording."""
    ref = metrics["detectors"]["per_frame"]["false_alarms"]
    got = {a["rec_names"][r]: int(n) for r, p, n in zip(segs["rec"], segs["part"], sg["false_alarms"]) if p == TEST}
    if got != {r: d["false_alarms"] for r, d in ref.items()}:
        raise SystemExit(f"[export] per-frame false alarms differ from metrics.json: {got}")
    print(f"[export] per-frame autoencoder: false alarms per recording identical to metrics.json ({got})")
    return {"false_alarms_by_recording": got, "identical_to_metrics_json": True}


def _attack_span(lab, episodes, ep, segs, settle):
    """(segment code, t_lo, t_hi) of an attack scenario: whole episodes only, as described in the module doc."""
    inside = np.flatnonzero((episodes["label"] == lab) & (ep["seg"] >= 0) & (ep["seg"] % 4 == TEST))
    i = inside[np.argmin(episodes["start"][inside])]
    code = int(ep["seg"][i])
    s = np.flatnonzero(segs["code"] == code)[0]
    same = np.flatnonzero(episodes["rec"] == episodes["rec"][i])
    same = same[np.argsort(episodes["start"][same])]
    j0 = int(np.flatnonzero(same == i)[0])
    t_lo = max(float(segs["t0"][s]), float(episodes["end"][same[j0 - 1]]) + settle if j0 else -np.inf)
    j = j0
    while (j + 1 < len(same) and episodes["end"][same[j + 1]] + settle - t_lo <= ATTACK_S
           and episodes["end"][same[j + 1]] <= segs["t1"][s]):
        j += 1
    t_hi = float(episodes["end"][same[j]]) + settle
    if j + 1 < len(same):
        t_hi = min(t_hi, float(episodes["start"][same[j + 1]]))
    return code, t_lo, min(t_hi, float(segs["t1"][s]))


def _scenario(sid, title, code, t_lo, t_hi, c):
    """Aggregates of the detector windows of segment `code` that cover [t_lo, t_hi)."""
    a, win, det, episodes, ev = c["a"], c["win"], c["det"], c["episodes"], c["ev"]
    ws = np.flatnonzero(win.seg == code)                       # contiguous, in time order
    seg_t0 = float(win.start[ws[0]])
    w0 = ws[0] + int(np.floor((t_lo - seg_t0) / det.width + 1e-9))
    w1 = min(ws[-1] + 1, ws[0] + int(np.ceil((t_hi - seg_t0) / det.width - 1e-9)))
    t0, t1 = float(win.start[w0]), float(win.end[w1 - 1])
    rec = code // 4
    rel = lambda t: round(float(t) - t0, 3)

    eps = np.flatnonzero((episodes["rec"] == rec) & (episodes["end"] >= t0) & (episodes["start"] < t1))
    eps = eps[np.argsort(episodes["start"][eps])]
    assert all(t0 <= episodes["start"][e] and episodes["end"][e] < t1 for e in eps), "partial episode in scenario"
    ep_rows = [{"start": rel(episodes["start"][e]), "end": rel(episodes["end"][e]), "label": str(episodes["label"][e]),
                "injected_frames": int(episodes["frames"][e]), "detected": bool(c["ep"]["detected"][e]),
                "latency_ms": _ms(c["ep"]["latency_s"][e]), "pf_latency_ms": _ms(c["ep_pf"]["latency_s"][e]),
                "pf_alarms": int(c["ep_pf"]["alerts"][e])} for e in eps]

    in_frames = (win.win >= w0) & (win.win < w1)              # the frames of exactly these windows
    unk = det.n_keys - 1
    unk_t, unk_id = a["t"][in_frames & (c["key"] == unk)], a["can_id"][in_frames & (c["key"] == unk)]
    names = det.key_names()
    ev_rows = []
    for i in np.flatnonzero((ev["seg"] == code) & (ev["raise_t"] >= t0) & (ev["raise_t"] < t1)):
        rt, et, k = float(ev["raise_t"][i]), float(ev["end_t"][i]), int(ev["key"][i])
        m = [j for j, e in enumerate(eps) if episodes["start"][e] <= rt <= episodes["end"][e] + det.width]
        rw = int(ev["raise_w"][i])
        span = slice(max(int(ws[0]), rw - det.n + 1), rw + int(round((et - rt) / det.width)) + 1)
        row = {"t": rel(rt), "end": rel(et), "close": rel(et + det.quiet * det.width), "key": names[k],
               "episode": m[0] if m else -1, "matched": str(episodes["label"][eps[m[0]]]) if m else "false alarm",
               "latency_ms": round((rt - float(episodes["start"][eps[m[0]]])) * 1e3, 1) if m else None,
               "anomalous_windows": int(ev["anomalous_windows"][i]), "peak_octaves": round(float(ev["peak"][i]), 2),
               "peak_count": int(c["counts"][span, k].max())}       # most frames of the key in one event window
        if k == unk:                                           # which never-seen IDs filled the unknown-ID key
            sel = (unk_t >= rt - det.n * det.width) & (unk_t <= et)
            ids = Counter(unk_id[sel].tolist())
            row["unknown_ids"] = {"distinct": len(ids), "frames": int(sel.sum()),
                                  "top": [[f"0x{i_:03x}", n] for i_, n in ids.most_common(3)]}
        ev_rows.append(row)

    sl = slice(w0, w1)
    pf_all, pf_inj = c["pf_all"][sl], c["pf_inj"][sl]
    nz = np.argwhere(pf_all > 0)
    inj = c["injected"][sl]
    nzi = np.argwhere(inj > 0)
    rec_name = a["rec_names"][rec]
    test_t0 = float(c["segs"]["t0"][np.flatnonzero(c["segs"]["code"] == code)[0]])
    totals = {"frames": int(in_frames.sum()), "injected": int((in_frames & a["is_attack"]).sum()),
              "episodes": len(ep_rows), "events": len(ev_rows),
              "false_events": sum(r["episode"] < 0 for r in ev_rows),
              "pf_alarms": int(pf_all.sum()), "pf_alarms_legit": int((pf_all - pf_inj).sum())}
    assert totals["frames"] == int(c["counts"][sl].sum())
    return {"id": sid, "title": title, "recording": rec_name, "capture_start_unix": round(t0, 3),
            "t_into_test": round(t0 - test_t0, 3), "duration_s": round(t1 - t0, 3), "n_windows": int(w1 - w0),
            "counts": c["counts"][sl].T.tolist(),              # [key][window]
            "injected": [[int(w), int(k), int(inj[w, k])] for w, k in nzi],
            "pf": [[int(w), int(k), int(pf_all[w, k]), int(pf_inj[w, k])] for w, k in nz],
            "episodes": ep_rows, "events": ev_rows, "totals": totals}


def _ms(v):
    return None if np.isnan(v) else round(float(v) * 1e3, 1)


def _headline(m, cross, gear):
    """The numbers the dashboard shows for the whole test part, straight from eval_windowed's metrics.json."""
    w, pf = m["detectors"]["windowed"], m["detectors"]["per_frame"]
    lat = [v for vs in w["latency_ms_all"].values() for v in vs if v is not None]
    pooled = w["false_alarms_pooled"]["all"]
    h = {"test_seconds": m["data"]["test_seconds"],
         "per_attack": {k: {"episodes": d["episodes"], "detected": d["detected"],
                            "latency_p50_ms": round(d["latency_ms"]["p50"], 1),
                            "latency_p99_ms": round(d["latency_ms"]["p99"], 1)} for k, d in w["per_attack"].items()},
         "episodes": sum(d["episodes"] for d in w["per_attack"].values()),
         "detected": sum(d["detected"] for d in w["per_attack"].values()),
         "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p99": round(float(np.percentile(lat, 99)), 1),
                        "max": round(float(np.max(lat)), 1), "episodes": len(lat)},
         "false_events": {"events": pooled["false_alarms"], "attack_free_s": pooled["attack_free_s"],
                          "per_hour_95_upper": round(pooled["per_hour_95_upper"], 1)},
         "false_events_normal_capture": {k: round(w["false_alarms"][NORMAL_REC][k], 2)
                                         for k in ("false_alarms", "attack_free_s", "per_hour_95_upper")},
         "per_frame": {"normal_capture_per_s": round(pf["false_alarms"][NORMAL_REC]["per_s"], 1),
                       "normal_capture_alarms": pf["false_alarms"][NORMAL_REC]["false_alarms"],
                       "normal_capture_s": pf["false_alarms"][NORMAL_REC]["attack_free_s"],
                       "all_attack_free_per_hour": round(pf["false_alarms_pooled"]["all"]["per_hour"]),
                       "all_attack_free_alarms": pf["false_alarms_pooled"]["all"]["false_alarms"],
                       "recall_per_attack": pf["frame_level"]["recall_per_attack"],
                       "gear_injected_test_frames": gear["injected_test_frames"],
                       "gear_injected_flagged": gear["flagged"]}}
    if cross:
        d = cross["detectors"]["windowed"]["false_alarms"][NORMAL_REC]
        h["cross_day"] = {"false_events": d["false_alarms"], "attack_free_s": d["attack_free_s"],
                          "per_hour_95_upper": round(d["per_hour_95_upper"], 1)}
    return h


if __name__ == "__main__":
    main()
