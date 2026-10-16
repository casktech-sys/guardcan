"""Evaluate the windowed detector on the time split's test part, next to the per-frame autoencoder baseline.

    python -m model.eval_windowed --data data/car-hacking --models models/hcrl-time --out bench/hcrl/windowed

For every detector it reports:
  * attack episodes detected per attack type (an episode = one burst of injected frames, model/windowed.py) and the
    detection latency from the episode's first injected frame to its first alert (p50 / p99 / max);
  * false alarms per second and per hour in each recording's attack-free test time, with a one-sided 95 % Poisson
    upper bound. The separate attack-free capture (normal_run_data) is the headline; its test part is 152 s, so
    even zero alarms there only bound the rate to below about 71 per hour;
  * frame-level precision / recall / F1 / ROC-AUC, for continuity with bench/hcrl/time_split.
Detectors:
  windowed          per-ID count envelope + alert rule (windowed.joblib from model/train_windowed.py)
  per-frame         the time-split autoencoder (autoencoder.joblib); every flagged frame is an alarm, as in
                    bench/hcrl/time_split
  per-frame + rule  the same flagged frames through the windowed alert rule: a window is anomalous for an ID when
                    any of its frames of that ID is flagged. This shows what the alert rule alone does.
Thresholds, K, N and QUIET all come from the saved models; nothing here is fitted on the test part.
"""
import os
import json
import argparse
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ingest.features import build_feature_matrix, FeatureScaler
from ingest.windows import WindowIndex
from model.common import make_split, peak_memory
from model.autoencoder import load_autoencoder
from model.windowed import (WindowedDetector, raise_events, attack_episodes, assess, summarize, time_segments,
                            poisson_upper, PARTS)
from model.train_windowed import load_windowed_inputs, print_summary

TEST = PARTS.index("test")


def frame_metrics(y, pred, score, label, rec, rec_names):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    normal = ~y
    out = {"frames": int(len(y)), "precision": float(p), "recall": float(r), "f1": float(f1),
           "roc_auc": float(roc_auc_score(y, score)), "fp_rate": float(pred[normal].mean()),
           "recall_per_attack": {str(c): float(pred[label == c].mean()) for c in sorted(set(label[y]))},
           "fp_rate_by_recording": {rec_names[g]: float(pred[normal & (rec == g)].mean())
                                    for g in sorted(set(rec[normal]))}}
    return out


def pooled_false_alarms(s, recordings):
    n = sum(s["false_alarms"][r]["false_alarms"] for r in recordings)
    secs = sum(s["false_alarms"][r]["attack_free_s"] for r in recordings)
    per_s = n / secs if secs > 0 else float("nan")
    return {"recordings": recordings, "false_alarms": n, "attack_free_s": round(secs, 2), "per_s": per_s,
            "per_hour": 3600 * per_s, "per_hour_95_upper": 3600 * poisson_upper(n, secs)}


def event_level(alert_seg, alert_t, episodes, segs, a, width):
    """Detection and false alarms on the test part. False alarms are also pooled over the attack-free captures
    (recordings without a single injected frame) and over the attack captures' time between episodes."""
    ep, sg = assess(alert_seg, alert_t, episodes, segs, width)
    s = summarize(ep, sg, episodes, segs, a["rec_names"], TEST)
    clean = {a["rec_names"][r] for r in np.unique(a["rec"])} - {a["rec_names"][r] for r in np.unique(episodes["rec"])}
    attack_free = [r for r in s["false_alarms"] if r in clean]
    s["false_alarms_pooled"] = {"attack_free_captures": pooled_false_alarms(s, attack_free),
                                "between_attack_episodes": pooled_false_alarms(
                                    s, [r for r in s["false_alarms"] if r not in attack_free]),
                                "all": pooled_false_alarms(s, list(s["false_alarms"]))}
    return s, ep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate GuardCAN's windowed detector against the per-frame baseline.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models/hcrl-time")
    ap.add_argument("--out", default="bench/hcrl/windowed")
    ap.add_argument("--plot-only", action="store_true", help="redraw the charts from an existing --out/metrics.json")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.plot_only:
        from model.plot_windowed import plot_all
        return plot_all(args.out)

    det = WindowedDetector.load(os.path.join(args.models, "windowed.joblib"))
    wmeta = json.load(open(os.path.join(args.models, "windowed_meta.json")))
    df, a = load_windowed_inputs(args.data, wmeta.get("holdout"))
    seg, segs = time_segments(a["rec"], a["t"], a["test"], wmeta["val_frac"])
    win = WindowIndex(a["t"], seg, det.width)
    test_w = (win.seg % 4) == TEST
    episodes = attack_episodes(a["t"], a["is_attack"], a["rec"], a["label"])
    te = a["test"]
    y, lab_te, rec_te = a["is_attack"][te], a["label"][te], a["rec"][te]
    res = {"data": {"path": args.data, "split": "time", "holdout": wmeta.get("holdout"),
                    "test_frames": int(te.sum()), "test_windows": int(test_w.sum()),
                    "window_ms": det.width * 1e3, "rule": wmeta["rule"],
                    "test_seconds": {a["rec_names"][r]: round(float(t1 - t0), 2) for r, p, t0, t1 in
                                     zip(segs["rec"], segs["part"], segs["t0"], segs["t1"]) if p == TEST}},
           "detectors": {}}

    # ---- windowed detector
    counts = det.counts(a["can_id"], win)
    anom, score = det.anomalous(counts), det.score(counts)
    ev = raise_events(anom & test_w[:, None], win, det.k, det.n, det.quiet, score)
    s, ep = event_level(ev["seg"], ev["raise_t"], episodes, segs, a, det.width)
    key = det.keys(a["can_id"])
    fw, fk = win.win[te], key[te]
    s["frame_level"] = frame_metrics(y, anom[fw, fk], score[fw, fk], lab_te, rec_te, a["rec_names"])
    s["events"] = _event_list(ev, det.key_names(), ep, episodes, segs, a, det.width)
    s["latency_ms_all"] = _latencies(ep, episodes)
    res["detectors"]["windowed"] = s
    timeline = _timeline(win, counts, det, ev, ep, episodes)
    del counts, anom, score
    if not wmeta.get("holdout"):
        s["rate_sensitivity"] = _rate_sensitivity(a, win, det, segs, episodes, test_w)

    # ---- per-frame autoencoder baseline (the time-split model), plain and through the alert rule
    ae_path = os.path.join(args.models, "autoencoder.joblib")
    if wmeta.get("holdout"):
        del df                                   # the autoencoder was not trained with this holdout
        print(f"[eval_windowed] holdout {wmeta['holdout']}: per-frame baseline skipped")
    elif os.path.exists(ae_path):
        meta = json.load(open(os.path.join(args.models, "ae_meta.json")))
        if meta.get("split") != "time":
            raise SystemExit(f"[eval_windowed] {ae_path} was trained with split={meta.get('split')!r}, not 'time'")
        X, _ = build_feature_matrix(df)
        _, test_idx = make_split(df, "time")
        assert np.array_equal(np.flatnonzero(te), test_idx)
        del df
        ae = load_autoencoder(ae_path)
        scaler = FeatureScaler.load(os.path.join(args.models, "scaler.joblib"))
        err = ae.reconstruction_error(scaler.transform(X[test_idx]))
        del X
        flagged = err > meta["threshold"]
        fl = test_idx[flagged]
        s, ep = event_level(seg[fl], a["t"][fl], episodes, segs, a, det.width)
        s["frame_level"] = frame_metrics(y, flagged, err, lab_te, rec_te, a["rec_names"])
        s["latency_ms_all"] = _latencies(ep, episodes)
        s["threshold"] = meta["threshold"]
        res["detectors"]["per_frame"] = s
        mask = np.zeros(len(a["t"]), dtype=bool); mask[fl] = True
        anom_b = det.counts(a["can_id"], win, mask) > 0
        ev_b = raise_events(anom_b & test_w[:, None], win, det.k, det.n, det.quiet)
        s, ep = event_level(ev_b["seg"], ev_b["raise_t"], episodes, segs, a, det.width)
        s["latency_ms_all"] = _latencies(ep, episodes)
        res["detectors"]["per_frame_rule"] = s
    else:
        del df
        print(f"[eval_windowed] no {ae_path}: per-frame baseline skipped")

    res.update(peak_memory())
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    if not wmeta.get("holdout"):
        np.savez_compressed(os.path.join(args.out, "timeline.npz"), **timeline)
    scope = "last 30 % of each recording" + (f", all of {wmeta['holdout']}" if wmeta.get("holdout") else "")
    for name, s in res["detectors"].items():
        print_summary(f"{name} — test split ({scope})", s)
        for k, d in s["false_alarms_pooled"].items():
            print(f"  pooled {k:<24}{d['false_alarms']:>7}{d['attack_free_s']:>15.1f}{d['per_hour']:>10.1f}"
                  f"{d['per_hour_95_upper']:>11.1f}")
        if "frame_level" in s:
            fm = s["frame_level"]
            print(f"  frame level: precision={fm['precision']:.3f} recall={fm['recall']:.3f} F1={fm['f1']:.3f} "
                  f"ROC-AUC={fm['roc_auc']:.4f} fp_rate={fm['fp_rate']:.4%}  recall/type: "
                  + " ".join(f"{c}={v:.3f}" for c, v in fm["recall_per_attack"].items()))
    for lab, d in res["detectors"]["windowed"].get("rate_sensitivity", {}).items():
        print(f"\nInjection-rate sensitivity, {lab} ({d['id']}, normal {d['normal_per_s']:.1f} frames/s, envelope "
              f"{d['envelope']}): derived attack, every m-th injected frame kept")
        print(f"  {'m':>4}{'injected/s':>12}{'detected':>12}{'latency p50':>13}")
        for r in d["rows"]:
            print(f"  {r['keep_every']:>4}{r['injected_per_s_median']:>12.1f}{r['detected']:>7}/{r['episodes']:<4}"
                  f"{r['latency_ms_p50'] if r['latency_ms_p50'] is not None else float('nan'):>13.0f}")
    print(f"\n[eval_windowed] wrote {args.out}/metrics.json")
    if not wmeta.get("holdout"):                 # the charts are for the main test split
        from model.plot_windowed import plot_all
        plot_all(args.out)


def _rate_sensitivity(a, win, det, segs, episodes, test_w, keep_every=(1, 2, 4, 8, 16, 32, 64, 128)):
    """How slow can a spoofing attack be and still be detected? A derived, weaker attack: of every gear / RPM
    test episode's injected frames only every m-th is kept; all other traffic is the recorded traffic. Only the
    spoofed ID's counts change, and they can only fall, so false alarms cannot rise and are not recomputed."""
    out = {}
    for lab in ("gear", "rpm"):
        inj = np.flatnonzero(a["is_attack"] & (a["label"] == lab))
        ids = np.unique(a["can_id"][inj])
        if len(ids) != 1:
            continue
        k = int(det.keys(ids)[0])
        e = np.flatnonzero(episodes["label"] == lab)
        e = e[np.argsort(episodes["start"][e])]
        t_inj = a["t"][inj]
        first = np.searchsorted(t_inj, episodes["start"][e])          # injected frames are in time order
        j = np.searchsorted(episodes["start"][e], t_inj, side="right") - 1
        rank = np.arange(len(inj)) - first[j]
        of_id = a["can_id"] == ids[0]
        fit_clean = ((win.seg % 4) == PARTS.index("fit")) & ~win.any(a["is_attack"])
        normal_per_s = (np.bincount(win.win[of_id], minlength=win.n_windows)[fit_clean].sum()
                        / (win.end - win.start)[fit_clean].sum())          # the ID's rate in attack-free fit windows
        rows = []
        for m in keep_every:
            mask = of_id & ~a["is_attack"]
            mask[inj[rank % m == 0]] = True
            c = np.bincount(win.win[mask], minlength=win.n_windows)
            anom = ((c > det.hi[k]) & test_w)[:, None]
            ev = raise_events(anom, win, det.k, det.n, det.quiet)
            ep, _ = assess(ev["seg"], ev["raise_t"], episodes, segs, det.width)
            sel = (episodes["label"] == lab) & (ep["seg"] >= 0) & (ep["seg"] % 4 == TEST)
            dur = episodes["end"][sel] - episodes["start"][sel]
            kept = np.ceil(episodes["frames"][sel] / m)
            lat = ep["latency_s"][sel & ep["detected"]] * 1e3
            rows.append({"keep_every": m, "injected_per_s_median": float(np.median(kept / dur)),
                         "episodes": int(sel.sum()), "detected": int(ep["detected"][sel].sum()),
                         "fraction": float(ep["detected"][sel].mean()),
                         "latency_ms_p50": float(np.median(lat)) if len(lat) else None,
                         "latency_ms_p99": float(np.percentile(lat, 99)) if len(lat) else None})
        out[lab] = {"id": f"0x{int(ids[0]):03x}", "envelope": int(det.hi[k]), "normal_per_s": float(normal_per_s),
                    "rows": rows}
    return out


def _latencies(ep, episodes):
    """Per test-episode latency in ms (nan if missed), grouped by attack type, for the charts."""
    out = {}
    m = (ep["seg"] >= 0) & (ep["seg"] % 4 == TEST)
    for lab in sorted(set(episodes["label"][m])):
        sel = m & (episodes["label"] == lab)
        out[str(lab)] = [None if np.isnan(v) else round(float(v) * 1e3, 1) for v in ep["latency_s"][sel]]
    return out


def _event_list(ev, names, ep, episodes, segs, a, width):
    """Every test event, with what it matched: the attack type of its episode, or 'false alarm'."""
    codes = np.asarray(segs["code"])
    rows = []
    for i in range(len(ev["raise_t"])):
        r = int(ev["seg"][i] // 4)
        m = np.flatnonzero((episodes["rec"] == r) & (episodes["start"] <= ev["raise_t"][i])
                           & (ev["raise_t"][i] <= episodes["end"] + width))
        what = "false alarm" if not len(m) else (str(episodes["label"][m[0]]) if ep["seg"][m[0]] >= 0
                                                 else "episode straddling the split")
        t0 = float(segs["t0"][np.flatnonzero(codes == ev["seg"][i])[0]])
        rows.append({"recording": a["rec_names"][r], "key": names[ev["key"][i]], "matched": what,
                     "raise_s_into_test": round(float(ev["raise_t"][i]) - t0, 3),
                     "duration_s": round(float(ev["end_t"][i] - ev["raise_t"][i]), 3),
                     "anomalous_windows": int(ev["anomalous_windows"][i]),
                     "peak_octaves": round(float(ev["peak"][i]), 2)})
    return rows


def _timeline(win, counts, det, ev, ep, episodes, before=3.0, after=8.0):
    """Window counts of the spoofed ID around the first gear and RPM test episodes, for the illustration chart."""
    out = {}
    for lab, cid in (("gear", 0x43f), ("rpm", 0x316)):
        first = np.flatnonzero((episodes["label"] == lab) & (ep["seg"] >= 0) & (ep["seg"] % 4 == TEST))
        if not len(first) or cid not in det.known_ids:
            continue
        i = first[0]
        t_ep, seg_i, k = episodes["start"][i], ep["seg"][i], int(np.searchsorted(det.known_ids, cid))
        sel = (win.seg == seg_i) & (win.start >= t_ep - before) & (win.end <= t_ep + after)
        evm = ((ev["seg"] == seg_i) & (ev["key"] == k)
               & (ev["raise_t"] >= t_ep - before) & (ev["raise_t"] <= t_ep + after))
        epm = ((episodes["rec"] == episodes["rec"][i]) & (episodes["end"] >= t_ep - before)
               & (episodes["start"] <= t_ep + after))
        out[f"{lab}_t"] = win.end[sel] - t_ep
        out[f"{lab}_count"] = counts[sel, k]
        out[f"{lab}_hi"] = np.array([det.hi[k]])
        out[f"{lab}_raise"] = ev["raise_t"][evm] - t_ep
        out[f"{lab}_ep"] = np.stack([episodes["start"][epm] - t_ep, episodes["end"][epm] - t_ep], 1)
        out[f"{lab}_id"] = np.array([cid])
    return out


if __name__ == "__main__":
    main()
