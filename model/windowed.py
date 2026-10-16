"""Event-level detection over time windows: per-ID count envelopes and a K-of-N alert rule.

Model (WindowedDetector). For every CAN ID seen in attack-free training windows, the envelope hi[k] is the most
frames that ID sent in any one of those 100 ms windows (ingest/windows.py). IDs never seen there share one key with
envelope 0. A window is anomalous for key k when counts[w, k] > hi[k]: more frames of that ID than normal training
traffic ever had in a window. The score log2((c + 1) / (hi + 1)) is that excess in octaves (> 0 = anomalous). It
ranks windows for ROC-AUC and says how far outside the envelope an alert went.

Only the upper side is checked: frame injection adds frames. A missing-frames check would need a lower envelope.
In HCRL's attack captures the frame counts of attack-free windows often dip (a 10 ms ID drops to 0-8 frames per
window between bursts), so a lower envelope learned from them would be 0 for most IDs. The dataset also has no
suspension attack to validate one against.

Alert rule (raise_events). Per key, an event is raised at the end of a window when at least K of the last N windows
of that key (this one included) were anomalous and no event is open on that key. While an event is open, anomalous
windows extend it and raise nothing new; it closes after QUIET windows without one. Defaults, fixed before any
evaluation: K = 3, N = 5, QUIET = 10, i.e. 300 ms of excess within 500 ms raises an event and 1 s of quiet ends it.
  * K = 3: a frame that arrives late moves from window w to w + 1, so it can push at most one window over the
    envelope. A burst of retransmissions or a phase slip touches one or two adjacent windows. Three windows
    need at least 200 ms of excess traffic, i.e. a sustained source.
  * N = 5 (T = 500 ms): tolerates two quiet windows inside an irregular flood. With K = 3 the alert comes 200-400 ms
    after the excess starts. That is 2-4 window periods, and well below the shortest HCRL burst (3.4 s).
  * QUIET = 1 s: an attack that pauses for under a second stays one event, so the V-SOC is not flooded with
    duplicates; longer pauses start a new event.
"""
from collections import deque

import numpy as np
import joblib

from ingest.windows import WINDOW_S, id_keys

K_WINDOWS, N_WINDOWS, QUIET_WINDOWS = 3, 5, 10
EPISODE_GAP_S = 1.0       # injected frames more than this apart belong to different attack episodes


class WindowedDetector:
    def __init__(self, width=WINDOW_S, k=K_WINDOWS, n=N_WINDOWS, quiet=QUIET_WINDOWS):
        self.width, self.k, self.n, self.quiet = width, k, n, quiet
        self.known_ids = None
        self.hi = None

    @property
    def n_keys(self):
        return len(self.known_ids) + 1                     # + the shared unknown-ID key

    def fit(self, can_id, windows, use):
        """Learn the envelope from the windows where use[w] is True (attack-free training windows)."""
        in_use = use[windows.win]
        self.known_ids = np.unique(np.asarray(can_id)[in_use])
        counts = self.counts(can_id, windows)[use]
        self.hi = counts.max(axis=0).astype(np.int32)
        assert self.hi[-1] == 0                            # no unknown IDs in the windows it was learned from
        self.fit_windows = int(use.sum())
        return self

    def keys(self, can_id):
        return id_keys(can_id, self.known_ids)

    def counts(self, can_id, windows, mask=None):
        return windows.count(self.keys(can_id), self.n_keys, mask)

    def score(self, counts):
        return np.log2((counts + 1.0) / (self.hi + 1.0)).astype(np.float32)

    def anomalous(self, counts):
        return counts > self.hi

    def key_names(self):
        return [f"0x{i:03x}" for i in self.known_ids] + ["unknown ID"]

    def save(self, path):
        joblib.dump({"width": self.width, "k": self.k, "n": self.n, "quiet": self.quiet,
                     "known_ids": self.known_ids, "hi": self.hi, "fit_windows": self.fit_windows}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        det = cls(d["width"], d["k"], d["n"], d["quiet"])
        det.known_ids, det.hi, det.fit_windows = d["known_ids"], d["hi"], d["fit_windows"]
        return det


def raise_events(anom, windows, k=K_WINDOWS, n=N_WINDOWS, quiet=QUIET_WINDOWS, score=None):
    """Apply the alert rule to anom[w, key] (bool). -> dict of per-event arrays:
    key, seg, raise_w (window whose end raised it), raise_t, end_t (end of its last anomalous window),
    anomalous_windows, peak (max score over the event's windows, if score is given)."""
    out = {f: [] for f in ("key", "seg", "raise_w", "raise_t", "end_t", "anomalous_windows", "peak")}

    def close(ev):
        for f, v in ev.items():
            out[f].append(v)

    for key in range(anom.shape[1]):
        idx = np.flatnonzero(anom[:, key])
        ev, last, recent = None, None, deque()
        for w in idx:
            s = windows.seg[w]
            if ev is not None and (s != ev["seg"] or w - last > quiet):
                close(ev); ev = None; recent.clear()
            if ev is not None:                             # ongoing event: extend it, raise nothing
                last = w
                ev["end_t"] = windows.end[w]; ev["anomalous_windows"] += 1
                if score is not None:
                    ev["peak"] = max(ev["peak"], float(score[w, key]))
                continue
            while recent and (recent[0] <= w - n or windows.seg[recent[0]] != s):
                recent.popleft()
            recent.append(w)
            if len(recent) >= k:
                ev = {"key": key, "seg": s, "raise_w": w, "raise_t": windows.end[w], "end_t": windows.end[w],
                      "anomalous_windows": len(recent),
                      "peak": max(float(score[j, key]) for j in recent) if score is not None else np.nan}
                last = w
                recent.clear()
        if ev is not None:
            close(ev)
    ev = {f: np.asarray(v) for f, v in out.items()}
    order = np.argsort(ev["raise_t"], kind="stable")
    return {f: v[order] for f, v in ev.items()}


def attack_episodes(t, is_attack, rec, label, gap=EPISODE_GAP_S):
    """Injected frames -> attack episodes: runs of injected frames of one recording, no two more than gap s apart.
    t sorted within each recording. -> dict of arrays: rec, label, start, end, frames."""
    t, is_attack, rec, label = map(np.asarray, (t, is_attack, rec, label))
    out = {"rec": [], "label": [], "start": [], "end": [], "frames": []}
    for r in np.unique(rec[is_attack]):
        m = is_attack & (rec == r)
        ta, la = t[m], label[m]
        first = np.r_[True, np.diff(ta) > gap]
        b = np.flatnonzero(first)
        e = np.r_[b[1:], len(ta)] - 1
        out["rec"] += [r] * len(b); out["label"] += list(la[b])
        out["start"] += list(ta[b]); out["end"] += list(ta[e]); out["frames"] += list(e - b + 1)
    return {f: np.asarray(v) for f, v in out.items()}


def poisson_upper(n, exposure, conf=0.95):
    """One-sided upper confidence bound on a Poisson rate, after n events in `exposure` (Garwood / chi-square)."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (n + 1)) / 2 / exposure) if exposure > 0 else float("nan")


def assess(alert_seg, alert_t, episodes, segs, width=WINDOW_S):
    """Score alerts against attack episodes.

    alert_seg, alert_t: segment code and raise time of every alert (an event, or a flagged frame for per-frame
    alarms). segs: dict of arrays code, rec, t0, t1, one entry per segment (time span of its frames).
    An alert detects an episode when it is raised in [start, end + width]: the last window that holds injected
    frames ends at most `width` after the last of them. An episode is scored in the segment that contains it
    entirely; one that straddles a segment boundary is scored nowhere, and alerts inside it count neither as
    detections nor as false alarms. Every other alert is a false alarm.

    -> episode dict (seg: -1 if it straddles; detected, first_alert_t, latency_s, alerts) and segment dict
       (false_alarms, attack_free_s: the segment's duration outside every [start, end + width] episode span).
    """
    alert_seg, alert_t = np.asarray(alert_seg), np.asarray(alert_t, dtype=np.float64)
    codes = np.asarray(segs["code"])
    order = np.argsort(codes)
    seg_pos = order[np.searchsorted(codes[order], alert_seg)] if len(alert_seg) else np.zeros(0, np.int64)
    alert_rec = np.asarray(segs["rec"])[seg_pos]
    n_ep = len(episodes["start"])
    ep_seg = np.full(n_ep, -1, dtype=np.int64)
    for i in range(n_ep):
        inside = np.flatnonzero((segs["rec"] == episodes["rec"][i]) & (segs["t0"] <= episodes["start"][i])
                                & (episodes["end"][i] <= segs["t1"]))
        if len(inside):
            ep_seg[i] = codes[inside[0]]
    matched = np.full(len(alert_t), -1, dtype=np.int64)
    for r in np.unique(episodes["rec"]):
        ep = np.flatnonzero(episodes["rec"] == r)
        ep = ep[np.argsort(episodes["start"][ep])]
        a = np.flatnonzero(alert_rec == r)
        j = np.searchsorted(episodes["start"][ep], alert_t[a], side="right") - 1
        hit = (j >= 0) & (alert_t[a] <= episodes["end"][ep[np.maximum(j, 0)]] + width)
        matched[a[hit]] = ep[j[hit]]
    first = np.full(n_ep, np.inf)
    scored = (matched >= 0)
    np.minimum.at(first, matched[scored], alert_t[scored])
    n_alerts = np.bincount(matched[scored], minlength=n_ep)
    detected = np.isfinite(first) & (ep_seg >= 0)
    ep_out = {"seg": ep_seg, "detected": detected, "first_alert_t": np.where(detected, first, np.nan),
              "latency_s": np.where(detected, first - episodes["start"], np.nan), "alerts": n_alerts}
    fa = np.bincount(seg_pos[matched < 0], minlength=len(codes))
    free = np.asarray(segs["t1"], dtype=np.float64) - np.asarray(segs["t0"], dtype=np.float64)
    for s in range(len(codes)):
        r, t0, t1 = segs["rec"][s], segs["t0"][s], segs["t1"][s]
        m = episodes["rec"] == r
        lo = np.maximum(episodes["start"][m], t0); hi = np.minimum(episodes["end"][m] + width, t1)
        free[s] -= np.clip(hi - lo, 0, None).sum()
    return ep_out, {"code": codes, "false_alarms": fa, "attack_free_s": free}


PARTS = ("fit", "val", "test")


def time_segments(rec, t, test_mask, val_frac=0.2):
    """Split every recording into fit / val / test segments for the time split.

    test_mask marks the time split's test frames (the last 30 % of each recording). Of the rest, the frames in the
    last val_frac of each recording's training time are validation; the earlier ones fit the envelope. Segment
    code = 4 * recording code + part (0 fit, 1 val, 2 test). -> (seg per frame, segs dict: code, rec, part, t0, t1)
    """
    rec, t, test_mask = np.asarray(rec), np.asarray(t, dtype=np.float64), np.asarray(test_mask, dtype=bool)
    part = np.full(len(t), 2, dtype=np.int64)
    for r in np.unique(rec):
        m = (rec == r) & ~test_mask
        if m.any():
            t0, t1 = t[m].min(), t[m].max()
            part[m] = np.where(t[m] < t0 + (1.0 - val_frac) * (t1 - t0), 0, 1)
    seg = 4 * rec.astype(np.int64) + part
    codes, inv = np.unique(seg, return_inverse=True)
    t0 = np.full(len(codes), np.inf); t1 = np.full(len(codes), -np.inf)
    np.minimum.at(t0, inv, t); np.maximum.at(t1, inv, t)
    return seg, {"code": codes, "rec": codes // 4, "part": codes % 4, "t0": t0, "t1": t1}


def summarize(ep, sg, episodes, segs, rec_names, part):
    """Per-attack-type detection and per-recording false alarms for one part (0 fit, 1 val, 2 test)."""
    out = {"per_attack": {}, "false_alarms": {}}
    in_part = (ep["seg"] >= 0) & (ep["seg"] % 4 == part)
    for lab in sorted(set(episodes["label"][in_part])):
        m = in_part & (episodes["label"] == lab)
        lat = ep["latency_s"][m & ep["detected"]] * 1e3
        out["per_attack"][str(lab)] = {
            "episodes": int(m.sum()), "detected": int(ep["detected"][m].sum()),
            "fraction": float(ep["detected"][m].mean()),
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)),
                           "min": float(lat.min()), "max": float(lat.max())} if len(lat) else None,
            "alerts_per_episode": {"median": float(np.median(ep["alerts"][m])), "max": int(ep["alerts"][m].max())}}
    for i in np.flatnonzero(np.asarray(segs["part"]) == part):
        n, s = int(sg["false_alarms"][i]), float(sg["attack_free_s"][i])
        per_s = n / s if s > 0 else float("nan")          # a segment can be all attack (synthetic data)
        out["false_alarms"][rec_names[segs["rec"][i]]] = {
            "false_alarms": n, "attack_free_s": round(s, 2), "per_s": per_s, "per_hour": 3600 * per_s,
            "per_hour_95_upper": 3600 * poisson_upper(n, s)}
    return out
