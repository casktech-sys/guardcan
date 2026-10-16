"""Charts for bench/hcrl/windowed/ from its metrics.json and timeline.npz (written by model/eval_windowed.py).

    python -m model.eval_windowed --out bench/hcrl/windowed --plot-only --data unused

Each detector keeps its colour in every chart: windowed = slot 1, per-frame = slot 2, per-frame + rule = slot 3.
"""
import os
import json
import numpy as np

from model.eval import SURFACE, INK, INK2, MUTED, GRID, SLOTS

DETECTORS = [("windowed", "windowed (count envelope + rule)", SLOTS[0]),
             ("per_frame", "per-frame autoencoder", SLOTS[1]),
             ("per_frame_rule", "per-frame autoencoder + rule", SLOTS[2])]
ATTACKS = ATTACKS_ALL = [("dos", "DoS"), ("fuzzy", "fuzzy"), ("gear", "gear spoofing"), ("rpm", "RPM spoofing")]
TARGET_PER_HOUR = 1.0
LABEL_X = 7e3                  # latency chart: x of the per-row percentile labels          # the operational bar stated in docs/windowed-detection.md


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    return plt


def _title(fig, ax, title, subtitle):
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    ax.set_title(subtitle, loc="left", fontsize=8.5, color=INK2)


def _attack_key(per_attack, short):
    return next((k for k in per_attack if k.lower() == short), None)


def plot_all(out):
    res = json.load(open(os.path.join(out, "metrics.json")))
    dets = [(k, n, c) for k, n, c in DETECTORS if k in res["detectors"]]
    present = res["detectors"][dets[0][0]]["per_attack"]
    global ATTACKS
    ATTACKS = [(s, l) for s, l in ATTACKS if _attack_key(present, s)]     # attack types with test episodes
    if not ATTACKS:
        print("[plot_windowed] no attack episodes in the test part: charts skipped")
        return
    plt = _style()
    sub = (f"HCRL Car-Hacking, time split (last 30 % of each recording), {res['data']['window_ms']:.0f} ms windows, "
           f"alert: {res['data']['rule']['k']} of {res['data']['rule']['n']} windows")
    _per_attack(plt, out, res, dets, sub)
    _latency(plt, out, res, dets, sub)
    _false_alarms(plt, out, res, dets, sub)
    _frame_level(plt, out, res, dets, sub)
    if "rate_sensitivity" in res["detectors"].get("windowed", {}):
        _sensitivity(plt, out, res["detectors"]["windowed"]["rate_sensitivity"], sub)
    tl = os.path.join(out, "timeline.npz")
    if os.path.exists(tl):
        _timeline(plt, out, dict(np.load(tl)), sub)
    print(f"[plot_windowed] wrote charts to {out}/")


def _legend(ax, dets, **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=n) for _, n, c in dets], **kw)


def _fig_legend(fig, dets):
    """One row of detector swatches under the plot (charts whose right side carries labels)."""
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=n) for _, n, c in dets], loc="lower center", ncol=len(dets), fontsize=8)


def _bar(ax, x, h, color, w):
    """A column with the reference mark spec: <= 24 px, 4 px rounded data end (approximated by a cap-less bar)."""
    return ax.bar(x, h, width=w, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)


def _chance_note(res, k, short, label):
    """For a detector that flags none of an attack's injected frames yet has alarms in its episodes: why those
    alarms are chance (its background false-alarm rate in that capture), else None."""
    fm = res["detectors"][k].get("frame_level")
    recall = fm and fm["recall_per_attack"].get(_attack_key(fm["recall_per_attack"], short))
    if recall is None or recall > 0:
        return None
    rec = {e["matched"]: e["recording"] for e in res["detectors"].get("windowed", {}).get("events", [])}.get(
        _attack_key(res["detectors"][k]["per_attack"], short))
    fa = res["detectors"][k]["false_alarms"].get(rec) if rec else None
    rate = f" at {fa['per_s']:.1f}/s in this capture" if fa else ""
    return f"chance: 0 injected {label} frames flagged; alarms on legitimate frames{rate}"


def _per_attack(plt, out, res, dets, sub):
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    h = 0.8 / len(dets)
    for j, (k, name, color) in enumerate(dets):
        pa = res["detectors"][k]["per_attack"]
        for i, (short, label) in enumerate(ATTACKS):
            d = pa[_attack_key(pa, short)]
            y = i + (j - (len(dets) - 1) / 2) * h
            chance = _chance_note(res, k, short, label) if d["detected"] else None
            ax.barh(y, d["fraction"], height=h, color=color, alpha=0.35 if chance else 1.0, edgecolor=SURFACE,
                    linewidth=2, zorder=2)
            ax.text(d["fraction"] + 0.01, y, f"{d['detected']}/{d['episodes']}", va="center", fontsize=7.5, color=INK2)
            if chance:                                 # faded bar + ink label inside it: not a detection
                ax.text(0.015, y, chance, va="center", fontsize=7.5, color=INK, zorder=3)
    ax.set_yticks(range(len(ATTACKS))); ax.set_yticklabels([l for _, l in ATTACKS]); ax.invert_yaxis()
    ax.set_xlim(0, 1.1); ax.set_xlabel("share of test attack episodes with an alert during the episode")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    _fig_legend(fig, dets)
    fa = res["detectors"].get("per_frame", {}).get("false_alarms_pooled", {}).get("between_attack_episodes")
    note = ("\nper-frame: any flagged frame counts. Between episodes it raises "
            f"{fa['per_s']:.0f} alarms/s, so almost any 3 s interval has one.\nFaded bar = chance hits, not detection."
            if fa else "")
    _title(fig, ax, "Attack episodes detected, per attack type", sub + note)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(out, "per_attack_detection.png"), dpi=130)
    plt.close(fig)


def _latency(plt, out, res, dets, sub):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    h = 0.8 / len(dets)
    for j, (k, name, color) in enumerate(dets):
        lat = res["detectors"][k]["latency_ms_all"]
        for i, (short, label) in enumerate(ATTACKS):
            v = np.array([x for x in lat[_attack_key(lat, short)] if x is not None], dtype=float)
            y = i + (j - (len(dets) - 1) / 2) * h
            if not len(v):
                ax.text(LABEL_X, y, "not detected", va="center", fontsize=7.5, color=INK2)
                continue
            p50, p99 = np.percentile(v, 50), np.percentile(v, 99)
            fmt = lambda x: "< 1" if x < 1 else f"{x:,.0f}"
            ax.text(LABEL_X, y, f"p50 {fmt(p50)} · p99 {fmt(p99)} ms  (n = {len(v)})", va="center", fontsize=7.5,
                    color=INK2)
            v = np.maximum(v, 1.0)                                    # log axis: < 1 ms is drawn at 1 ms
            jitter = (np.random.default_rng(i * 7 + j).random(len(v)) - 0.5) * h * 0.6
            ax.scatter(v, y + jitter, s=9, color=color, alpha=0.55, linewidths=0, zorder=2)
            ax.plot([max(p50, 1.0)] * 2, [y - h * 0.42, y + h * 0.42], color=INK, lw=2, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.8, 2.5e5)
    ax.set_yticks(range(len(ATTACKS))); ax.set_yticklabels([l for _, l in ATTACKS]); ax.invert_yaxis()
    ax.set_xlabel("time from the episode's first injected frame to its first alert (ms, log; < 1 ms drawn at 1 ms)")
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    _fig_legend(fig, dets)
    _title(fig, ax, "Detection latency per attack episode", sub + "\ndot = one episode, bar = median; per-frame "
           "autoencoder: first flagged frame of any kind during the episode")
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(out, "latency.png"), dpi=130); plt.close(fig)


def _false_alarms(plt, out, res, dets, sub):
    groups = [("normal_run_data", "separate attack-free capture\n(normal_run_data)"),
              ("DoS_dataset#1", "attack-free tail of the\nattack CSVs (DoS_dataset#1)"),
              ("between_attack_episodes", "attack captures, between\nepisodes (the ~3 s gaps)")]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    h = 0.8 / len(dets)
    lo = 0.3
    for j, (k, name, color) in enumerate(dets):
        s = res["detectors"][k]
        for i, (g, label) in enumerate(groups):
            d = s["false_alarms"].get(g) or s["false_alarms_pooled"].get(g)
            y = i + (j - (len(dets) - 1) / 2) * h
            ub = d["per_hour_95_upper"]
            if d["false_alarms"] == 0:
                ax.plot([lo, ub], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
                ax.plot(ub, y, "|", color=color, ms=9, mew=2, zorder=3)
                ax.text(ub * 1.3, y, f"0 in {d['attack_free_s']:,.0f} s (95 % bound < {ub:,.0f}/h)",
                        va="center", fontsize=7.5, color=INK2)
            else:
                ax.barh(y, d["per_hour"] - lo, left=lo, height=h * 0.85, color=color, edgecolor=SURFACE, lw=2, zorder=2)
                ax.text(d["per_hour"] * 1.3, y, f"{d['false_alarms']:,} in {d['attack_free_s']:,.0f} s = "
                        f"{d['per_hour']:,.0f}/h", va="center", fontsize=7.5, color=INK2)
    ax.axvline(TARGET_PER_HOUR, color=INK2, lw=1, zorder=1)
    ax.text(TARGET_PER_HOUR * 1.1, -0.5, "operational bar: 1 per hour", fontsize=7.5, color=INK2, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(lo, 3e7); ax.set_ylim(len(groups) - 0.5, -0.65)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([l for _, l in groups])
    ax.set_xlabel("false alarms per hour of attack-free test traffic (log)\na line = none observed, drawn up to the 95 % upper bound")
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    _fig_legend(fig, dets)
    _title(fig, ax, "False alarms per hour", sub + "\nper-frame: every flagged frame is an alarm; the others: one alarm per event")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(out, "false_alarms.png"), dpi=130); plt.close(fig)


def _frame_level(plt, out, res, dets, sub):
    dets = [d for d in dets if "frame_level" in res["detectors"][d[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    w = 0.8 / len(dets)
    panels = [("all attacks vs normal", [("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
                                         ("roc_auc", "ROC-AUC")]),
              ("recall per attack type", [(s, l) for s, l in ATTACKS])]
    for ax, (title, metrics) in zip(axes, panels):
        for j, (k, name, color) in enumerate(dets):
            fm = res["detectors"][k]["frame_level"]
            for i, (m, label) in enumerate(metrics):
                v = fm[m] if m in fm else fm["recall_per_attack"][_attack_key(fm["recall_per_attack"], m)]
                x = i + (j - (len(dets) - 1) / 2) * w
                _bar(ax, x, v, color, w)
                ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([l for _, l in metrics])
        ax.set_ylim(0, 1.12); ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    _legend(axes[1], dets, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.suptitle("Frame-level scores: per-frame autoencoder vs windowed detector", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.01, 0.885, sub + "; a windowed verdict applies to every frame of that ID in that window",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(out, "baseline_vs_windowed.png"), dpi=130)
    plt.close(fig)


def _timeline(plt, out, tl, sub):
    labs = [l for l in ("gear", "rpm") if f"{l}_t" in tl]
    if not labs:
        return
    fig, axes = plt.subplots(len(labs), 1, figsize=(8.6, 2.5 * len(labs) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        t, c, hi = tl[f"{lab}_t"], tl[f"{lab}_count"], int(tl[f"{lab}_hi"][0])
        for s, e in tl[f"{lab}_ep"]:
            ax.axvspan(s, e, color=SLOTS[1], alpha=0.10, lw=0, zorder=0)
        ax.step(t, c, where="post", color=SLOTS[0], lw=2, zorder=2)
        ax.axhline(hi, color=INK2, lw=1, zorder=1)
        ax.text(t[0], hi * 0.55, f"envelope: {hi} frames (the most in any normal training window)", fontsize=7.5,
                color=INK2, va="top")
        for r in tl[f"{lab}_raise"]:
            ax.plot(r, hi, "v", ms=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, max(100, c.max() * 1.6))
        ax.set_ylabel("frames / 100 ms")
        ax.set_xlim(t[0] - 0.1, t[-1] + 0.1)
        ax.set_title(f"{dict(ATTACKS_ALL)[lab]}: ID 0x{int(tl[f'{lab}_id'][0]):03x}, first test episode "
                     "(shaded = injection, ▼ = alert raised)", loc="left", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[-1].set_xlabel("seconds from the episode's first injected frame")
    fig.suptitle("Why the windowed detector sees spoofing: per-ID frame counts", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "timeline.png"), dpi=130); plt.close(fig)


def _sensitivity(plt, out, sens, sub):
    labs = [l for l in ("gear", "rpm") if l in sens]
    fig, axes = plt.subplots(1, len(labs), figsize=(10.0, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        d = sens[lab]
        x = np.array([r["injected_per_s_median"] for r in d["rows"]])
        y = np.array([r["fraction"] for r in d["rows"]])
        ax.plot(x, y, color=SLOTS[0], lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for r, xi, yi in zip(d["rows"], x, y):
            ax.text(xi, yi + 0.05, f"{r['detected']}/{r['episodes']}", ha="center", fontsize=7, color=INK2)
        ax.axvline(d["normal_per_s"], color=INK2, lw=1, zorder=1)
        ax.text(d["normal_per_s"] * 1.08, 0.04, f"the ID's normal rate\n({d['normal_per_s']:.0f} frames/s)",
                fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.15)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("injected frames per second (median over episodes, log)")
        ax.set_title(f"{dict(ATTACKS_ALL)[lab]}: ID {d['id']}, envelope {d['envelope']} frames / 100 ms", loc="left",
                     fontsize=9, fontweight="bold")
    axes[0].set_ylabel("test episodes detected")
    fig.suptitle("How slow can a spoofing attack be? (derived: only every m-th injected frame kept)", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, 0.87, "windowed detector; right-most point = the recorded HCRL attack, each step left halves "
             "the injected frames; all other traffic as recorded", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(out, "rate_sensitivity.png"), dpi=130)
    plt.close(fig)
