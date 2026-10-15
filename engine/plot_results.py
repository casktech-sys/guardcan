"""Turn the engine's measured JSON (bench/hpc/*.json) into charts, tables, and a headline summary.

Nothing here measures or invents a number: every value is read from the JSON written by the CUDA /
C++ benchmarks, or from the Nsight Systems trace CSV, and the derived figures (speedups, overlap,
hand-off gaps) are simple ratios / interval arithmetic on those measurements.

    python -m engine.plot_results --dir bench/hpc

Writes into --dir:
  summary.json                  headline numbers (+ where each came from)
  results.md / results.csv      full tables
  mem_strategy_throughput.png   end-to-end ingest throughput vs micro-batch, per memory strategy
  mem_strategy_latency.png      p50 / p99 / p99.9 end-to-end latency per strategy (small multiples)
  streams_sweep.png             engine throughput: depth-first vs breadth-first issue, 1-8 streams
  nsys_timeline.png             GPU timeline from the nsys trace (if `make nsys` was run)
"""
import os
import csv
import json
import argparse

# Reference palette (dataviz skill, light mode), validated for these 5 adjacent slots.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRATEGIES = ["pageable", "pinned", "zero-copy", "zero-copy+wc", "unified", "unified+prefetch"]
STRAT_COLOR = {s: SLOTS[i] for i, s in enumerate(STRATEGIES)}       # colour follows the entity
CPU_LABEL = "CPU, 1 core (SIMD)"


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def kernel_results(d, eng):
    """All kernel-only measurements as [(label, record)], sustained first. Sources: kernel_only.json
    (`infer_streams --kernel-only`) and the kernel-only pass at the end of the infer_streams.json sweep."""
    out = []
    for src, where in ((_load(d, "kernel_only.json"), "kernel_only.json"), (eng, "end of the sweep run")):
        if not src:
            continue
        if "kernel_only_sustained" in src:
            out.append((f"sustained ({where})", src["kernel_only_sustained"]))
        if "kernel_only" in src:
            out.append((f"sync ({where})", src["kernel_only"]))
    out.sort(key=lambda t: not t[0].startswith("sustained"))
    return out


def verify_records(d, eng):
    """[(label, verify)]: the standalone verify.json (`make verify`, the current models/) first, then the
    check the timing run itself performed against the weights it was given."""
    out = []
    v = _load(d, "verify.json")
    if v:
        out.append((f"current model (verify.json, threshold {v['threshold']:.6f})", v["verify"]))
    if eng and "verify" in eng:
        out.append(("weights used by the timing runs (infer_streams.json)", eng["verify"]))
    return out


def _m(x):
    return x / 1e6


def _fmt_rate(x):
    return f"{x / 1e6:,.1f} M" if x >= 1e6 else f"{x / 1e3:,.0f} k"


def _clk(r):
    c = r.get("gpu_clocks")
    return f"{c['sm_clock_mhz_median']:,.0f} MHz {c['pstate_mode']}" if c else "–"


def _fmt_us(x):
    if x is None:
        return "–"
    return f"{x:,.0f}" if x >= 100 else (f"{x:,.1f}" if x >= 1 else f"{x:.2f}")


# --------------------------------------------------------------------------- matplotlib style
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    })
    return plt


def _title(fig, title, subtitle):
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=8.5, color=INK2)


def _platform(dev):
    plat = "WSL2" if "WSL2" in dev.get("platform", "") else "native Linux"
    rt = int(dev["cuda_runtime_version"])
    return f"{dev['gpu'].replace('NVIDIA GeForce ', '')} · {plat} · CUDA {rt // 1000}.{rt % 1000 // 10}"


# --------------------------------------------------------------------------- charts
def plot_mem_throughput(plt, mem, cpu, out):
    rows = mem["throughput"]
    strategies = [s for s in STRATEGIES if any(r["strategy"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.13)
    for s in strategies:
        rs = sorted((r for r in rows if r["strategy"] == s), key=lambda r: r["batch"])
        xs = [r["batch"] for r in rs]
        ax.plot(xs, [_m(r["frames_per_s_median"]) for r in rs], color=STRAT_COLOR[s], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=s)
        ax.vlines(xs, [_m(r["frames_per_s_min"]) for r in rs], [_m(r["frames_per_s_max"]) for r in rs],
                  color=STRAT_COLOR[s], lw=1, alpha=0.6, zorder=2)
    ymax = max(_m(r["frames_per_s_max"]) for r in rows) * 1.12
    if cpu:
        simd = next(t for t in cpu["throughput"] if t["impl"] == "simd")
        ax.axhline(_m(simd["frames_per_s_median"]), color=MUTED, lw=1.2, zorder=1)
        xs_all = sorted({r["batch"] for r in rows})
        ax.annotate(f"{CPU_LABEL}: {_fmt_rate(simd['frames_per_s_median'])} frames/s",
                    xy=(xs_all[len(xs_all) // 2], _m(simd["frames_per_s_median"])), xytext=(0, -5),
                    textcoords="offset points", fontsize=8, color=INK2, va="top", ha="center")
    ax.set_xscale("log", base=2)
    xt = sorted({r["batch"] for r in rows})
    ax.set_xticks(xt, [f"{x:,}" for x in xt])
    ax.minorticks_off()
    ax.set_ylim(0, ymax)
    ax.set_xlabel("micro-batch size (frames)")
    ax.set_ylabel("frames / s (millions)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(strategies), bbox_to_anchor=(0.08, 0.875),
               fontsize=8.5, handlelength=1.6, columnspacing=1.2)
    _title(fig, "End-to-end ingest throughput by memory strategy",
           f"{_platform(mem['device'])} · produce → score → consume · {rows[0]['streams_per_wave']} streams × 2 waves "
           f"in flight · median of {rows[0]['reps']} runs (whiskers: min–max)")
    fig.savefig(os.path.join(out, "mem_strategy_throughput.png"), dpi=130)
    plt.close(fig)


def plot_mem_latency(plt, mem, cpu, out):
    import matplotlib
    import matplotlib.ticker  # noqa: F401
    rows = mem["latency"]
    batches = sorted({r["batch"] for r in rows})
    strategies = [s for s in STRATEGIES if any(r["strategy"] == s for r in rows)]
    labels = strategies + ([CPU_LABEL] if cpu else [])
    fig, axes = plt.subplots(1, len(batches), figsize=(2.6 * len(batches) + 1.4, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.76, bottom=0.2, wspace=0.08)
    marks = [("p50", "o", "p50"), ("p99", "s", "p99"), ("p99_9", "D", "p99.9")]
    for ax, B in zip(axes, batches):
        for i, lab in enumerate(labels):
            if lab == CPU_LABEL:
                r = next((x for x in cpu["latency"] if x["batch"] == B), None)
                color = MUTED
            else:
                r = next((x for x in rows if x["strategy"] == lab and x["batch"] == B), None)
                color = STRAT_COLOR[lab]
            if r is None:
                continue
            e = r["e2e_us"]
            y = len(labels) - 1 - i
            ax.plot([e["p50"], e["p99_9"]], [y, y], color=color, lw=1.2, zorder=2)
            for key, mk, _ in marks:
                ax.plot(e[key], y, marker=mk, ms=5.5, color=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                        zorder=3, ls="none")
        ax.set_title(f"B = {B:,} frame{'s' if B > 1 else ''}", fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)), labels[::-1])
    axes[0].tick_params(axis="y", colors=INK2, length=0)
    fig.supxlabel("end-to-end latency per micro-batch (µs, log scale)", fontsize=9, color=INK2, y=0.03)
    handles = [plt.Line2D([], [], marker=mk, ls="none", color=INK2, ms=5.5, label=lab) for _, mk, lab in marks]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.885), fontsize=8.5)
    _title(fig, "Tail latency by memory strategy",
           f"{_platform(mem['device'])} · closed loop, one batch in flight · host clock from produce start to consume end · "
           f"{rows[0]['iters']:,} batches per point")
    fig.savefig(os.path.join(out, "mem_strategy_latency.png"), dpi=130)
    plt.close(fig)


def plot_streams(plt, eng, cpu, out):
    rows = eng["sweep"]
    streams = sorted({r["streams"] for r in rows})
    s_color = {s: SLOTS[i] for i, s in enumerate(streams)}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.14, wspace=0.08)
    ymax = max(_m(r["frames_per_s_max"]) for r in rows) * 1.1
    for ax, order in zip(axes, ["depth-first", "breadth-first"]):
        for S in streams:
            rs = sorted((r for r in rows if r["order"] == order and r["streams"] == S), key=lambda r: r["batch"])
            if order == "breadth-first" and S == 1:      # identical to depth-first; shown for reference
                rs = sorted((r for r in rows if r["streams"] == 1), key=lambda r: r["batch"])
            if not rs:
                continue
            xs = [r["batch"] for r in rs]
            ax.plot(xs, [_m(r["frames_per_s_median"]) for r in rs], color=s_color[S], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{S} stream{'s' if S > 1 else ''}")
            ax.vlines(xs, [_m(r["frames_per_s_min"]) for r in rs], [_m(r["frames_per_s_max"]) for r in rs],
                      color=s_color[S], lw=1, alpha=0.6)
        if cpu:
            simd = next(t for t in cpu["throughput"] if t["impl"] == "simd")
            ax.axhline(_m(simd["frames_per_s_median"]), color=MUTED, lw=1.2, zorder=1)
            ax.annotate(CPU_LABEL, xy=(min(r["batch"] for r in rows), _m(simd["frames_per_s_median"])),
                        xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, va="bottom")
        ax.set_title(f"{order} issue", loc="left")
        ax.set_xscale("log", base=2)
        xt = sorted({r["batch"] for r in rows})
        ax.set_xticks(xt, [f"{x:,}" for x in xt])
        ax.minorticks_off()
        ax.set_ylim(0, ymax)
        ax.set_xlabel("micro-batch size (frames)")
    axes[0].set_ylabel("frames / s (millions)")
    axes[1].legend(loc="upper left", fontsize=8.5)
    _title(fig, "Stream pipeline (pinned H2D → kernel → D2H): depth-first vs breadth-first issue",
           f"{_platform(eng['device'])} · {eng['device']['async_engine_count']} copy engine · "
           f"median of {rows[0]['reps']} runs (whiskers: min–max) · GPU clock per point in results.md")
    fig.savefig(os.path.join(out, "streams_sweep.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- nsys trace analysis
def _read_trace(d):
    p = os.path.join(d, "nsys_cuda_gpu_trace.csv")
    if not os.path.exists(p):
        return None, None
    ops = []
    for r in csv.DictReader(open(p)):
        name = r["Name"]
        kind = "kernel" if "score_kernel" in name else ("H2D" if "Host-to-Device" in name else
                                                         ("D2H" if "Device-to-Host" in name else None))
        if kind:
            s = int(r["Start (ns)"])
            ops.append({"kind": kind, "start": s, "end": s + int(r["Duration (ns)"]), "stream": r["Strm"]})
    ranges = []
    p2 = os.path.join(d, "nsys_nvtx_pushpop_trace.csv")
    if os.path.exists(p2):
        for r in csv.DictReader(open(p2)):
            name = r.get("Name", "").lstrip(":")            # nsys prefixes push/pop ranges with ':'
            if "S=" in name:
                ranges.append({"name": name, "start": int(r["Start (ns)"]), "end": int(r["End (ns)"])})
    return ops, ranges


def _union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def trace_stats(ops, rng):
    sel = [o for o in ops if o["start"] >= rng["start"] and o["end"] <= rng["end"]]
    if not sel:
        return None
    t0, t1 = min(o["start"] for o in sel), max(o["end"] for o in sel)
    span = t1 - t0
    copies = [(o["start"], o["end"]) for o in sel if o["kind"] != "kernel"]
    kernels = [(o["start"], o["end"]) for o in sel if o["kind"] == "kernel"]
    busy = _union(copies + kernels)
    # kernel time that overlaps a copy
    ov = 0
    for ks, ke in kernels:
        ov += _union([(max(ks, cs), min(ke, ce)) for cs, ce in copies if cs < ke and ce > ks])
    # same-stream hand-off gaps: H2D end -> kernel start, kernel end -> D2H start
    gaps = {"h2d_to_kernel": [], "kernel_to_d2h": []}
    by_stream = {}
    for o in sel:
        by_stream.setdefault(o["stream"], []).append(o)
    for lst in by_stream.values():
        lst.sort(key=lambda o: o["start"])
        for a, b in zip(lst, lst[1:]):
            if a["kind"] == "H2D" and b["kind"] == "kernel":
                gaps["h2d_to_kernel"].append((b["start"] - a["end"]) / 1e3)
            if a["kind"] == "kernel" and b["kind"] == "D2H":
                gaps["kernel_to_d2h"].append((b["start"] - a["end"]) / 1e3)
    n_batches = len(kernels)
    return {"range": rng["name"], "batches": n_batches, "gpu_span_us": span / 1e3,
            "us_per_batch": span / 1e3 / max(1, n_batches),
            "gpu_busy_fraction": busy / span, "copy_busy_fraction": _union(copies) / span,
            "kernel_busy_fraction": _union(kernels) / span,
            "kernel_time_overlapping_copy_fraction": ov / max(1, sum(e - s for s, e in kernels)),
            "median_gap_h2d_to_kernel_us": _median(gaps["h2d_to_kernel"]),
            "median_gap_kernel_to_d2h_us": _median(gaps["kernel_to_d2h"])}


def plot_timeline(plt, ops, ranges, dev, out, stats=None, window_us=700):
    rngs = [r for r in ranges if r["name"].startswith(("depth-first", "breadth-first"))]
    if not rngs:
        return
    fig, axes = plt.subplots(len(rngs), 1, figsize=(9.6, 1.55 * len(rngs) + 1.1), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.14, hspace=0.55)
    rows = ["H2D copy", "kernel", "D2H copy"]
    kind_row = {"H2D": 2, "kernel": 1, "D2H": 0}
    n_streams = 0
    for ax, rng in zip(axes, rngs):
        sel = sorted((o for o in ops if o["start"] >= rng["start"] and o["end"] <= rng["end"]), key=lambda o: o["start"])
        # start the window in steady state (middle of the timed region) on a fresh H2D
        mid = len(sel) // 2
        while mid < len(sel) - 1 and sel[mid]["kind"] != "H2D":
            mid += 1
        t0 = sel[mid]["start"]
        stream_ids = []                                     # stream 0..S-1 within this range
        for o in sel:
            if o["stream"] not in stream_ids:
                stream_ids.append(o["stream"])
        n_streams = max(n_streams, len(stream_ids))
        for o in sel:
            s, e = (o["start"] - t0) / 1e3, (o["end"] - t0) / 1e3
            if e < 0 or s > window_us:
                continue
            ax.broken_barh([(s, max(e - s, 0.6))], (kind_row[o["kind"]] - 0.32, 0.64),
                           facecolors=SLOTS[stream_ids.index(o["stream"]) % len(SLOTS)], edgecolor=SURFACE, linewidth=0.8)
        ax.set_yticks(range(3), rows[::-1])
        ax.tick_params(axis="y", colors=INK2, length=0)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlim(0, window_us)
        ax.grid(axis="y", visible=False)
        ax.set_title(rng["name"].replace("S=", "· S=").replace(" B=", " · B="), loc="left", fontsize=9)
        st = next((x for x in (stats or []) if x["range"] == rng["name"]), None)
        if st:
            ax.set_title(f"whole run: {st['us_per_batch']:.0f} µs/batch · GPU busy {st['gpu_busy_fraction']:.0%} · "
                         f"median gap H2D→kernel {st['median_gap_h2d_to_kernel_us']:.0f} µs, "
                         f"kernel→D2H {st['median_gap_kernel_to_d2h_us']:.0f} µs", loc="right", fontsize=8, color=INK2,
                         fontweight="normal")
    axes[-1].set_xlabel("time (µs) — a steady-state window from the profiled run")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[i]) for i in range(n_streams)]
    fig.legend(handles, [f"stream {i}" for i in range(n_streams)], loc="upper right", ncol=n_streams,
               bbox_to_anchor=(0.98, 0.9), fontsize=8.5)
    _title(fig, "Nsight Systems GPU timeline: where the time goes",
           f"{_platform(dev)} · every bar is a measured GPU operation from the nsys trace")
    fig.savefig(os.path.join(out, "nsys_timeline.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- tables + summary
def build_tables(mem, eng, cpu, nsys_stats, d="."):
    md, flat = [], []

    def row(section, cfg, metric, value):
        flat.append({"section": section, "config": cfg, "metric": metric, "value": value})

    records = verify_records(d, eng)
    if records:
        md += ["## Verification (GPU engine vs the Python model)", "",
               "| model checked | frames | max abs diff | max rel diff | out of tolerance | decision flips | result |",
               "|---|---:|---:|---:|---:|---:|---|"]
        for label, v in records:
            md.append(f"| {label} | {v['frames']:,.0f} | {v['max_abs_diff']:.3g} | {v['max_rel_diff']:.3g} | "
                      f"{v['out_of_tolerance']:.0f} | {v['decision_flips']:.0f} | {'PASS' if v['pass'] else 'FAIL'} |")
            for k in ("max_abs_diff", "max_rel_diff", "out_of_tolerance", "decision_flips"):
                row("verify", label, k, v[k])
        md.append("")
    if mem:
        md += ["## Memory strategy — end-to-end throughput (frames/s, median of reps; min–max)", ""]
        bs = sorted({r["batch"] for r in mem["throughput"]})
        md += ["| strategy | " + " | ".join(f"B={b:,}" for b in bs) + " |", "|---|" + "---:|" * len(bs)]
        for s in STRATEGIES:
            rs = {r["batch"]: r for r in mem["throughput"] if r["strategy"] == s}
            if not rs:
                continue
            cells = [f"{_fmt_rate(rs[b]['frames_per_s_median'])} ({_fmt_rate(rs[b]['frames_per_s_min'])}–"
                     f"{_fmt_rate(rs[b]['frames_per_s_max'])})" if b in rs else "–" for b in bs]
            md.append(f"| {s} | " + " | ".join(cells) + " |")
            for b, r in rs.items():
                row("mem_throughput", f"{s} B={b}", "frames_per_s_median", r["frames_per_s_median"])
                row("mem_throughput", f"{s} B={b}", "frames_per_s_min", r["frames_per_s_min"])
                row("mem_throughput", f"{s} B={b}", "frames_per_s_max", r["frames_per_s_max"])
        for sk in mem.get("skipped", []):
            md.append(f"| {sk['strategy']} | not run: {sk['reason']} |" + " |" * (len(bs) - 1))
        md += ["", "GPU clock / P-state during each throughput run (NVML median):", "",
               "| strategy | " + " | ".join(f"B={b:,}" for b in bs) + " |", "|---|" + "---|" * len(bs)]
        for s_ in STRATEGIES:
            rs = {r["batch"]: r for r in mem["throughput"] if r["strategy"] == s_}
            if rs:
                md.append(f"| {s_} | " + " | ".join(_clk(rs[b]) if b in rs else "–" for b in bs) + " |")
        md += ["", "## Memory strategy — end-to-end latency per micro-batch (µs)", "",
               "| B | strategy | p50 | p99 | p99.9 | max | GPU-event p50 | produce p50 | n | GPU clock |",
               "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for b in sorted({r["batch"] for r in mem["latency"]}):
            for s in STRATEGIES:
                r = next((x for x in mem["latency"] if x["strategy"] == s and x["batch"] == b), None)
                if not r:
                    continue
                e = r["e2e_us"]
                md.append(f"| {b:,} | {s} | {_fmt_us(e['p50'])} | {_fmt_us(e['p99'])} | {_fmt_us(e['p99_9'])} | "
                          f"{_fmt_us(e['max'])} | {_fmt_us(r['gpu_event_us']['p50'])} | {_fmt_us(r['produce_us']['p50'])} | {e['n']:,} | {_clk(r)} |")
                for k in ("p50", "p99", "p99_9", "max"):
                    row("mem_latency_e2e_us", f"{s} B={b}", k, e[k])
            if cpu:
                c = next((x for x in cpu["latency"] if x["batch"] == b), None)
                if c:
                    e = c["e2e_us"]
                    md.append(f"| {b:,} | *{CPU_LABEL}* | {_fmt_us(e['p50'])} | {_fmt_us(e['p99'])} | {_fmt_us(e['p99_9'])} | "
                              f"{_fmt_us(e['max'])} | – | – | {e['n']:,} | – |")
        md.append("")
    if eng:
        md += ["## Stream pipeline sweep (pinned; frames/s median of reps; per-batch latency from CUDA events, µs)", "",
               "| order | streams | B | frames/s | p50 | p99 | p99.9 | max | GPU clock |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in eng["sweep"]:
            l = r["batch_latency_us"]
            md.append(f"| {r['order']} | {r['streams']} | {r['batch']:,} | {_fmt_rate(r['frames_per_s_median'])} | "
                      f"{_fmt_us(l['p50'])} | {_fmt_us(l['p99'])} | {_fmt_us(l['p99_9'])} | {_fmt_us(l['max'])} | {_clk(r)} |")
            cfg = f"{r['order']} S={r['streams']} B={r['batch']}"
            row("stream_sweep", cfg, "frames_per_s_median", r["frames_per_s_median"])
            for k in ("p50", "p99", "p99_9"):
                row("stream_sweep_latency_us", cfg, k, l[k])
        ks = kernel_results(d, eng)
        if ks:
            md += ["", "## Kernel only (device-resident, no PCIe)", "",
                   "| mode | frames/launch | launches | p50 µs/launch | frames/s at p50 | DRAM GB/s | GPU clock |",
                   "|---|---:|---:|---:|---:|---:|---|"]
            for label, k in ks:
                md.append(f"| {label} | {k['frames_per_launch']:,.0f} | {k['launches']:.0f} | {k['launch_us']['p50']:.1f} | "
                          f"{_fmt_rate(k['frames_per_s_at_p50'])} | {k['dram_gb_per_s_at_p50']:.0f} | {_clk(k)} |")
                row("kernel_only", label, "frames_per_s_at_p50", k["frames_per_s_at_p50"])
            md += ["", "*sustained*: back-to-back launches after a ≥300 ms clock ramp (the kernel's ceiling). "
                       "*sync*: each launch timed in isolation; the GPU idles between launches, so the driver may hold a lower clock."]
        md.append("")
    if cpu:
        md += ["## CPU baseline (1 thread)", "", f"`{cpu['compiler_flags']}` on {cpu['host_cpu']} ({cpu['platform']})", "",
               "| implementation | frames/s (median) | ns/frame | verify |", "|---|---:|---:|---|"]
        for t in cpu["throughput"]:
            md.append(f"| {t['impl']} | {_fmt_rate(t['frames_per_s_median'])} | {t['ns_per_frame_median']:.1f} | "
                      f"{'PASS' if t['verify']['pass'] else 'FAIL'} (max rel {t['verify']['max_rel_diff']:.2g}) |")
            row("cpu_baseline", t["impl"], "frames_per_s_median", t["frames_per_s_median"])
        md.append("")
    if nsys_stats:
        md += ["## Nsight Systems trace (profiled run, S=4, B=4,096)", "",
               "| issue order | µs per batch | GPU busy | copy engine busy | kernel time overlapping a copy | "
               "median gap H2D→kernel (µs) | median gap kernel→D2H (µs) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for s in nsys_stats:
            md.append(f"| {s['range']} | {s['us_per_batch']:.1f} | {s['gpu_busy_fraction']:.0%} | {s['copy_busy_fraction']:.0%} | "
                      f"{s['kernel_time_overlapping_copy_fraction']:.0%} | {s['median_gap_h2d_to_kernel_us']:.1f} | "
                      f"{s['median_gap_kernel_to_d2h_us']:.1f} |")
            for k in ("us_per_batch", "gpu_busy_fraction", "copy_busy_fraction", "kernel_time_overlapping_copy_fraction",
                      "median_gap_h2d_to_kernel_us", "median_gap_kernel_to_d2h_us"):
                row("nsys", s["range"], k, s[k])
        md.append("")
    return md, flat


def build_summary(mem, eng, cpu, nsys_stats, d="."):
    s = {"note": "Derived only from the measured JSON / nsys trace in this directory. POC measurements on one machine."}
    simd = next(t for t in cpu["throughput"] if t["impl"] == "simd") if cpu else None
    if mem:
        best = max(mem["throughput"], key=lambda r: r["frames_per_s_median"])
        s["best_end_to_end"] = {k: best[k] for k in ("strategy", "batch", "frames_per_s_median", "frames_per_s_min", "frames_per_s_max")}
        s["lowest_latency_by_batch"] = {}
        for b in sorted({r["batch"] for r in mem["latency"]}):
            r = min((x for x in mem["latency"] if x["batch"] == b), key=lambda x: x["e2e_us"]["p50"])
            entry = {"strategy": r["strategy"], **{k: r["e2e_us"][k] for k in ("p50", "p99", "p99_9")}}
            if cpu:
                c = next((x for x in cpu["latency"] if x["batch"] == b), None)
                if c:
                    entry["cpu_simd"] = {k: c["e2e_us"][k] for k in ("p50", "p99", "p99_9")}
            s["lowest_latency_by_batch"][str(b)] = entry
    if eng:
        best = max((r for r in eng["sweep"]), key=lambda r: r["frames_per_s_median"])
        s["best_engine_pinned"] = {k: best[k] for k in ("order", "streams", "batch", "frames_per_s_median")}
        gains = {}
        for b in sorted({r["batch"] for r in eng["sweep"]}):
            dp = next((r for r in eng["sweep"] if r["order"] == "depth-first" and r["streams"] == 4 and r["batch"] == b), None)
            br = next((r for r in eng["sweep"] if r["order"] == "breadth-first" and r["streams"] == 4 and r["batch"] == b), None)
            if dp and br:
                gains[str(b)] = br["frames_per_s_median"] / dp["frames_per_s_median"]
        s["breadth_over_depth_S4"] = gains
        ks = kernel_results(d, eng)
        if ks:
            label, k = ks[0]
            s["kernel_only_frames_per_s"] = k["frames_per_s_at_p50"]
            s["kernel_only_mode"] = label
            s["kernel_only_gpu_clocks"] = k.get("gpu_clocks")
        recs = verify_records(d, eng)
        s["verify"] = recs[0][1] if recs else eng["verify"]
        s["verify_model"] = recs[0][0] if recs else "infer_streams.json"
    if simd:
        s["cpu_simd_frames_per_s"] = simd["frames_per_s_median"]
        s["cpu_scalar_frames_per_s"] = next(t for t in cpu["throughput"] if t["impl"] == "scalar")["frames_per_s_median"]
        sp = {}
        if mem:
            sp["end_to_end_best_vs_cpu_simd"] = s["best_end_to_end"]["frames_per_s_median"] / simd["frames_per_s_median"]
        if eng:
            sp["engine_pinned_best_vs_cpu_simd"] = s["best_engine_pinned"]["frames_per_s_median"] / simd["frames_per_s_median"]
            if "kernel_only_frames_per_s" in s:
                sp["kernel_only_vs_cpu_simd"] = s["kernel_only_frames_per_s"] / simd["frames_per_s_median"]
        s["speedup"] = sp
    if nsys_stats:
        s["nsys"] = nsys_stats
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Charts + tables from the GuardCAN engine benchmarks.")
    ap.add_argument("--dir", default="bench/hpc", help="directory with the benchmark JSON (and nsys CSVs)")
    args = ap.parse_args(argv)
    d = args.dir
    mem, eng, cpu = _load(d, "mem_strategy.json"), _load(d, "infer_streams.json"), _load(d, "cpu_baseline.json")
    if not (mem or eng or cpu):
        raise SystemExit(f"no benchmark JSON in {d} — run `make -C engine bench` first")
    ops, ranges = _read_trace(d)
    nsys_stats = [x for x in (trace_stats(ops, r) for r in (ranges or []) if r["name"].startswith(("depth", "breadth")))
                  if x] if ops else []

    plt = _plt()
    if mem:
        plot_mem_throughput(plt, mem, cpu, d)
        plot_mem_latency(plt, mem, cpu, d)
    if eng:
        plot_streams(plt, eng, cpu, d)
    if ops and ranges:
        plot_timeline(plt, ops, ranges, (eng or mem)["device"], d, nsys_stats)

    md, flat = build_tables(mem, eng, cpu, nsys_stats, d)
    dev = (eng or mem or {}).get("device", {})
    head = ["# GuardCAN Week-2 engine — measured results", "",
            f"Generated by `engine/plot_results.py` from the JSON in this directory. "
            f"GPU: {dev.get('gpu', '?')} ({dev.get('platform', '?')}); host CPU: {dev.get('host_cpu', '?')}. "
            "Proof-of-concept measurements on one machine; synthetic Car-Hacking-format input.", ""]
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("\n".join(head + md) + "\n")
    with open(os.path.join(d, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "config", "metric", "value"])
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(build_summary(mem, eng, cpu, nsys_stats, d), f, indent=2)
    print(f"[plot] wrote charts, results.md, results.csv, summary.json to {d}/")


if __name__ == "__main__":
    main()
