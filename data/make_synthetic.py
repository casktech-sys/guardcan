"""Generate a small Car-Hacking-FORMAT dataset so the pipeline can be run without a download.

Writes CSVs to data/synthetic/ in the same ragged format as the real HCRL dataset:
    Timestamp, CAN_ID(hex), DLC, DATA[0] ... DATA[DLC-1] (hex, comma-separated), Flag(R/T)

This is for smoke-testing only. Real numbers come from the real HCRL Car-Hacking dataset.
"""
import os
import random
import numpy as np

SEED = 7
OUT = os.path.join(os.path.dirname(__file__), "synthetic")

# A small, fixed set of "ECU" message IDs, each with a base payload and a send period (seconds).
NORMAL_IDS = {
    0x0C9: dict(period=0.010, base=[0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6f]),
    0x130: dict(period=0.020, base=[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x140: dict(period=0.020, base=[0xff, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x18F: dict(period=0.050, base=[0x10, 0x20, 0x30, 0x40, 0x00, 0x00, 0x00, 0x00]),
    0x1F1: dict(period=0.010, base=[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x260: dict(period=0.100, base=[0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x316: dict(period=0.010, base=[0x05, 0x28, 0x84, 0x66, 0x6d, 0x00, 0x00, 0x3a]),
    0x350: dict(period=0.100, base=[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x430: dict(period=0.200, base=[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    0x4B0: dict(period=0.020, base=[0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
}


def _payload(base, rng, n_vary=2):
    """Normal payload: mostly the base pattern with small drift on a couple of bytes."""
    b = list(base)
    for _ in range(n_vary):
        i = rng.randrange(8)
        b[i] = (b[i] + rng.randint(-4, 4)) & 0xFF
    return b


def gen_normal(duration_s, t0, rng):
    """Interleave the IDs on their own periods -> realistic per-ID periodicity."""
    rows = []
    next_t = {cid: t0 + rng.random() * cfg["period"] for cid, cfg in NORMAL_IDS.items()}
    while True:
        cid = min(next_t, key=next_t.get)
        t = next_t[cid]
        if t - t0 > duration_s:
            break
        cfg = NORMAL_IDS[cid]
        rows.append((t, cid, 8, _payload(cfg["base"], rng), "R"))
        jitter = 1.0 + rng.uniform(-0.05, 0.05)
        next_t[cid] = t + cfg["period"] * jitter
    return rows


def inject(rows, kind, rng):
    """Add injected attack rows (flag T) inside the time span of `rows`."""
    t_lo = rows[0][0]
    t_hi = rows[-1][0]
    span = t_hi - t_lo
    atk = []
    if kind == "dos":                      # flood a high-priority ID at high rate
        t = t_lo + span * 0.3
        while t < t_lo + span * 0.6:
            atk.append((t, 0x000, 8, [0x00] * 8, "T"))
            t += 0.0003
    elif kind == "fuzzy":                  # random IDs and payloads
        for _ in range(int(len(rows) * 0.15)):
            t = t_lo + rng.random() * span
            cid = rng.randrange(0x800)
            atk.append((t, cid, 8, [rng.randrange(256) for _ in range(8)], "T"))
    elif kind == "gear":                   # masquerade a legit ID with abnormal payload
        t = t_lo + span * 0.4
        while t < t_lo + span * 0.7:
            atk.append((t, 0x43F, 8, [rng.randrange(256) for _ in range(8)], "T"))
            t += 0.001
    elif kind == "rpm":                    # spoof RPM-carrying ID with out-of-range values
        t = t_lo + span * 0.4
        while t < t_lo + span * 0.7:
            atk.append((t, 0x316, 8, [0xff, 0xff, rng.randrange(256), rng.randrange(256),
                                      0x00, 0x00, 0x00, 0x00], "T"))
            t += 0.001
    merged = rows + atk
    merged.sort(key=lambda r: r[0])
    return merged


def write_csv(path, rows):
    with open(path, "w") as f:
        for t, cid, dlc, data, flag in rows:
            bytes_str = ",".join("%02x" % b for b in data[:dlc])
            f.write("%.6f,%03x,%d,%s,%s\n" % (t, cid, dlc, bytes_str, flag))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # attack-free training traffic
    write_csv(os.path.join(OUT, "normal_run.csv"), gen_normal(30.0, 1_478_000_000.0, rng))

    # attack files = normal background + injected rows
    for kind, fname in [("dos", "DoS_dataset.csv"), ("fuzzy", "Fuzzy_dataset.csv"),
                        ("gear", "gear_dataset.csv"), ("rpm", "RPM_dataset.csv")]:
        base = gen_normal(15.0, 1_478_000_100.0, rng)
        write_csv(os.path.join(OUT, fname), inject(base, kind, rng))

    print("Wrote synthetic dataset to", OUT)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print("  %-20s %7d rows" % (f, sum(1 for _ in open(p))))


if __name__ == "__main__":
    main()
