# Data

GuardCAN trains and evaluates on the **HCRL Car-Hacking Dataset** (Hacking and Countermeasure
Research Lab, Korea University). It is **not** redistributed in this repo — download it yourself
under its own terms.

## Get it

- Official: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- Also mirrored on Kaggle (search "Car Hacking Dataset").

Download these CSVs into `data/car-hacking/`:

```
data/car-hacking/
├── normal_run_data.txt     # attack-free traffic (used to train the autoencoder)
├── DoS_dataset.csv
├── Fuzzy_dataset.csv
├── gear_dataset.csv
└── RPM_dataset.csv
```

## Format (what the loader expects)

Each row is one CAN message. Because the DLC (number of data bytes) varies per message, rows are
**ragged** — the flag is always the *last* field:

```
Timestamp,         CAN_ID, DLC, DATA[0], ..., DATA[DLC-1], Flag
1478198376.389427, 0316,   8,   05,21,68,09,21,21,00,6f,   R
```

Each data byte is its own comma-separated field (not space-separated), so a full row has
`3 + DLC + 1` fields.

- `Timestamp` — seconds (float)
- `CAN_ID` — hex (e.g. `0316`)
- `DLC` — 0..8
- `DATA[i]` — hex bytes, `DLC` of them
- `Flag` — `R` = normal, `T` = injected (attack)

`ingest/can_loader.py` parses this robustly (handles the variable column count and files with no
flag column, like `normal_run_data.txt`).

## What the files actually contain (measured on our copy)

These were checked before the first training run on the real data (17,558,346 rows in the five files).
The loader parses every attack CSV with 0 failures, and the R/T counts per file are: DoS 3,078,250 /
587,521, Fuzzy 3,347,013 / 491,847, gear 3,845,890 / 597,252, RPM 3,966,805 / 654,897.

- **The four attack CSVs end with the same 891,068 rows.** Each file is its own capture (32–41 minutes,
  300 injection bursts), followed by an identical 481.6-second attack-free block. The block starts 22 s
  (DoS) to 7,745 s (RPM) after the capture ends. Merged naively, those frames appear four times, and
  the copies straddle the train/test split. The loader keeps the first copy (as recording `DoS_dataset#1`) and
  drops 2,673,204 duplicate rows. That leaves 14,885,142 frames.
- **Every file is a separate recording**, so the timing features are computed per recording (the loader's
  `recording` column). The four attack captures don't overlap in time, but the duplicated block did: in a
  merged stream, its copies sat next to each other and 15.2 % of all rows got an inter-arrival time of exactly 0.
- **Our `normal_run_data.txt` ends mid-line** at an exact 4 KiB boundary (87,371,776 bytes), so this copy looks
  truncated. The loader drops the partial last line and keeps 988,871 frames (506 s, 1,953 frames/s).
- **The DoS, gear and RPM attacks are each one fixed frame, repeated:** DoS `0x000 00 00 00 00 00 00 00 00`
  (an ID never seen in normal traffic), gear `0x43f 01 45 60 ff 6b 00 00 00`, and RPM `0x316 45 29 24 ff 29 24 00 ff`.
  Gear and RPM use legitimate IDs, but their payloads never appear in normal frames. Fuzzy injects 491,847 distinct
  random frames; 1.9 % of them use an ID that also occurs in normal traffic.

## No download? Run on synthetic data

`make_synthetic.py` writes a small dataset in the exact same format so you can exercise the whole
pipeline immediately:

```bash
python data/make_synthetic.py          # writes data/synthetic/*.csv
```

The synthetic set is for smoke-testing only — real numbers come from the HCRL data.
