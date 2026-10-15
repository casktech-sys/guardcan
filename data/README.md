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

## No download? Run on synthetic data

`make_synthetic.py` writes a small dataset in the exact same format so you can exercise the whole
pipeline immediately:

```bash
python data/make_synthetic.py          # writes data/synthetic/*.csv
```

The synthetic set is for smoke-testing only — real numbers come from the HCRL data.
