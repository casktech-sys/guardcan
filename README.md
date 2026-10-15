# GuardCAN

**GPU-accelerated AI intrusion detection for in-vehicle networks (CAN / CAN-FD), with a fleet V-SOC console.**

A portfolio proof-of-concept that demonstrates both **applied AI** (anomaly detection on automotive network traffic) and **HPC/systems** (a line-rate GPU inference engine), built on NVIDIA developer tools and mapped to the regulations that make in-vehicle intrusion detection a near-future requirement (UN R155/R156, ISO/SAE 21434, ISO 26262, ISO 42001).

> **Status: proof-of-concept.** Not deployed in real vehicles. The CAN-layer detector is trained and evaluated on the public HCRL Car-Hacking dataset; the service-oriented (SOME/IP) layer and the fleet scale are represented with synthetic data and clearly labelled as such. All reported numbers are POC measurements.

---

## What's in Week 1 (this drop)

The CAN detection core — everything you need to train and evaluate a detector today:

```
guardcan/
├── data/
│   ├── README.md          # how to get the real HCRL Car-Hacking dataset
│   └── make_synthetic.py  # generates a small Car-Hacking-format dataset so you can run NOW
├── ingest/
│   ├── can_loader.py      # robust parser for the (ragged) Car-Hacking CSV format
│   └── features.py        # per-message features: ID bits, payload bytes, timing
├── model/
│   ├── autoencoder.py     # unsupervised AE (PyTorch on GPU; scikit-learn fallback)
│   ├── train_autoencoder.py
│   ├── train_classifier.py# supervised attack-type classifier
│   └── eval.py            # per-class precision/recall/F1, confusion matrix, ROC-AUC
├── docs/                  # architecture, compliance mapping, threat model
└── run_week1.py           # one command: synth data -> features -> train -> eval
```

## Run it in 30 seconds (no dataset needed)

```bash
pip install -r requirements.txt
python run_week1.py          # generates synthetic data, trains, evaluates, prints an F1 table
```

This runs end-to-end on CPU with the scikit-learn fallback, so you can confirm the pipeline before downloading anything or opening Colab.

## Run it on the real dataset (Colab GPU)

1. Get the HCRL Car-Hacking dataset — see `data/README.md`.
2. Put the CSVs in `data/car-hacking/`.
3. Train + evaluate:

```bash
# unsupervised anomaly detector (learns "normal", flags deviations incl. novel attacks)
python -m model.train_autoencoder --data data/car-hacking --backend torch --out models/

# supervised attack-type classifier (DoS / fuzzy / gear-spoof / rpm-spoof)
python -m model.train_classifier  --data data/car-hacking --out models/

# evaluate both; writes bench/metrics.json
python -m model.eval --data data/car-hacking --models models/
```

On Colab, `--backend torch` uses the GPU automatically (`torch.cuda.is_available()`).

## The two models (and why two)

- **Autoencoder (unsupervised).** Trained only on *normal* traffic; the reconstruction error is the anomaly score. Story: it flags attacks it was never trained on (novel / zero-day).
- **Classifier (supervised).** The dataset is labelled, so a multi-class model names the attack type for the SOC console's class breakdown and the per-class F1 table.

## Coming in later weeks (see `docs/` and the build spec)

- Week 2: the HPC engine — CUDA stream concurrency + a pinned/zero-copy/unified **memory-strategy benchmark**, with p50/p99/p99.9 latency via Nsight.
- Week 3: the SOA/Automotive-Ethernet layer (synthetic SOME/IP) + the zero-copy GPU tap, and the live V-SOC dashboard.
- Week 4: TensorRT INT8 edge build, Triton/Morpheus framing, packaging.

## License

MIT (template). Dataset is **not** redistributed here — download it from HCRL under its own terms.
