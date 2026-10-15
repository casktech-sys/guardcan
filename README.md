# GuardCAN

**GPU-accelerated AI intrusion detection for in-vehicle networks (CAN / CAN-FD), with a fleet V-SOC console.**

A portfolio proof-of-concept that demonstrates both **applied AI** (anomaly detection on automotive network traffic) and **HPC/systems** (a line-rate GPU inference engine), built on NVIDIA developer tools and mapped to the regulations that make in-vehicle intrusion detection a near-future requirement (UN R155/R156, ISO/SAE 21434, ISO 26262, ISO 42001).

> **Status: proof-of-concept.** Not deployed in real vehicles. **All results so far are on synthetic data.** The CAN-layer detector is built for the public HCRL Car-Hacking dataset (the loader reads its format) but has **not yet been evaluated on it**. Every number below comes from a synthetic Car-Hacking-format set (`data/make_synthetic.py`). The service-oriented (SOME/IP) layer and the fleet scale will also be represented with synthetic data, labelled as such. All reported numbers are POC measurements on one machine.

---

## Repository layout

The CAN detection core (Week 1), the HPC engine (Week 2), and their docs:

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
├── engine/                # Week 2: CUDA inference engine + benchmarks (see below)
├── bench/hpc/             # Week 2: measured results, charts, Nsight Systems profile
├── docs/                  # architecture, compliance mapping, threat model, hpc-engine
├── run_week1.py           # one command: synth data -> features -> train -> eval
└── run_week2.py           # one command: export weights -> build -> verify -> benchmark
```

## Run it in 30 seconds (no dataset needed)

```bash
pip install -r requirements.txt
python run_week1.py          # generates synthetic data, trains, evaluates, prints an F1 table
```

With PyTorch installed, the autoencoder trains with it (on the GPU if one is available). Without PyTorch, the
scikit-learn fallback runs on the CPU. Either way the autoencoder trains for 200 epochs; scikit-learn stops earlier
once its loss converges. Measured end to end: 28 s on an RTX 4060 laptop with PyTorch. The point is to confirm the
pipeline before you download anything or open Colab.

## Run it on the real dataset (local GPU or Colab)

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

Locally or on Colab, `--backend torch` uses the GPU automatically when one is available
(`torch.cuda.is_available()`).

The autoencoder trains for 200 epochs by default; at 30 the PyTorch model was undertrained (detection F1 0.36 on
the synthetic set). Each epoch on the full HCRL set has far more batches, so fewer epochs may be enough there.
Set `--epochs` to change it; this has not been measured on the real data yet.

## The two models (and why two)

- **Autoencoder (unsupervised).** Trained only on *normal* traffic; the reconstruction error is the anomaly score. Story: it flags attacks it was never trained on (novel / zero-day).
- **Classifier (supervised).** The dataset is labelled, so a multi-class model names the attack type for the SOC console's class breakdown and the per-class F1 table.

## Week 1 results (synthetic data)

These come from the **synthetic** Car-Hacking-format set (`data/make_synthetic.py`). They show the pipeline works
end to end and are **not** results on the real HCRL dataset. Held-out test split: 20,903 frames, seed 42, produced
by `python run_week1.py`.

| detector (attack vs normal) | backend | precision | recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Autoencoder (unsupervised) | PyTorch, 200 epochs (GPU) | 0.988 | 0.821 | **0.897** | 0.989 |
| Autoencoder (unsupervised) | scikit-learn (converged after 35 iterations) | 0.989 | 0.821 | **0.897** | 0.991 |

The autoencoder threshold is the 99.5th percentile of its error on normal training traffic. With PyTorch that gives
78 false positives on 13,368 normal test frames, and it misses 18 % of attack frames (recall 0.821).

Attack-type classifier (supervised MLP), per-class F1: dos 1.000 · fuzzy 0.994 · gear 0.999 · normal 1.000 ·
rpm 1.000 (accuracy 0.9998). The synthetic attacks are easy to separate, so expect lower numbers on the real data.

## Week 2 — the HPC engine (measured on a local RTX 4060 Laptop GPU)

The Week-1 autoencoder (22→16→8→16→22, 1,022 parameters) runs on the GPU with **hand-written CUDA**. A fused
kernel keeps the weights in `__constant__` memory and does the scaler, the four layers and the MSE in registers. It
runs inside a **multi-stream, micro-batched pipeline**, and a **memory-strategy benchmark** compares how frames
should reach the GPU: pageable vs pinned vs zero-copy vs unified. Full design, methodology and analysis:
[`docs/hpc-engine.md`](docs/hpc-engine.md).

```
engine/
├── infer_streams.cu       # stream-concurrent batched engine: verify vs Python, streams x batch x issue-order sweep
├── mem_strategy_bench.cu  # pageable / pinned / zero-copy / unified: throughput + p50/p99/p99.9 latency
├── cpu_baseline.cpp       # single-thread CPU baseline (scalar + SIMD), same forward pass
├── ae_kernel.cuh  ae_model.h  npz.h  bench_util.h  gpu_monitor.h
├── plot_results.py        # charts + tables + summary.json from the measured JSON
├── Makefile               # make / verify / bench / nsys / plot
└── colab_benchmark.ipynb  # reproducibility fallback on a Colab T4
model/export_weights.py    # trained model (PyTorch or sklearn) + scaler -> models/ae_weights.npz (+ reference set)
```

**Measured results.** RTX 4060 Laptop GPU, WSL2, CUDA 13.1; i7-14650HX. Synthetic Car-Hacking-format input,
replayed. POC numbers from one machine, all in [`bench/hpc/`](bench/hpc/results.md).

| | result |
|---|---|
| Correctness vs the PyTorch model (200 epochs; 20,903 test frames) | **0** scores out of tolerance, **0** decision flips (max rel. diff 7.9 × 10⁻⁶) |
| CPU baseline, 1 thread, SIMD | 30.9 M frames/s |
| GPU end-to-end, best memory strategy (pinned, B = 65,536) | 104.0 M frames/s (**3.4×**) |
| GPU engine from a pinned ring (breadth-first, 4 streams) | 125.6 M frames/s (**4.1×**) |
| GPU kernel only (device-resident, sustained) | 2.50 G frames/s (**81×**), 230 GB/s ≈ 90 % of DRAM peak |
| Lowest GPU latency, 1-frame batch (zero-copy) | p50 44 µs · p99 129 µs · p99.9 292 µs |
| Same, 16,384-frame batch (zero-copy) vs CPU | p99.9 **912 µs vs 1,960 µs** (CPU) |

The timings used an earlier 30-epoch export of the same architecture. Timing doesn't depend on the weight values;
this was checked, see [docs/hpc-engine.md §5.1](docs/hpc-engine.md#51-correctness).

![End-to-end ingest throughput by memory strategy](bench/hpc/mem_strategy_throughput.png)

What the measurements say:
- **Zero-copy wins small batches** (2.4× pinned at B = 1,024, half the latency). **Pinned DMA wins bulk.**
  Pageable is worst throughout.
- On this WSL2 setup, every copy↔kernel hand-off costs ~33–37 µs of idle GPU (measured in the Nsight trace).
  **Breadth-first issue** across streams beats depth-first by 1.3–9.5×; adding streams to a depth-first pipeline
  does nothing. This is a **WSL2 single-copy-queue effect**, not a general CUDA rule: the GPU exposes one copy engine
  here (`asyncEngineCount = 1`), and under WDDM the streams' work runs as one in-order sequence. On native Linux,
  depth-first multi-stream pipelines normally overlap. That was not measured here; the Colab T4 notebook is the
  comparison point.
- The GPU's driver-managed power state matters. A mostly idle pipeline drops to P8/P5, so every result records
  the clock it ran at.
- **Honest scope:** for this tiny model, one optimised CPU core already exceeds a vehicle's CAN load by orders of
  magnitude. It also has the lower typical latency (p50 and p99) for batches of up to 1,024 frames. At B = 1,024
  the GPU already wins p99.9 (343 µs vs 374 µs). The GPU's measured advantages are aggregated throughput,
  tail latency, and compute headroom (the kernel is ~24× faster than the end-to-end path, which is PCIe/host-bound).

Run it (needs an NVIDIA GPU + CUDA toolkit; `nsys` optional):

```bash
python run_week2.py            # export weights -> build -> verify against the Python model
python run_week2.py --bench    # + full benchmark suite, Nsight timeline, charts  (~20 min)
```

No GPU? Use [`engine/colab_benchmark.ipynb`](engine/colab_benchmark.ipynb) on a free Colab T4.

## Coming in later weeks (see `docs/` and the build spec)

- Week 3: the SOA/Automotive-Ethernet layer (synthetic SOME/IP) + the zero-copy GPU tap, and the live V-SOC dashboard.
- Week 4: TensorRT INT8 edge build, Triton/Morpheus framing, packaging.

## License

MIT — see [LICENSE](LICENSE). The dataset is **not** redistributed here; download it from HCRL under its own terms.
