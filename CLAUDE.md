# GuardCAN — project context for Claude Code

GPU-accelerated AI intrusion detection for in-vehicle networks (CAN/CAN-FD), with a fleet V-SOC
console. A portfolio proof-of-concept demonstrating **applied AI** (anomaly detection) and
**HPC/systems** (a line-rate GPU inference engine), built on NVIDIA developer tools and mapped to
automotive cyber regulations (UN R155/R156, ISO 21434/26262/42001).

Full plan: see the build spec (`GuardCAN_Build_Spec.md`, kept in the PerfStack project docs).

## Non-negotiables

- **Honesty.** This is a POC — never imply real-vehicle deployment or production use. Report only
  **measured** numbers; never invent or hardcode results. Label synthetic data as synthetic.
- **Must run on a free Colab T4 GPU.** CUDA work that needs a GPU is delivered as `.cu` sources plus
  a Colab notebook that compiles with `nvcc` and runs. Don't assume a local GPU.
- **Match the existing structure and style.** Don't restructure the repo or rewrite working code
  without reason. Keep the PyTorch path primary with the scikit-learn fallback.

## Repo layout

```
ingest/   can_loader.py (HCRL Car-Hacking parser), features.py (22-feature per-message vectors)
model/    autoencoder.py (torch + sklearn), train_autoencoder.py, train_classifier.py, eval.py, common.py
engine/   (Week 2) CUDA inference engine + memory-strategy benchmark
soa/      (Week 3) synthetic SOME/IP + zero-copy tap
serve/    (Week 4) TensorRT + Triton
dashboard/(Week 3) V-SOC console
docs/     architecture, compliance-mapping, threat-model, (Week 2) hpc-engine
data/     README (dataset download) + make_synthetic.py
bench/    committed results: metrics.json, Nsight profiles, charts
```

Features are 22 per message: 11 CAN-ID bits + DLC/8 + 8 payload bytes/255 + 2 timing features
(global and same-ID inter-arrival, log1p + standardized). Autoencoder: 22→16→8→16→22.

## Run (Week 1)

```bash
pip install -r requirements.txt
python run_week1.py                              # synthetic data -> train -> eval
# real dataset (Colab GPU):
python -m model.train_autoencoder --data data/car-hacking --backend torch --out models/
python -m model.train_classifier  --data data/car-hacking --out models/
python -m model.eval              --data data/car-hacking --models models/
```

## Current focus — Week 2: the HPC engine (the differentiator)

Build `engine/`: a CUDA stream-concurrent batched inference engine running the autoencoder's forward
pass on the GPU (hand-written CUDA MLP; overlap H2D/kernel/D2H across streams; micro-batches), a
**memory-strategy benchmark** (pageable vs pinned vs zero-copy vs unified — throughput + latency),
a single-thread CPU baseline, and **p50/p99/p99.9 latency** via Nsight. Export the Week 1
autoencoder weights (ONNX or a NumPy dump) so the CUDA engine can load them. Commit profiles +
a results table + chart to `bench/`, add `docs/hpc-engine.md`, and update the README results.

## Conventions

- Python 3.11, numpy/pandas/sklearn; PyTorch for the GPU model; CUDA C++ for the engine.
- Keep scripts runnable standalone (argparse `main(argv=None)`) and importable from `run_*.py`.
- Deterministic where possible (seed 42; shared split via `model/common.py`).
