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
model/    autoencoder.py (torch + sklearn), train_autoencoder.py, train_classifier.py, eval.py, common.py,
          epoch_study.py, export_weights.py
engine/   (Week 2) CUDA inference engine + memory-strategy benchmark
soa/      (Week 3) synthetic SOME/IP + zero-copy tap
serve/    (Week 4) TensorRT + Triton
dashboard/(Week 3) V-SOC console
docs/     architecture, compliance-mapping, threat-model, (Week 2) hpc-engine
data/     README (dataset download) + make_synthetic.py
bench/    committed results: hcrl/ (HCRL metrics + charts, time_split/, epoch study), hpc/ (engine, Nsight).
          bench/metrics.json + roc/confusion.png = synthetic smoke test from run_week1.py (gitignored)
```

Features are 22 per message: 11 CAN-ID bits + DLC/8 + 8 payload bytes/255 + 2 timing features
(global and same-ID inter-arrival, log1p + standardized, computed per recording). Autoencoder: 22→16→8→16→22.

## Run (Week 1)

```bash
pip install -r requirements.txt
python run_week1.py                              # synthetic data -> train -> eval
# real dataset (Colab GPU):
python -m model.train_autoencoder --data data/car-hacking --backend torch --out models/
python -m model.train_classifier  --data data/car-hacking --out models/
python -m model.eval              --data data/car-hacking --models models/ --out bench/hcrl
# harder holdout (last 30 % of each recording): add --split time, --out models/hcrl-time,
# and eval with --models models/hcrl-time --out bench/hcrl/time_split
```

## Status

- **Phase 1 (CAN detection core): done.** Loader, features, autoencoder + classifier, eval.
- **Phase 2 (HPC engine): done.** `engine/`, results in `bench/hpc/`, `docs/hpc-engine.md`.
- **HCRL evaluation: done.** Results in `bench/hcrl/` and the README's HCRL section (random + time split, epoch
  study). The per-frame detector misses gear spoofing, its per-attack recall is unstable across training runs,
  and on the attack-free capture it raises 28–61 false alarms per second (time / random split).

## Current focus — windowed / event-level detection

Move from per-frame alarms to detection over time windows / events, so that the score reflects what a V-SOC
would act on (an attack episode, not a single frame) and the false-alarm rate is reported per unit time.
Measure it on HCRL with both the random and the time split, and keep the per-frame results as the baseline.

## Conventions

- Python 3.11, numpy/pandas/sklearn; PyTorch for the GPU model; CUDA C++ for the engine.
- Keep scripts runnable standalone (argparse `main(argv=None)`) and importable from `run_*.py`.
- Deterministic where possible (seed 42; shared split via `model/common.py`).
