"""One command for the Week-2 HPC engine: (train if needed) -> export weights -> build -> verify -> benchmark.

    python run_week2.py            # build + verify the CUDA engine against the Python model
    python run_week2.py --bench    # ... then the full measured suite, Nsight timeline (if nsys exists), charts

Needs an NVIDIA GPU and the CUDA toolkit (`nvcc` on PATH); `nsys` is optional. Results land in bench/hpc/
(see docs/hpc-engine.md). On a machine without a GPU use engine/colab_benchmark.ipynb on a Colab T4.
"""
import os
import sys
import shutil
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _make(*targets, **env):
    cmd = ["make", "-C", os.path.join(ROOT, "engine"), *targets] + [f"{k}={v}" for k, v in env.items()]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GuardCAN Week-2: build, verify and benchmark the CUDA engine.")
    ap.add_argument("--bench", action="store_true", help="run the full measured benchmark suite (~25 min)")
    ap.add_argument("--arch", default="native", help="nvcc -arch (native, sm_75 for T4, sm_89 for RTX 40xx)")
    args = ap.parse_args(argv)

    if shutil.which("nvcc") is None:
        sys.exit("nvcc not found on PATH — install the CUDA toolkit (see docs/hpc-engine.md) or use the Colab notebook")

    synth = os.path.join(ROOT, "data", "synthetic")
    models = os.path.join(ROOT, "models")
    if not os.path.exists(os.path.join(models, "autoencoder.joblib")):
        print("== no trained model yet: running Week 1 ==")
        import run_week1
        run_week1.main()

    print("\n== export weights for the engine ==")
    from model import export_weights
    export_weights.main(["--models", models, "--data", synth])

    py = sys.executable
    print("\n== build + verify ==")
    _make("all", ARCH=args.arch)
    _make("verify")

    if args.bench:
        print("\n== benchmark (measured on this GPU) ==")
        _make("bench", PYTHON=py)
        if shutil.which("nsys"):
            _make("nsys")
        else:
            print("nsys not found — skipping the Nsight Systems timeline")
        _make("plot", PYTHON=py)
        print("\nDone. Results in bench/hpc/ (results.md, summary.json, *.png)")


if __name__ == "__main__":
    main()
