"""One command for the Week-1 CAN core: synthetic data -> train AE -> train classifier -> eval.

    python run_week1.py

Uses PyTorch (on the GPU if available) when torch is installed, otherwise the scikit-learn fallback,
so you can confirm the pipeline before downloading the real dataset or opening Colab. Both backends
train for the train_autoencoder default of 200 epochs. For the real dataset, call the three modules
with --data data/car-hacking (see README).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    synth = os.path.join(ROOT, "data", "synthetic")
    models = os.path.join(ROOT, "models")
    bench = os.path.join(ROOT, "bench")

    if not os.path.isdir(synth) or not os.listdir(synth):
        print("== generating synthetic dataset ==")
        from data import make_synthetic
        make_synthetic.main()

    from model import train_autoencoder, train_classifier, eval as eval_mod

    print("\n== train autoencoder ==")
    train_autoencoder.main(["--data", synth, "--out", models, "--backend", "auto"])   # default --epochs 200

    print("\n== train classifier ==")
    train_classifier.main(["--data", synth, "--out", models])

    print("\n== evaluate ==")
    eval_mod.main(["--data", synth, "--models", models, "--out", bench])

    print("\nDone. Metrics in bench/metrics.json")


if __name__ == "__main__":
    main()
