"""Unsupervised autoencoder with a uniform interface and two backends.

  * TorchAutoencoder  -> for Colab / local CUDA (GPU auto-detected). This is the real path.
  * SklearnAutoencoder-> dependency-light fallback so the pipeline runs anywhere (incl. CI / CPU).

Both expose: fit(X_normal), reconstruction_error(X) -> per-sample MSE, save(path).
Use build_autoencoder(backend=...) to construct and load_autoencoder(path) to restore.
"""
import numpy as np
import joblib

HIDDEN = (16, 8)        # encoder widths; decoder mirrors them. Bottleneck = 8.


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- torch
class TorchAutoencoder:
    backend = "torch"

    def __init__(self, input_dim, hidden=HIDDEN, epochs=30, batch=512, lr=1e-3, seed=42):
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.net = None
        self.device = None

    def _build(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        h1, h2 = self.hidden
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, self.input_dim),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)

    def fit(self, X):
        import torch
        if self.net is None:
            self._build()
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        n = X.shape[0]
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for i in range(0, n, self.batch):
                b = X[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = loss_fn(self.net(b), b)
                loss.backward()
                opt.step()
                total += loss.item() * b.shape[0]
            if ep == 0 or (ep + 1) % 10 == 0:
                print(f"    [torch] epoch {ep+1:>3}/{self.epochs}  mse={total/n:.5f}")
        return self

    def reconstruction_error(self, X):
        import torch
        self.net.eval()
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            err = ((self.net(X) - X) ** 2).mean(dim=1)
        return err.cpu().numpy().astype(np.float32)

    def save(self, path):
        import torch
        state = {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}
        joblib.dump({"backend": "torch", "state": state, "input_dim": self.input_dim,
                     "hidden": self.hidden}, path)


# --------------------------------------------------------------------------- sklearn
class SklearnAutoencoder:
    backend = "sklearn"

    def __init__(self, input_dim, hidden=HIDDEN, epochs=200, seed=42):
        from sklearn.neural_network import MLPRegressor
        h1, h2 = hidden
        self.input_dim = input_dim
        self.model = MLPRegressor(hidden_layer_sizes=(h1, h2, h1), activation="relu",
                                  solver="adam", max_iter=epochs, random_state=seed,
                                  early_stopping=False)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.fit(X, X)           # learn identity through an 8-wide bottleneck
        print(f"    [sklearn] trained in {self.model.n_iter_} iters, final loss={self.model.loss_:.5f}")
        return self

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=np.float32)
        recon = self.model.predict(X)
        return ((recon - X) ** 2).mean(axis=1).astype(np.float32)

    def save(self, path):
        joblib.dump({"backend": "sklearn", "model": self.model, "input_dim": self.input_dim}, path)


# --------------------------------------------------------------------------- factory
def build_autoencoder(backend, input_dim, **kw):
    if backend == "auto":
        backend = "torch" if torch_available() else "sklearn"
    if backend == "torch":
        return TorchAutoencoder(input_dim, **kw), "torch"
    if backend == "sklearn":
        kw.pop("batch", None); kw.pop("lr", None)   # not used by sklearn
        return SklearnAutoencoder(input_dim, **kw), "sklearn"
    raise ValueError(f"unknown backend {backend!r}")


def load_autoencoder(path):
    d = joblib.load(path)
    if d["backend"] == "sklearn":
        ae = SklearnAutoencoder(d["input_dim"])
        ae.model = d["model"]
        return ae
    # torch
    ae = TorchAutoencoder(d["input_dim"], hidden=tuple(d["hidden"]))
    ae._build()
    import torch
    sd = {k: torch.as_tensor(v) for k, v in d["state"].items()}
    ae.net.load_state_dict(sd)
    return ae
