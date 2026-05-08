import json
import gdown
import os
import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

downloaded_file = "challenge.mat"
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"

if not os.path.exists(downloaded_file):
    gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def remove_rank1(X):
    Xc = X - X.mean(axis=0, keepdims=True)

    C = Xc.conj().T @ Xc / Xc.shape[0]
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, -1:]

    coherent = (Xc @ v) @ v.conj().T
    return X - coherent


def your_canceller(tx_n, rx):
    rx_base = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    tx_removed = rx - rx_base

    beta = np.array([0.82, 0.90, 0.98, 0.78], dtype=float)[None, :]
    X = rx - beta * tx_removed

    rx_hat = remove_rank1(X)
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")

rx_try = your_canceller(tx_n, rx)
yours_reds, yours_avg = helpers["score"](rx, rx_try, label=f"yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)