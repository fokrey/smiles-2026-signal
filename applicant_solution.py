import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
# url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
# downloaded_file = "challenge.mat"
# gdown.download(url, downloaded_file, quiet=False)

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


def your_canceller(tx_n, rx, beta=1.0):
    rx_base = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    tx_removed = rx - rx_base

    rx_scaled = rx - beta * tx_removed

    rx_hat = remove_rank1(rx_scaled)
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
# for beta in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:

rx_try = your_canceller(tx_n, rx, beta=0.9)
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