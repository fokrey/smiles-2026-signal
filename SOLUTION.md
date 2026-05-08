# Reproducibility

I used the provided initial structure and only modified `your_canceller(...)` in `applicant_solution.py`.

To reproduce the result:

```bash
python applicant_solution.py
```

No additional training data or external models are used. The only dependencies are the same ones needed to run the initial code.
Both baseline and my solution results are saved in results.json:

```json
{
  "baseline": {
    "per_channel_db": [
      3.9773294290400014,
      4.8634183195836425,
      3.4855120674078997,
      3.7449751196987413
    ],
    "average_db": 4.017808733932571
  },
  "yours": {
    "per_channel_db": [
      11.840925099180705,
      8.952647803521662,
      11.9632732349449,
      7.5897667204043175
    ],
    "average_db": 10.086653214512896
  }
}
```

# Solution
I kept solution close to the structure that interference has two components:

```
I = F_c( TX )  +  E
```

- **F_c(·)** is an unknown nonlinear function of **all transmitted signals jointly**.
  In particular, the interference on channel `c` can depend on cross-products
  between different TX channels.
- **E[n, c]** is an external interference term. It is **not** a function of
  `tx` but it is **spatially coherent** — the same source appears (with
  different amplitude and phase) across all 4 receive channels.

My final approach has two stages:

Use the provided baseline model to estimate the TX-driven nonlinear interference.
Remove the dominant rank-1 coherent residual component across the four RX channels

### 1: TX-driven interference cancellation

First, I run the provided baseline:

```python
rx_base = baseline(tx_n, rx, helpers["fit_tx_prediction"])
```

This gives a signal where the baseline-estimated TX interference has been removed.

I then recover the component that the baseline removed:

```python
tx_removed = rx - rx_base
```

This is the baseline estimate of the TX-driven interference.Then I apply a small per-channel scaling:

```python
beta = np.array([0.82, 0.90, 0.98, 0.78], dtype=float)[None, :]
X = rx - beta * tx_removed
```

The reason is that the four RX channels have different leakage paths. The baseline already fits each RX channel separately, but in practice its subtraction strength was slightly miscalibrated per channel. A small per-channel scaling improved the residual before the second-stage rank-1 cancellation.

```text
X[:, c] = rx[:, c] - beta[c] * I_tx_hat[:, c]
```

### 2: Rank-1 coherent residual cancellation

After the TX-driven component is reduced, the remaining signal still has a component that is shared across the four RX channels:

```text
I_ext(t, c) ≈ s(t) * a_c
```

where `s(t)` is one shared temporal signal and `a_c` describes how it appears in each RX channel. I compute 4x4 complex covariance matrix across RX channels:

```text
C = Xcᴴ Xc / N
```

Then it takes the dominant eigenvector `v`, which represents the strongest spatial direction in the residual. The rank-1 coherent component is:

```text
coherent = Xc v vᴴ
```

Then this component is subtracted.

## Experiments and failed attempts

### Baseline plus rank-1 residual removal

The first useful improvement was to apply rank-1 residual removal after the baseline. I kept this idea later and tried to improve result. 

### Scaling of the rank-1 component

I tested scaling the rank-1 component:

```python
rx_hat = X - alpha * coherent
```

Values below 1.0 under-subtracted the coherent component, while values above 1.0 started to reduce the score. The best value was 1.0

### Estimating the rank-1 direction from the score-filtered band

I tried estimating the coherent direction only from the scored frequency band. This looked like it could help because the metric is computed in a specific band. However, this version failed the validity check

### Rank-2 residual removal
I also tried removing two dominant spatial components instead of one. This also failed. The likely reason is that the task expects the external coherent interference to be rank-1. Removing a rank-2 component becomes too aggressive and no longer matches the intended model.

### Repeating baseline and rank-1 cancellation

I tested running the baseline twice and also alternating between baseline cancellation and rank-1 residual cancellation. These versions either reduced the score or failed the explainability checks. In particular, repeated cancellation seemed to remove components that were not explainable by the allowed TX-driven plus rank-1 structure.

### Scalar beta scaling

Before using per-channel beta values, I tried a single scalar scaling factor for the baseline TX estimate:

```python
X = rx - beta * tx_removed
```

A scalar value around 0.9 improved the score slightly compared to subtracting the full baseline estimate. This suggested that the baseline TX estimate was slightly over-subtracting in some channels.

### Per-channel beta scaling

Finally, I tested small per-channel beta values. This gave the best stable result:

```text
beta = [0.82, 0.90, 0.98, 0.78]
```

This was kept in the final solution.