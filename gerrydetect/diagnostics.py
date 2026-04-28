"""MCMC convergence diagnostics.

We track three quantities to evaluate whether our chain has produced a
well-mixed sample of the plan distribution:

- **Gelman-Rubin R-hat** across multiple chains. Values close to 1 (≤ 1.05
  is the conventional threshold) indicate that within-chain and between-chain
  variances agree.
- **Lagged autocorrelation** of any scalar metric across one chain.
- **Effective sample size** derived from integrated autocorrelation time.
  ESS << N is the warning sign that our N samples really only contain a few
  hundred effectively-independent draws.

All three operate on *metric trajectories* — a 1-D array of values one of our
metrics took at each saved sample. We never feed full partition objects in.
"""

from __future__ import annotations

import numpy as np


def gelman_rubin(chains: np.ndarray) -> float:
    """Brooks-Gelman R-hat for `m` chains of length `n`.

    Args:
        chains: shape (m, n). Each row is one chain's metric trajectory.

    Returns:
        R-hat. R-hat = 1.0 means the chains are statistically indistinguishable.

    Formula (the simple "PSRF" form, R = 1 exactly when chains are identical):
        B = n * Var(chain_means)               (between-chain variance)
        W = mean(within-chain variances)
        V_hat = W + B/n
        R-hat = sqrt(V_hat / W)

    The asymptotically equivalent (n-1)/n-corrected form is also widely used;
    we omit it here so that R-hat = 1.0 cleanly identifies identical chains.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must be 2-D: (n_chains, n_samples)")
    m, n = chains.shape
    if m < 2:
        raise ValueError("R-hat requires at least 2 chains")
    if n < 2:
        raise ValueError("R-hat requires at least 2 samples per chain")

    chain_means = chains.mean(axis=1)
    b = n * chain_means.var(ddof=1)               # between-chain
    w = chains.var(axis=1, ddof=1).mean()         # mean within-chain
    if w == 0.0:
        return 1.0
    v_hat = w + b / n
    return float(np.sqrt(v_hat / w))


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation function up to `max_lag`, normalized to lag 0.

    Returns array of length `max_lag + 1` where index k is rho_hat(k).
    rho_hat(0) is always 1.

    For a constant series (zero variance), returns [1, 0, 0, ...] to avoid
    divide-by-zero rather than NaN — the constant series has no autocovariance
    structure to report.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be < n ({n})")

    x_centered = x - x.mean()
    var = float(np.dot(x_centered, x_centered) / n)
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    if var == 0.0:
        return out
    for k in range(1, max_lag + 1):
        cov_k = float(np.dot(x_centered[: n - k], x_centered[k:]) / n)
        out[k] = cov_k / var
    return out


def effective_sample_size(series: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size from integrated autocorrelation time.

    Uses Geyer's (1992) initial monotone sequence estimator, truncated when
    the *sum of consecutive pairs* of autocorrelations becomes negative.

    Returns:
        ESS in [0, N]. Independent samples → ESS ≈ N. Heavily correlated
        samples → ESS << N.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_lag is None:
        # Cap at N/4; further lags are noise-dominated for any practical chain.
        max_lag = max(1, n // 4)
    max_lag = min(max_lag, n - 1)

    rho = autocorrelation(x, max_lag)
    if np.all(rho[1:] == 0):
        return float(n)

    # Sum consecutive pairs (rho_{2k} + rho_{2k+1}); stop once a pair is non-positive.
    tau = 1.0  # 1 + 2 * sum(rho_k)
    for k in range(1, max_lag, 2):
        pair = rho[k] + rho[k + 1] if (k + 1 <= max_lag) else rho[k]
        if pair <= 0:
            break
        tau += 2 * pair
    if tau <= 0:
        return float(n)
    return float(n / tau)
