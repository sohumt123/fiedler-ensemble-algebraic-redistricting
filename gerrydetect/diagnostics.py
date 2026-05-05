"""MCMC convergence diagnostics: R-hat, autocorrelation, ESS."""

from __future__ import annotations

import numpy as np


def gelman_rubin(chains: np.ndarray) -> float:
    """Brooks-Gelman R-hat for m chains of length n (shape: m × n)."""
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
    """ESS via Geyer's (1992) initial monotone sequence estimator."""
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
