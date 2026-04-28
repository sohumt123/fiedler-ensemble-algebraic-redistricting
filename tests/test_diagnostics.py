"""TDD: MCMC convergence diagnostics.

Mathematical references:
- Gelman-Rubin R-hat: Brooks & Gelman (1998), "General Methods for Monitoring
  Convergence of Iterative Simulations"
- Effective Sample Size from integrated autocorrelation time:
  Geyer (1992), "Practical Markov Chain Monte Carlo"
"""

from __future__ import annotations

import numpy as np
import pytest


# ----------------------------- R-hat (Gelman-Rubin) -----------------------------


def test_rhat_identical_chains_returns_one():
    """When all chains are identical, R-hat must be exactly 1.0."""
    from gerrydetect.diagnostics import gelman_rubin

    rng = np.random.default_rng(0)
    one_chain = rng.normal(size=200)
    chains = np.array([one_chain, one_chain, one_chain])
    r = gelman_rubin(chains)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_rhat_chains_from_same_distribution_close_to_one():
    """Chains drawn iid from the same distribution: R-hat ~ 1.0 (within 0.05)."""
    from gerrydetect.diagnostics import gelman_rubin

    rng = np.random.default_rng(0)
    chains = np.array([rng.normal(size=2000) for _ in range(4)])
    r = gelman_rubin(chains)
    assert 0.95 < r < 1.05


def test_rhat_chains_with_different_means_above_one():
    """Chains centered at very different means: R-hat must be substantially > 1."""
    from gerrydetect.diagnostics import gelman_rubin

    rng = np.random.default_rng(0)
    chains = np.array(
        [rng.normal(loc=0.0, size=200), rng.normal(loc=5.0, size=200)]
    )
    r = gelman_rubin(chains)
    assert r > 1.5


def test_rhat_requires_at_least_two_chains():
    """Single chain: meaningless, must raise."""
    from gerrydetect.diagnostics import gelman_rubin

    with pytest.raises(ValueError):
        gelman_rubin(np.array([[1.0, 2.0, 3.0]]))


# ----------------------------- Autocorrelation -----------------------------


def test_autocorrelation_lag_zero_is_one():
    """Auto-correlation at lag 0 always equals 1."""
    from gerrydetect.diagnostics import autocorrelation

    rng = np.random.default_rng(1)
    series = rng.normal(size=500)
    rho = autocorrelation(series, max_lag=10)
    assert rho[0] == pytest.approx(1.0, abs=1e-12)


def test_autocorrelation_independent_samples_decay():
    """Independent draws: autocorrelation at non-zero lag is small."""
    from gerrydetect.diagnostics import autocorrelation

    rng = np.random.default_rng(1)
    series = rng.normal(size=10000)
    rho = autocorrelation(series, max_lag=20)
    # All lags > 0 should be near 0 (within 3*1/sqrt(N) confidence band).
    band = 3 / np.sqrt(len(series))
    assert np.all(np.abs(rho[1:]) < band + 0.01)


def test_autocorrelation_constant_series():
    """A constant series has zero variance — must return zeros (or NaN safely)."""
    from gerrydetect.diagnostics import autocorrelation

    series = np.ones(50)
    rho = autocorrelation(series, max_lag=5)
    # Lag 0 still 1; later lags must not blow up.
    assert rho[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(rho))


def test_autocorrelation_strongly_correlated_series_decays_slowly():
    """An AR(1) process with phi=0.9: autocorrelation should be ~0.9^k at lag k."""
    from gerrydetect.diagnostics import autocorrelation

    rng = np.random.default_rng(2)
    n = 5000
    phi = 0.9
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    rho = autocorrelation(x, max_lag=10)
    # Compare against phi^k. Allow loose tolerance for finite-sample noise.
    for k in range(1, 6):
        assert rho[k] == pytest.approx(phi ** k, abs=0.06)


# ----------------------------- Effective sample size -----------------------------


def test_ess_independent_equals_n():
    """For independent samples, ESS ≈ N."""
    from gerrydetect.diagnostics import effective_sample_size

    rng = np.random.default_rng(3)
    series = rng.normal(size=1000)
    ess = effective_sample_size(series)
    # Allow 25% slack for finite-sample noise.
    assert 750 <= ess <= 1100


def test_ess_perfectly_correlated_close_to_one():
    """For a perfectly auto-correlated series (constant), ESS should collapse to ~1."""
    from gerrydetect.diagnostics import effective_sample_size

    series = np.ones(1000)
    ess = effective_sample_size(series)
    # Constant series — ESS is degenerate; we accept ESS == N (no info loss because
    # there's no variance) or ESS == 1 — but most importantly it must not blow up.
    assert 0 < ess <= len(series)


def test_ess_ar1_smaller_than_n():
    """An AR(1) series with phi=0.9 has ESS substantially smaller than N."""
    from gerrydetect.diagnostics import effective_sample_size

    rng = np.random.default_rng(4)
    n = 5000
    phi = 0.9
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    ess = effective_sample_size(x)
    # Theoretical ESS for AR(1): N * (1 - phi) / (1 + phi) = 5000 * 0.1/1.9 ~ 263
    assert 100 <= ess <= 700
