"""TDD: bootstrap confidence interval for outlier p-values.

A p-value computed against a finite ensemble has its own sampling uncertainty:
the same MCMC chain run again would give a slightly different distribution
and therefore a slightly different p-value. Bootstrap resampling lets us
report a confidence interval around our point estimate so the reader can
distinguish "p = 0.01 ± 0.005" (firm outlier) from "p = 0.01 ± 0.04" (could
just be noise).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_bootstrap_p_value_returns_three_floats():
    """Returns (point_estimate, lower, upper)."""
    from gerrydetect.analysis import bootstrap_p_value

    rng = np.random.default_rng(0)
    dist = rng.normal(size=500)
    result = bootstrap_p_value(value=2.0, distribution=dist, n_boot=200, seed=1)
    assert isinstance(result, tuple) and len(result) == 3
    p, lo, hi = result
    assert all(isinstance(x, float) for x in (p, lo, hi))
    assert lo <= p <= hi


def test_bootstrap_p_value_in_unit_interval():
    from gerrydetect.analysis import bootstrap_p_value

    rng = np.random.default_rng(1)
    dist = rng.normal(size=500)
    p, lo, hi = bootstrap_p_value(value=0.0, distribution=dist, n_boot=200, seed=1)
    for x in (p, lo, hi):
        assert 0.0 <= x <= 1.0


def test_bootstrap_value_far_outside_distribution_gives_low_p():
    """A value 10 standard deviations from the median has p ≈ 0 with tight CI."""
    from gerrydetect.analysis import bootstrap_p_value

    rng = np.random.default_rng(2)
    dist = rng.normal(size=2000)
    p, lo, hi = bootstrap_p_value(
        value=10.0, distribution=dist, n_boot=500, seed=3
    )
    assert p < 0.005
    assert hi - lo < 0.01


def test_bootstrap_value_at_median_gives_high_p():
    """Value at the median of the distribution gives p close to 1."""
    from gerrydetect.analysis import bootstrap_p_value

    rng = np.random.default_rng(3)
    dist = rng.normal(size=2000)
    p, lo, hi = bootstrap_p_value(
        value=float(np.median(dist)),
        distribution=dist,
        n_boot=500,
        seed=4,
    )
    assert p > 0.5


def test_bootstrap_constant_distribution_handles_gracefully():
    """If every ensemble value is identical, p-value is 0 if the enacted differs,
    1 if it matches; CI collapses."""
    from gerrydetect.analysis import bootstrap_p_value

    dist = np.full(100, 0.42)
    p_match, lo_m, hi_m = bootstrap_p_value(
        value=0.42, distribution=dist, n_boot=100, seed=5
    )
    assert p_match == pytest.approx(1.0)
    p_diff, lo_d, hi_d = bootstrap_p_value(
        value=1.0, distribution=dist, n_boot=100, seed=5
    )
    assert p_diff == pytest.approx(0.0)


def test_bootstrap_reproducible_with_seed():
    from gerrydetect.analysis import bootstrap_p_value

    rng = np.random.default_rng(99)
    dist = rng.normal(size=300)
    a = bootstrap_p_value(value=1.5, distribution=dist, n_boot=200, seed=7)
    b = bootstrap_p_value(value=1.5, distribution=dist, n_boot=200, seed=7)
    assert a == b
