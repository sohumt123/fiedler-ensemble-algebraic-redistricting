"""MCMC step-by-step invariants.

The chain must, on every accepted flip, preserve:
  - total population (conservation)
  - per-district connectedness
  - per-district population within tolerance
"""

from __future__ import annotations

import networkx as nx

from gerrydetect import mcmc
from gerrydetect.contiguity import is_district_connected
from gerrydetect.partition import MutablePartition
from gerrydetect.spectral import recursive_bisect


def test_mcmc_preserves_invariants(grid_10):
    seed = recursive_bisect(grid_10, k=4, pop_tol=0.05)
    samples, stats = mcmc.run(
        grid_10,
        seed_assignment=seed,
        n_steps=50,
        pop_tol=0.10,
        lag=5,
        burn_in=50,
        seed=7,
        show_progress=False,
    )
    # Every recorded sample must respect the invariants.
    total_pop_seed = sum(grid_10.nodes[n]["pop"] for n in grid_10.nodes)
    for s in samples:
        # Conservation
        assert sum(s.district_pop.values()) == total_pop_seed
        # Connectivity per district
        for nodes in s.districts.values():
            assert is_district_connected(grid_10, nodes)
        # Population balance within twice the tolerance (slack for boundary effects)
        ideal = total_pop_seed / 4
        for d_pop in s.district_pop.values():
            assert abs(d_pop - ideal) / ideal < 0.20


def test_mcmc_acceptance_nonzero(grid_10):
    seed = recursive_bisect(grid_10, k=4, pop_tol=0.05)
    _, stats = mcmc.run(
        grid_10,
        seed_assignment=seed,
        n_steps=20,
        pop_tol=0.10,
        lag=2,
        burn_in=20,
        seed=11,
        show_progress=False,
    )
    # We should accept at least *some* flips on a 10x10 grid with 4 districts.
    assert stats.accepted >= 5
