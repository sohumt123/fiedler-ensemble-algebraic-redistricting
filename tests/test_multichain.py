"""TDD: multi-chain MCMC runner.

The runner launches several MCMC chains from independent seeds, returns
their samples and per-chain stats, and computes R-hat across chains for
any scalar metric. This is the standard tool to detect under-mixing — if
R-hat is high, the chains haven't agreed on the distribution yet.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from gerrydetect.spectral import recursive_bisect


@pytest.fixture
def small_graph():
    g = nx.grid_2d_graph(8, 8)
    g = nx.convert_node_labels_to_integers(g)
    rng = np.random.default_rng(0)
    for n in g.nodes:
        g.nodes[n]["pop"] = float(rng.integers(900, 1100))
        g.nodes[n]["votes_d"] = float(rng.uniform(200, 400))
        g.nodes[n]["votes_r"] = float(rng.uniform(200, 400))
    for u, v in g.edges:
        g.edges[u, v]["weight"] = 1.0
    return g


def test_run_multichain_returns_one_sample_set_per_chain(small_graph):
    """Each chain must produce exactly the requested number of samples."""
    from gerrydetect.multichain import run_multichain

    seed = recursive_bisect(small_graph, k=4, pop_tol=0.05)
    result = run_multichain(
        small_graph,
        seed_assignment=seed,
        n_chains=3,
        n_steps=10,
        lag=2,
        burn_in=20,
        pop_tol=0.10,
        seeds=[1, 2, 3],
        show_progress=False,
    )
    assert len(result.chains) == 3
    for samples in result.chains:
        assert len(samples) == 10


def test_run_multichain_distinct_seeds_give_distinct_trajectories(small_graph):
    """Two chains from different seeds must not produce identical samples."""
    from gerrydetect.multichain import run_multichain

    seed = recursive_bisect(small_graph, k=4, pop_tol=0.05)
    result = run_multichain(
        small_graph,
        seed_assignment=seed,
        n_chains=2,
        n_steps=10,
        lag=2,
        burn_in=20,
        pop_tol=0.10,
        seeds=[1, 2],
        show_progress=False,
    )
    a = [s.assignment for s in result.chains[0]]
    b = [s.assignment for s in result.chains[1]]
    # At least one sample pair must differ.
    differs = any(a[i] != b[i] for i in range(len(a)))
    assert differs


def test_run_multichain_same_seed_reproducible(small_graph):
    """Two runs with the same seeds must produce identical sample sequences."""
    from gerrydetect.multichain import run_multichain

    seed = recursive_bisect(small_graph, k=4, pop_tol=0.05)
    kwargs = dict(
        graph=small_graph,
        seed_assignment=seed,
        n_chains=2,
        n_steps=8,
        lag=2,
        burn_in=10,
        pop_tol=0.10,
        seeds=[42, 43],
        show_progress=False,
    )
    a = run_multichain(**kwargs)
    b = run_multichain(**kwargs)
    for ca, cb in zip(a.chains, b.chains):
        for sa, sb in zip(ca, cb):
            assert sa.assignment == sb.assignment


def test_multichain_metric_trajectories_shape_matches_chains(small_graph):
    """metric_trajectories(metric_fn) returns shape (n_chains, n_samples)."""
    from gerrydetect.metrics import cut_edge_ratio
    from gerrydetect.multichain import run_multichain

    seed = recursive_bisect(small_graph, k=4, pop_tol=0.05)
    result = run_multichain(
        small_graph,
        seed_assignment=seed,
        n_chains=4,
        n_steps=10,
        lag=2,
        burn_in=10,
        pop_tol=0.10,
        seeds=[1, 2, 3, 4],
        show_progress=False,
    )
    traj = result.metric_trajectories(cut_edge_ratio)
    assert traj.shape == (4, 10)


def test_multichain_rhat_for_well_mixed_chains_is_close_to_one(small_graph):
    """If chains start from the same seed partition and run long enough,
    R-hat for the metric trajectories should be close to 1 (well-mixed)."""
    from gerrydetect.metrics import cut_edge_ratio
    from gerrydetect.multichain import run_multichain

    seed = recursive_bisect(small_graph, k=4, pop_tol=0.05)
    result = run_multichain(
        small_graph,
        seed_assignment=seed,
        n_chains=4,
        n_steps=100,
        lag=5,
        burn_in=200,
        pop_tol=0.10,
        seeds=[10, 20, 30, 40],
        show_progress=False,
    )
    rhat = result.rhat(cut_edge_ratio)
    # Generous bound — chains might not have fully mixed at this length, but
    # they should at least not be wildly disagreeing.
    assert rhat < 1.5
