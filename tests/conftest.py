"""Shared pytest fixtures."""

from __future__ import annotations

import networkx as nx
import pytest


@pytest.fixture
def four_cycle() -> nx.Graph:
    """C4: 0-1-2-3-0. Two-district split should have cut ratio 0.5."""
    g = nx.cycle_graph(4)
    for n in g.nodes:
        g.nodes[n]["pop"] = 100.0
        g.nodes[n]["votes_d"] = 50.0
        g.nodes[n]["votes_r"] = 50.0
    return g


@pytest.fixture
def six_grid() -> nx.Graph:
    """A 2x3 grid (6 nodes, 7 edges) with uniform pop."""
    g = nx.grid_2d_graph(2, 3)
    g = nx.convert_node_labels_to_integers(g)
    for n in g.nodes:
        g.nodes[n]["pop"] = 100.0
        g.nodes[n]["votes_d"] = 60.0
        g.nodes[n]["votes_r"] = 40.0
    return g


@pytest.fixture
def grid_10() -> nx.Graph:
    """10x10 grid with random uniform-ish pops."""
    import numpy as np
    rng = np.random.default_rng(0)
    g = nx.grid_2d_graph(10, 10)
    g = nx.convert_node_labels_to_integers(g)
    for n in g.nodes:
        g.nodes[n]["pop"] = float(rng.integers(900, 1100))
        g.nodes[n]["votes_d"] = float(rng.uniform(300, 500))
        g.nodes[n]["votes_r"] = float(rng.uniform(300, 500))
    return g
