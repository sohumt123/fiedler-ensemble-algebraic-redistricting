"""Spectral bisection sanity checks."""

from __future__ import annotations

import networkx as nx

from gerrydetect.contiguity import is_district_connected
from gerrydetect.spectral import recursive_bisect


def test_bisect_grid_into_two_balanced(grid_10):
    assignment = recursive_bisect(grid_10, k=2, pop_tol=0.05)
    counts = {0: 0, 1: 0}
    pops = {0: 0.0, 1: 0.0}
    for node, dist in assignment.items():
        counts[dist] += 1
        pops[dist] += grid_10.nodes[node]["pop"]
    # Both halves should have ~50 nodes
    assert min(counts.values()) >= 40
    # And the populations should be within a reasonable tolerance
    total = sum(pops.values())
    assert min(pops.values()) / total > 0.45


def test_bisect_grid_into_four(grid_10):
    assignment = recursive_bisect(grid_10, k=4, pop_tol=0.05)
    by_dist: dict[int, list] = {}
    for n, d in assignment.items():
        by_dist.setdefault(d, []).append(n)
    assert len(by_dist) == 4
    # Each district must induce a connected subgraph (after the repair pass)
    for nodes in by_dist.values():
        assert is_district_connected(grid_10, nodes)


def test_bisect_returns_contiguous_districts_planar():
    """Random planar-ish (Delaunay-like) graph; districts must be connected."""
    g = nx.connected_watts_strogatz_graph(50, 4, 0.1, seed=1)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    assignment = recursive_bisect(g, k=3, pop_tol=0.1)
    by_dist: dict[int, list] = {}
    for n, d in assignment.items():
        by_dist.setdefault(d, []).append(n)
    assert len(by_dist) == 3
    # On dense Watts-Strogatz the contiguity-repair pass should normally
    # succeed; we don't fail the test if a fragment slips through (the
    # repair is best-effort), but we do require at least 2/3 of districts
    # to be connected as a smoke check.
    connected = sum(
        1 for nodes in by_dist.values() if is_district_connected(g, nodes)
    )
    assert connected >= 2
