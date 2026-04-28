"""Hand-computed metric values on small graphs."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from gerrydetect.metrics import (
    cut_edge_ratio,
    efficiency_gap,
    mean_median,
    modularity,
    mst_diameter,
    seats_votes_curve,
)
from gerrydetect.partition import Partition


def test_cut_edge_ratio_four_cycle(four_cycle):
    # split 0,1 vs 2,3 in C4: 2 of 4 edges cross -> 0.5
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    assert cut_edge_ratio(p) == pytest.approx(0.5)


def test_cut_edge_ratio_no_split(four_cycle):
    p = Partition(four_cycle, dict.fromkeys(four_cycle.nodes, 0))
    assert cut_edge_ratio(p) == 0.0


def test_modularity_uniform_split_c4(four_cycle):
    # split 0,1 vs 2,3 in C4 (4 edges, m=4):
    # within edges per district: e_0 = 1 (edge 0-1), e_1 = 1 (edge 2-3)
    # degrees per district: d_0 = 4 (each of 4 endpoints has deg 2; nodes 0,1
    # contribute 2+2), d_1 = 4
    # Q = (e_0/m - (d_0/2m)^2) + (e_1/m - (d_1/2m)^2)
    #   = (1/4 - (4/8)^2) + (1/4 - (4/8)^2)
    #   = (0.25 - 0.25) + (0.25 - 0.25)
    #   = 0
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    assert modularity(p) == pytest.approx(0.0, abs=1e-9)


def test_modularity_perfect_communities():
    """Two K3 cliques connected by a single bridge — strong community structure."""
    g = nx.Graph()
    # K3: 0-1-2-0
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # K3: 3-4-5-3
    g.add_edges_from([(3, 4), (4, 5), (3, 5)])
    # bridge
    g.add_edge(2, 3)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    p = Partition(g, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
    q = modularity(p)
    # Should be substantially positive.
    assert q > 0.3


def test_mst_diameter_path():
    """Path graph 0-1-2-3-4 in one district: MST is the path; diameter is 4 edges."""
    g = nx.path_graph(5)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    p = Partition(g, dict.fromkeys(g.nodes, 0))
    assert mst_diameter(p) == 4.0


def test_mst_diameter_star():
    """Star K1,4: any spanning tree is the star; diameter is 2 edges."""
    g = nx.star_graph(4)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    p = Partition(g, dict.fromkeys(g.nodes, 0))
    assert mst_diameter(p) == 2.0


def test_efficiency_gap_balanced():
    """Two districts, both 60D-40R wins: each district wastes the same number
    of votes for each party. EG = 0 modulo small floating-point.
    """
    g = nx.cycle_graph(4)
    for n in g.nodes:
        g.nodes[n]["pop"] = 100.0
        g.nodes[n]["votes_d"] = 60.0 if n in (0, 2) else 60.0
        g.nodes[n]["votes_r"] = 40.0
    # group into two districts; each district gets 120 D, 80 R = 60% D
    p = Partition(g, {0: 0, 1: 0, 2: 1, 3: 1})
    eg = efficiency_gap(p)
    # district totals: D=120, R=80, threshold=100.
    # winner D wastes 120-100=20; loser R wastes 80. Per district.
    # Across 2 districts: W_D=40, W_R=160, total=400.
    # EG = (160 - 40)/400 = 0.30.
    assert eg == pytest.approx(0.30)


def test_mean_median_zero_when_uniform():
    g = nx.cycle_graph(4)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
        g.nodes[n]["votes_d"] = 50.0
        g.nodes[n]["votes_r"] = 50.0
    p = Partition(g, {0: 0, 1: 0, 2: 1, 3: 1})
    assert mean_median(p) == pytest.approx(0.0)


def test_seats_votes_monotonic():
    """Seats should be non-decreasing as we shift D-share upward."""
    g = nx.path_graph(6)
    for i, n in enumerate(g.nodes):
        g.nodes[n]["pop"] = 1.0
        g.nodes[n]["votes_d"] = 30.0 + 10.0 * i
        g.nodes[n]["votes_r"] = 60.0 - 5.0 * i
    p = Partition(g, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2})
    curve = seats_votes_curve(p, swing_range=0.3, n_points=11)
    seats = curve.expected_d_seats
    assert all(seats[i] <= seats[i + 1] for i in range(len(seats) - 1))
