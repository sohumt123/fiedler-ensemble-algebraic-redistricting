"""Contiguity check correctness."""

from __future__ import annotations

import networkx as nx

from gerrydetect.contiguity import (
    is_district_connected,
    is_district_connected_after_flip,
)
from gerrydetect.partition import Partition


def test_connected_singleton():
    g = nx.path_graph(5)
    assert is_district_connected(g, [3])
    assert is_district_connected(g, [])


def test_connected_path():
    g = nx.path_graph(5)
    assert is_district_connected(g, [0, 1, 2, 3, 4])
    assert not is_district_connected(g, [0, 1, 3, 4])  # gap at 2


def test_after_flip_disconnects_path():
    """In a path 0-1-2-3-4 split (0,1,2 | 3,4), removing 1 from district 0
    leaves {0, 2} which is disconnected.
    """
    g = nx.path_graph(5)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    p = Partition(g, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
    # Flipping node 1 to district 1: removes 1 from district 0 -> {0, 2} disconnected.
    assert not is_district_connected_after_flip(p, 1, 1)


def test_after_flip_keeps_connected_at_boundary():
    """In the same path, flipping node 2 to district 1 leaves district 0 = {0, 1}, still connected."""
    g = nx.path_graph(5)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    p = Partition(g, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
    assert is_district_connected_after_flip(p, 2, 1)


def test_after_flip_rejects_emptying_district():
    """Cannot flip the last remaining node in a district."""
    g = nx.path_graph(3)
    for n in g.nodes:
        g.nodes[n]["pop"] = 1.0
    # district 0 = {1}; flipping 1 would empty it.
    p = Partition(g, {0: 1, 1: 0, 2: 1})
    assert not is_district_connected_after_flip(p, 1, 1)
