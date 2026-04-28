"""TDD: realistic synthetic state generator.

Real precinct adjacency graphs are *planar* (no edge crossings) with a
mostly-uniform interior and ragged borders. We mimic this by:

  1. sampling N centroids in a state-shaped bounding region,
  2. computing the Delaunay triangulation — each edge of the triangulation
     becomes an adjacency edge,
  3. assigning population from a low-frequency density field with one or
     two "urban centers" superimposed,
  4. assigning a D-share field that is correlated with that density (a
     stylized urban-D / rural-R gradient) plus per-precinct noise.

Tests verify connectivity, expected node count, population distribution,
and the urban-D correlation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest


def test_synthetic_state_has_requested_size():
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=300, n_districts=4, seed=0)
    assert g.number_of_nodes() == 300


def test_synthetic_state_is_connected():
    """Delaunay triangulation always produces a connected graph for N >= 3."""
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=200, n_districts=4, seed=1)
    assert nx.is_connected(g)


def test_synthetic_state_has_required_node_attrs():
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=100, n_districts=3, seed=2)
    sample = next(iter(g.nodes))
    attrs = g.nodes[sample]
    for key in ("pop", "votes_d", "votes_r", "x", "y"):
        assert key in attrs


def test_synthetic_state_population_strictly_positive():
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=200, n_districts=4, seed=3)
    pops = np.array([g.nodes[n]["pop"] for n in g.nodes])
    assert np.all(pops > 0)
    # Expect some heterogeneity — coefficient of variation > 5%.
    assert pops.std() / pops.mean() > 0.05


def test_synthetic_state_partisan_share_in_unit_interval():
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=200, n_districts=4, seed=4)
    for n in g.nodes:
        d = g.nodes[n]["votes_d"]
        r = g.nodes[n]["votes_r"]
        assert d >= 0 and r >= 0
        share = d / (d + r) if (d + r) > 0 else 0.5
        assert 0.0 <= share <= 1.0


def test_synthetic_state_urban_centers_lean_democratic():
    """The pop-weighted D-share in the densest 20% of precincts must exceed
    the pop-weighted D-share in the sparsest 20% — that is the deliberate
    urban-D / rural-R gradient our generator imprints.
    """
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=400, n_districts=8, seed=5)
    pops = np.array([g.nodes[n]["pop"] for n in g.nodes])
    shares = np.array(
        [
            g.nodes[n]["votes_d"]
            / max(g.nodes[n]["votes_d"] + g.nodes[n]["votes_r"], 1e-9)
            for n in g.nodes
        ]
    )
    order = np.argsort(pops)
    cutoff = len(order) // 5
    rural_pops = pops[order[:cutoff]]
    rural_shares = shares[order[:cutoff]]
    urban_pops = pops[order[-cutoff:]]
    urban_shares = shares[order[-cutoff:]]
    rural_d = (rural_shares * rural_pops).sum() / rural_pops.sum()
    urban_d = (urban_shares * urban_pops).sum() / urban_pops.sum()
    assert urban_d > rural_d + 0.05  # at least 5 percentage points of gradient


def test_synthetic_state_supports_district_count_metadata():
    """Generator should advertise the intended number of districts via a
    graph attribute, so downstream code knows what k to bisect to."""
    from gerrydetect.synthetic import make_synthetic_state

    g = make_synthetic_state(n_precincts=120, n_districts=6, seed=6)
    assert g.graph.get("n_districts") == 6


def test_synthetic_state_reproducible():
    """Same seed must give the same graph (node attrs identical)."""
    from gerrydetect.synthetic import make_synthetic_state

    a = make_synthetic_state(n_precincts=80, n_districts=3, seed=99)
    b = make_synthetic_state(n_precincts=80, n_districts=3, seed=99)
    for n in a.nodes:
        assert a.nodes[n]["pop"] == b.nodes[n]["pop"]
        assert a.nodes[n]["votes_d"] == b.nodes[n]["votes_d"]
