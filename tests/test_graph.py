"""Graph builder smoke test on synthetic polygons.

We avoid pulling real shapefiles into the test suite; instead we construct
a small set of square polygons whose adjacencies are easy to predict, then
verify build_graph wires up the right edges.
"""

from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import box

from gerrydetect.graph import build_graph


def _three_in_a_row_gdf():
    # Three unit squares stacked left-to-right:
    #   [0,1] [1,2] [2,3]
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame(
        {
            "pop": [100, 100, 100],
            "votes_d": [50, 60, 40],
            "votes_r": [50, 40, 60],
            "district": [0, 0, 1],
            "geometry": polys,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    return gdf


def test_build_graph_basic_adjacency():
    gdf = _three_in_a_row_gdf()
    g = build_graph(gdf)
    # Two adjacencies: 0-1 and 1-2; no 0-2.
    assert g.number_of_nodes() == 3
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 2)
    assert not g.has_edge(0, 2)


def test_build_graph_node_attrs_present():
    gdf = _three_in_a_row_gdf()
    g = build_graph(gdf)
    for n in g.nodes:
        assert "pop" in g.nodes[n]
        assert "votes_d" in g.nodes[n]
        assert "votes_r" in g.nodes[n]
        assert "district" in g.nodes[n]


def test_build_graph_edge_weight_is_shared_length():
    gdf = _three_in_a_row_gdf()
    g = build_graph(gdf)
    # Shared boundary between unit squares is length 1.
    assert g[0][1]["weight"] == pytest.approx(1.0)


def test_build_graph_island_stitching():
    """An isolated polygon should be stitched into the main component via a
    synthetic edge.
    """
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1)]
    gdf = gpd.GeoDataFrame(
        {
            "pop": [100, 100, 100],
            "votes_d": [50, 50, 50],
            "votes_r": [50, 50, 50],
            "district": [0, 0, 1],
            "geometry": polys,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    g = build_graph(gdf)
    # Node 2 is an island; it must be reachable from {0, 1} via a synthetic edge.
    import networkx as nx
    assert nx.is_connected(g)
