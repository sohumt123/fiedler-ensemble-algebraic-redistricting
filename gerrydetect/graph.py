"""Build precinct adjacency graph from a GeoDataFrame."""

from __future__ import annotations

import logging
from typing import Mapping

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)


def _shared_boundary_length(geom_a, geom_b) -> float:
    """Length of the shared boundary between two polygons, in the same units
    as the geometries' CRS. Returns 0 if they only touch at a point.
    """
    inter = geom_a.boundary.intersection(geom_b.boundary)
    if inter.is_empty:
        return 0.0
    # `length` is 0 for points and >0 for line/multilinestring intersections.
    return float(inter.length)


def build_graph(
    gdf,
    pop_col: str = "pop",
    votes_d_col: str = "votes_d",
    votes_r_col: str = "votes_r",
    district_col: str | None = "district",
    epsilon: float = 1e-6,
) -> nx.Graph:
    """Construct the precinct adjacency graph from a GeoDataFrame.

    The GeoDataFrame index is used as node IDs. Geometries must be in a
    projected CRS (so `length` and `area` are meaningful); use
    `gdf.to_crs(epsg=...)` upstream to e.g. EPSG:5070 (NAD83 / Conus Albers)
    for U.S. states.

    `district_col` is the enacted district id; pass None if not available
    (e.g. while building a graph for a state whose enacted plan is not yet
    annotated).

    `epsilon` is the minimum shared boundary length to count as an edge
    (in CRS units, typically meters). Lower this for very small polygons.
    """
    try:
        from shapely.strtree import STRtree
    except ImportError as e:
        raise RuntimeError("shapely (>=2.0) is required for graph construction") from e

    geometries = list(gdf.geometry)
    indices = list(gdf.index)
    tree = STRtree(geometries)

    g = nx.Graph()
    for idx, row in gdf.iterrows():
        attrs: dict = {
            "pop": float(row.get(pop_col, 0) or 0),
            "votes_d": float(row.get(votes_d_col, 0) or 0),
            "votes_r": float(row.get(votes_r_col, 0) or 0),
            "centroid": (row.geometry.centroid.x, row.geometry.centroid.y),
        }
        if district_col is not None and district_col in gdf.columns:
            attrs["district"] = row[district_col]
        g.add_node(idx, **attrs)

    # Adjacency via shapely STRtree. Iterate every polygon, query its candidate
    # neighbors, then verify with `intersects` and shared-boundary-length check.
    edges_added = 0
    for i, geom in enumerate(geometries):
        # query() with a polygon returns indices of geometries whose envelopes
        # intersect; we then filter precisely.
        candidates = tree.query(geom)
        for j in candidates:
            j_int = int(j)
            if j_int <= i:
                continue
            other = geometries[j_int]
            if not geom.touches(other) and not geom.intersects(other):
                continue
            shared = _shared_boundary_length(geom, other)
            if shared < epsilon:
                continue
            g.add_edge(indices[i], indices[j_int], weight=shared)
            edges_added += 1

    log.info("Built graph: %d nodes, %d edges", g.number_of_nodes(), edges_added)

    return _stitch_islands(g, gdf)


def _stitch_islands(graph: nx.Graph, gdf) -> nx.Graph:
    """Reduce to the largest connected component, but instead of dropping
    smaller components, attach them by adding synthetic edges between each
    island and its nearest non-island precinct (centroid distance).
    """
    if graph.number_of_nodes() == 0:
        return graph
    components = list(nx.connected_components(graph))
    if len(components) == 1:
        return graph

    components.sort(key=len, reverse=True)
    main = components[0]
    main_centroids = np.array(
        [graph.nodes[n]["centroid"] for n in main]
    )
    main_nodes = list(main)

    # Walk every smaller component, link via the closest mainland precinct.
    for fragment in components[1:]:
        for v in fragment:
            cx, cy = graph.nodes[v]["centroid"]
            dx = main_centroids[:, 0] - cx
            dy = main_centroids[:, 1] - cy
            d2 = dx * dx + dy * dy
            j = int(np.argmin(d2))
            partner = main_nodes[j]
            distance = float(np.sqrt(d2[j]))
            graph.add_edge(v, partner, weight=distance, synthetic=True)
            log.info(
                "Stitched island node %s -> %s (distance %.0f)", v, partner, distance
            )
        main = main | fragment
        main_centroids = np.vstack(
            [
                main_centroids,
                np.array([graph.nodes[v]["centroid"] for v in fragment]),
            ]
        )
        main_nodes = main_nodes + list(fragment)

    return graph


def graph_summary(graph: nx.Graph) -> Mapping[str, float]:
    """Quick stats for logging / report figure 1."""
    pops = [d["pop"] for _, d in graph.nodes(data=True)]
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "total_pop": sum(pops),
        "mean_pop": float(np.mean(pops)) if pops else 0.0,
        "is_connected": nx.is_connected(graph),
    }
