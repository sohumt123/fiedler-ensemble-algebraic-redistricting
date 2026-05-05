"""Recursive spectral bisection via the Fiedler vector into k population-balanced pieces."""

from __future__ import annotations

import logging
from typing import Hashable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gerrydetect.contiguity import is_district_connected

NodeId = Hashable
log = logging.getLogger(__name__)


def fiedler_vector(graph: nx.Graph, nodes: list[NodeId] | None = None) -> np.ndarray:
    """Fiedler vector of the graph Laplacian on the subgraph induced by `nodes`."""
    if nodes is None:
        nodes = list(graph.nodes)
    sub = graph.subgraph(nodes)

    # Build sparse Laplacian with rows/cols ordered as `nodes`.
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows, cols, data = [], [], []
    deg = np.zeros(n)
    for u, v, edata in sub.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(edata.get("weight", 1.0))
        rows += [i, j]
        cols += [j, i]
        data += [-w, -w]
        deg[i] += w
        deg[j] += w
    rows += list(range(n))
    cols += list(range(n))
    data += list(deg)

    if n <= 2:
        return np.zeros(n)

    # For small subgraphs use the dense solver — deterministic, fast enough
    # below ~1500 nodes (subgraphs we recurse into will quickly fall below
    # this for any state-sized problem).
    if n < 1500:
        Ld = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).toarray()
        vals, vecs = np.linalg.eigh(Ld)
        # eigh returns eigenvalues in ascending order
        return vecs[:, 1]

    # Large subgraphs: sparse Lanczos. Pass a deterministic v0 to make the
    # solver reproducible (ARPACK uses a random v0 by default, which leads
    # to non-deterministic Fiedler vectors when eigenvalues are close).
    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    v0 = np.full(n, 1.0 / np.sqrt(n))
    try:
        # 'SM' (smallest magnitude) requires shift-invert; 'SA' (smallest
        # algebraic) on a positive-semidefinite Laplacian gives the same
        # answer and is more stable.
        vals, vecs = spla.eigsh(L, k=2, which="SA", tol=1e-7, v0=v0)
    except Exception:
        Ld = L.toarray()
        vals, vecs = np.linalg.eigh(Ld)
        return vecs[:, 1]
    order = np.argsort(vals)
    return vecs[:, order[1]]


def _balanced_split_indices(
    fiedler: np.ndarray,
    populations: np.ndarray,
    target_left_fraction: float = 0.5,
) -> np.ndarray:
    """Boolean mask putting ~`target_left_fraction` of population in the low-Fiedler half."""
    order = np.argsort(fiedler)
    sorted_pop = populations[order]
    cumulative = np.cumsum(sorted_pop)
    total = cumulative[-1]

    target = total * target_left_fraction
    cut_idx = int(np.argmin(np.abs(cumulative - target)))
    cut_idx = min(max(cut_idx, 1), len(fiedler) - 2)

    mask = np.zeros(len(fiedler), dtype=bool)
    mask[order[: cut_idx + 1]] = True
    return mask


def _repair_contiguity(
    graph: nx.Graph,
    assignment: dict[NodeId, int],
) -> dict[NodeId, int]:
    """If any district is disconnected, reassign small fragments to a
    neighboring district so every district induces a connected subgraph.

    Strategy: for each district, find connected components; keep the largest
    component, and reassign each smaller component to whichever neighboring
    district it has the most boundary edges with.
    """
    by_district: dict[int, list[NodeId]] = {}
    for n, d in assignment.items():
        by_district.setdefault(d, []).append(n)

    for dist_id, nodes in list(by_district.items()):
        sub = graph.subgraph(nodes)
        components = list(nx.connected_components(sub))
        if len(components) <= 1:
            continue
        components.sort(key=len, reverse=True)
        # Keep the largest; reassign smaller fragments.
        for fragment in components[1:]:
            # Tally boundary edges to each neighboring district.
            tallies: dict[int, int] = {}
            for v in fragment:
                for nbr in graph.neighbors(v):
                    nbr_d = assignment[nbr]
                    if nbr_d != dist_id:
                        tallies[nbr_d] = tallies.get(nbr_d, 0) + 1
            if not tallies:
                # Fragment is an island within the district — attach to the
                # nearest district by giving up; assign to dist_id+1 wraparound.
                target = (dist_id + 1) % max(by_district.keys())
            else:
                target = max(tallies, key=tallies.get)
            for v in fragment:
                assignment[v] = target
    return assignment


def recursive_bisect(
    graph: nx.Graph, k: int, pop_tol: float = 0.02
) -> dict[NodeId, int]:
    """Recursively bisect `graph` into k population-balanced pieces.

    The recursion is *proportional*: at each level we know the piece will
    become `k_target` districts, so we split it into a "left" piece destined
    to become `k_left = k_target // 2` districts and a "right" piece destined
    to become `k_right = k_target - k_left`. The Fiedler-ordered cut puts
    `k_left / k_target` of the population on the left. This works cleanly
    for any k, not just powers of 2.

    Returns a dict mapping node -> district id in [0, k).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    populations = {n: float(graph.nodes[n].get("pop", 1.0)) for n in graph.nodes}

    def split(nodes: list[NodeId], k_target: int, label_base: int) -> dict[NodeId, int]:
        if k_target == 1 or len(nodes) <= 1:
            return {n: label_base for n in nodes}
        k_left = k_target // 2
        k_right = k_target - k_left

        fiedler = fiedler_vector(graph, nodes)
        pop_array = np.array([populations[n] for n in nodes])
        target_frac = k_left / k_target
        low_mask = _balanced_split_indices(fiedler, pop_array, target_frac)

        left_nodes = [nodes[i] for i in range(len(nodes)) if low_mask[i]]
        right_nodes = [nodes[i] for i in range(len(nodes)) if not low_mask[i]]

        left = split(left_nodes, k_left, label_base)
        right = split(right_nodes, k_right, label_base + k_left)
        return {**left, **right}

    assignment = split(list(graph.nodes), k, 0)
    assignment = _repair_contiguity(graph, assignment)

    by_district: dict[int, list[NodeId]] = {}
    for n, d in assignment.items():
        by_district.setdefault(d, []).append(n)
    for d, ns in by_district.items():
        if not is_district_connected(graph, ns):
            log.warning("District %d still disconnected after repair", d)

    return assignment
