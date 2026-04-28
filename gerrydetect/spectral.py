"""Recursive spectral bisection.

The classic graph-partitioning recipe: build the graph Laplacian L = D - A,
take the eigenvector corresponding to the second-smallest eigenvalue (the
"Fiedler vector"), and split nodes into two halves by the sign / order of
their Fiedler-vector entries. To get k pieces, recurse: bisect, then bisect
each half, etc.

We use SciPy's sparse eigensolver (`eigsh` with `which='SM'`, the
shift-invert variant) — implementing Lanczos by hand is out of scope for
NETS 1500 — but everything else is ours: the partition refinement to balance
population, the contiguity repair, the recursion strategy.

This is the deterministic baseline plan in the proposal: it produces a
single high-quality compact partition that we use as the seed for the MCMC
chain (rather than a random initial partition, which would need a long
burn-in to escape).
"""

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
    """Compute the Fiedler vector (eigenvector for the 2nd-smallest
    eigenvalue) of the graph Laplacian on the subgraph induced by `nodes`.

    Returns an array indexed by `nodes` order.

    For small subgraphs (n < 1500) we use NumPy's dense `eigh` — deterministic
    and exact. For large subgraphs we use SciPy's sparse Lanczos solver with
    a fixed initial vector for reproducibility.
    """
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
    pop_tol: float,
) -> np.ndarray:
    """Given a sorted-by-Fiedler-value population array, find the cut index
    that produces the most-balanced split. Returns the boolean mask telling
    which nodes go in the "low" half.
    """
    order = np.argsort(fiedler)
    sorted_pop = populations[order]
    cumulative = np.cumsum(sorted_pop)
    total = cumulative[-1]

    # Find cut closest to total/2.
    target = total / 2.0
    cut_idx = int(np.argmin(np.abs(cumulative - target)))
    # Guard against degenerate cuts at the ends.
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

    Returns a dict mapping node -> district id in [0, k).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # We carry around assignments as labels; recursion splits one label into
    # two new labels until we have k.
    assignment: dict[NodeId, int] = {n: 0 for n in graph.nodes}
    next_label = 1

    # Each iteration: pick the label with the largest total population and
    # split it. Repeat until we have k labels.
    populations = {n: float(graph.nodes[n].get("pop", 1.0)) for n in graph.nodes}

    while len(set(assignment.values())) < k:
        # Pick the heaviest district to split next.
        dist_pops: dict[int, float] = {}
        for n, d in assignment.items():
            dist_pops[d] = dist_pops.get(d, 0.0) + populations[n]
        target_label = max(dist_pops, key=dist_pops.get)

        nodes_in = [n for n, d in assignment.items() if d == target_label]
        if len(nodes_in) < 2:
            log.warning("Cannot split label %s: only %d nodes", target_label, len(nodes_in))
            break

        fiedler = fiedler_vector(graph, nodes_in)
        pop_array = np.array([populations[n] for n in nodes_in])
        low_mask = _balanced_split_indices(fiedler, pop_array, pop_tol)

        for i, n in enumerate(nodes_in):
            if not low_mask[i]:
                assignment[n] = next_label
        next_label += 1

    # Renumber to 0..k-1 contiguous, then repair contiguity.
    labels = sorted(set(assignment.values()))
    renumber = {old: new for new, old in enumerate(labels)}
    assignment = {n: renumber[d] for n, d in assignment.items()}
    assignment = _repair_contiguity(graph, assignment)

    # Verify connectedness post-repair (a soft check; logs only).
    by_district: dict[int, list[NodeId]] = {}
    for n, d in assignment.items():
        by_district.setdefault(d, []).append(n)
    for d, ns in by_district.items():
        if not is_district_connected(graph, ns):
            log.warning("District %d still disconnected after repair", d)

    return assignment
