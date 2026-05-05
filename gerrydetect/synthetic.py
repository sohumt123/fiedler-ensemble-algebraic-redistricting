"""Synthetic precinct graph generator (Delaunay triangulation + partisan gradient)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


def _density_field(x: np.ndarray, y: np.ndarray, urban_centers: np.ndarray) -> np.ndarray:
    """Population density at points (x, y).

    Urban centers contribute a Gaussian bump; the baseline is uniform low
    density. Returns un-normalized density values >= 0.
    """
    base = 0.4 * np.ones_like(x)
    for cx, cy, radius in urban_centers:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        base += np.exp(-d2 / (2 * radius ** 2))
    return base


def make_synthetic_state(
    n_precincts: int,
    n_districts: int,
    seed: int = 0,
    n_urban_centers: int = 2,
    pop_min: int = 600,
    pop_max: int = 2200,
    border_warp: float = 0.15,
) -> nx.Graph:
    """Generate a synthetic state-shaped precinct adjacency graph.

    Args:
        n_precincts: target number of precincts (graph nodes).
        n_districts: number of congressional districts (carried in
            `graph.graph['n_districts']`; downstream code uses this as `k`).
        seed: RNG seed for reproducibility.
        n_urban_centers: how many high-density "city" bumps to plant.
        pop_min, pop_max: bounds on per-precinct population.
        border_warp: how irregular the state's outer boundary is (0 = a
            square, 0.3 = visibly stateline-shaped).

    Returns:
        A connected `nx.Graph` with `n_precincts` nodes. Each node has
        `pop`, `votes_d`, `votes_r`, `x`, `y` attributes. Edge weights are
        Euclidean distances between centroids.
    """
    rng = np.random.default_rng(seed)

    # ---- 1. sample centroids inside a slightly warped rectangle ---------
    # Reject samples falling outside the warped boundary.
    width, height = 10.0, 6.0
    points: list[tuple[float, float]] = []
    while len(points) < n_precincts:
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        # Sinusoidal border warp: exclude a rolling sliver on the left edge.
        warp = border_warp * height * np.sin(2 * np.pi * y / height)
        if x < warp + 0.01:
            continue
        points.append((x, y))
    pts = np.array(points)

    # ---- 2. urban centers (a few hot spots) -----------------------------
    # Place urban centers away from the borders.
    centers = np.array(
        [
            (
                rng.uniform(width * 0.25, width * 0.85),
                rng.uniform(height * 0.20, height * 0.80),
                rng.uniform(0.6, 1.4),
            )
            for _ in range(n_urban_centers)
        ]
    )

    # ---- 3. populations -------------------------------------------------
    density = _density_field(pts[:, 0], pts[:, 1], centers)
    # Map density to pop, with extra noise so it is not deterministic.
    norm = (density - density.min()) / (density.max() - density.min() + 1e-9)
    pops = pop_min + norm * (pop_max - pop_min)
    pops = pops * rng.uniform(0.85, 1.15, size=len(pops))
    pops = np.clip(pops, pop_min * 0.5, None)

    # ---- 4. partisan share: urban-D / rural-R gradient + noise ---------
    # higher density → higher D-share, scaled to roughly [0.30, 0.75].
    d_share = 0.30 + 0.45 * norm + rng.normal(0, 0.07, size=len(pops))
    d_share = np.clip(d_share, 0.05, 0.95)
    total_votes = pops * rng.uniform(0.4, 0.7, size=len(pops))
    votes_d = total_votes * d_share
    votes_r = total_votes * (1.0 - d_share)

    # ---- 5. adjacency via Delaunay triangulation -----------------------
    tri = Delaunay(pts)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = simplex[i], simplex[j]
            if a == b:
                continue
            edges.add((min(int(a), int(b)), max(int(a), int(b))))

    # ---- 6. assemble graph ---------------------------------------------
    g = nx.Graph(n_districts=n_districts, seed=seed, source="synthetic")
    for i in range(n_precincts):
        g.add_node(
            i,
            pop=float(pops[i]),
            votes_d=float(votes_d[i]),
            votes_r=float(votes_r[i]),
            x=float(pts[i, 0]),
            y=float(pts[i, 1]),
            centroid=(float(pts[i, 0]), float(pts[i, 1])),
        )
    for u, v in edges:
        d = float(np.hypot(pts[u, 0] - pts[v, 0], pts[u, 1] - pts[v, 1]))
        g.add_edge(u, v, weight=d)

    # If for any reason the triangulation produced a disconnected graph
    # (extremely unlikely except in degenerate point sets), stitch it.
    if not nx.is_connected(g):
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        main = list(components[0])
        for fragment in components[1:]:
            v = next(iter(fragment))
            distances = np.hypot(
                pts[main, 0] - pts[v, 0], pts[main, 1] - pts[v, 1]
            )
            partner = main[int(np.argmin(distances))]
            g.add_edge(v, partner, weight=float(distances.min()), synthetic=True)
            main += list(fragment)

    return g
