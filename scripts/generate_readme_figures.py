"""Generate pipeline figures for docs/. Usage: python scripts/generate_readme_figures.py"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gerrydetect import mcmc
from gerrydetect.analysis import compute_metrics_on_ensemble, outlier_analysis
from gerrydetect.metrics import all_metrics, seats_votes_curve
from gerrydetect.partition import Partition
from gerrydetect.spectral import recursive_bisect

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("readme-figs")


def make_synthetic_state(side: int = 14, seed: int = 7) -> nx.Graph:
    """A `side`x`side` grid with mildly non-uniform pop and a deliberate
    partisan gradient — D-leaning in the south-west, R-leaning in the
    north-east. This makes the seats-votes curve and partisan metrics
    non-trivial.
    """
    rng = np.random.default_rng(seed)
    g = nx.grid_2d_graph(side, side)
    # Carry the (x, y) coordinates as node attributes for the choropleth.
    for (x, y) in list(g.nodes):
        g.nodes[(x, y)]["coord"] = (float(x), float(y))
    g = nx.convert_node_labels_to_integers(g, label_attribute="coord_pair")
    for n in g.nodes:
        x, y = g.nodes[n]["coord_pair"]
        g.nodes[n]["pop"] = float(rng.integers(900, 1100))
        # Smooth partisan gradient + noise.
        d_share_base = 0.35 + 0.30 * (1.0 - (x + y) / (2 * (side - 1)))
        d_share = float(np.clip(d_share_base + rng.normal(0, 0.05), 0.05, 0.95))
        total = float(rng.integers(400, 600))
        g.nodes[n]["votes_d"] = total * d_share
        g.nodes[n]["votes_r"] = total * (1 - d_share)
    for u, v in g.edges:
        g.edges[u, v]["weight"] = 1.0
    return g


def _grid_coords(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    return {n: tuple(graph.nodes[n]["coord_pair"]) for n in graph.nodes}


def plot_district_grid(graph, assignment, ax, title, side: int):
    """Tile a `side`x`side` grid colored by district id. The graph carries
    `coord_pair` attributes giving each node's (x, y) grid position.
    """
    coords = _grid_coords(graph)
    cmap = plt.colormaps["tab10"].resampled(max(assignment.values()) + 1)
    img = np.zeros((side, side, 3))
    for n, (x, y) in coords.items():
        img[int(y), int(x)] = cmap(assignment[n])[:3]
    ax.imshow(img, origin="lower", extent=(-0.5, side - 0.5, -0.5, side - 0.5))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for x in np.arange(-0.5, side, 1):
        ax.axvline(x, color="white", linewidth=0.4, alpha=0.3)
    for y in np.arange(-0.5, side, 1):
        ax.axhline(y, color="white", linewidth=0.4, alpha=0.3)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    side = 14
    log.info("Building synthetic %dx%d state ...", side, side)
    graph = make_synthetic_state(side=side, seed=7)
    log.info(
        "  %d precincts, %d edges", graph.number_of_nodes(), graph.number_of_edges()
    )

    log.info("Running spectral bisection (k=4) ...")
    seed_assign = recursive_bisect(graph, k=4, pop_tol=0.02)
    seed_partition = Partition(graph, seed_assign)
    log.info("  district pops: %s", dict(seed_partition.district_pop))

    # Treat the spectral baseline as the "enacted" plan for figure purposes.
    # Then run a longer MCMC chain so we have a real distribution.
    log.info("Running MCMC (500 saved plans) ...")
    samples, stats = mcmc.run(
        graph,
        seed_assign,
        n_steps=500,
        pop_tol=0.04,
        lag=20,
        burn_in=2000,
        seed=42,
        show_progress=False,
    )
    log.info(
        "  proposed=%d accepted=%d acc_rate=%.3f",
        stats.proposed, stats.accepted, stats.acceptance_rate(),
    )

    log.info("Computing metrics on ensemble ...")
    ens_df = compute_metrics_on_ensemble(samples)
    enacted_metrics = all_metrics(seed_partition)
    results = outlier_analysis(enacted_metrics, ens_df)

    # ---------- Figure 1: histogram panel (3 metrics side by side) ----------
    log.info("Writing histograms panel ...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panel_metrics = [
        ("cut_edge_ratio", "cut edge ratio", "lower = more compact"),
        ("modularity", "Newman–Girvan modularity", "higher = more clustered"),
        ("efficiency_gap", "efficiency gap", "0 = symmetric"),
    ]
    for ax, (metric, xlabel, subtitle) in zip(axes, panel_metrics):
        values = ens_df[metric].to_numpy()
        enacted = enacted_metrics[metric]
        # find the OutlierResult for this metric for the p-value
        r = next((rr for rr in results if rr.metric == metric), None)
        ax.hist(values, bins=30, color="#a8c5e6", edgecolor="white")
        ax.axvline(enacted, color="#c0392b", linewidth=2.5, label=f"baseline = {enacted:.4f}")
        ax.set_title(f"{xlabel}\n({subtitle})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("ensemble plans")
        if r is not None:
            ax.text(
                0.02, 0.97,
                f"p = {r.p_value_two_sided:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"),
            )
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Spectral baseline vs. MCMC ensemble (synthetic 14×14 state, k=4 districts)",
        fontsize=12,
    )
    fig.tight_layout()
    p = FIG_DIR / "histograms_panel.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", p)

    # ---------- Figure 2: district map comparison ----------
    log.info("Writing district map comparison ...")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    plot_district_grid(graph, seed_assign, axes[0], "spectral baseline", side)
    for i, idx in enumerate([0, len(samples) // 2, len(samples) - 1]):
        plot_district_grid(
            graph,
            samples[idx].assignment,
            axes[i + 1],
            f"MCMC sample #{idx}",
            side,
        )
    fig.suptitle(
        "Spectral baseline (left) vs. three MCMC ensemble draws", fontsize=12
    )
    fig.tight_layout()
    p = FIG_DIR / "district_maps.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", p)

    # ---------- Figure 3: seats-votes overlay ----------
    log.info("Computing seats-votes curves ...")
    enacted_curve = seats_votes_curve(seed_partition, swing_range=0.2, n_points=41)
    ensemble_curves = [
        seats_votes_curve(s, swing_range=0.2, n_points=41)
        for s in samples[::20]  # sub-sample 25 curves for visual clarity
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in ensemble_curves:
        ax.plot(c.statewide_d_share, c.expected_d_seats, color="#bbbbbb", linewidth=0.8, alpha=0.7)
    ax.plot(
        enacted_curve.statewide_d_share,
        enacted_curve.expected_d_seats,
        color="#c0392b",
        linewidth=2.5,
        label="spectral baseline",
    )
    # symmetric ideal: y = N * (x - 0.5) + N/2 capped to [0, N]
    n_seats = len(seed_partition.districts)
    xs = enacted_curve.statewide_d_share
    ideal = np.clip(n_seats * (xs - 0.5) + n_seats / 2, 0, n_seats)
    ax.plot(xs, ideal, color="black", linestyle="--", linewidth=1.2, label="symmetric ideal")
    ax.set_xlabel("statewide D vote share")
    ax.set_ylabel("expected D seats")
    ax.set_title("Seats–votes curves: ensemble (grey) vs. spectral baseline (red)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "seats_votes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", p)

    # ---------- Figure 4: pipeline diagram ----------
    log.info("Writing pipeline diagram ...")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, "Shapefiles\n+ election\n+ ACS pop", "#cfd8dc"),
        (1.95, "Adjacency\ngraph", "#b0bec5"),
        (3.7, "Partition\nlayer", "#90a4ae"),
        (5.45, "Spectral\nbisect", "#a8c5e6"),
        (5.45, "Single-flip\nMCMC", "#a8c5e6"),
        (7.55, "Compactness\n+ partisan\nmetrics", "#80cbc4"),
        (9.45, "Outlier\nanalysis\n+ figures", "#a5d6a7"),
    ]
    # We draw the two samplers stacked vertically at the same x.
    y_main = 1.2
    for i, (x, label, color) in enumerate(boxes):
        if i == 4:
            y = 0.2
            h = 0.9
        elif i == 3:
            y = 1.5
            h = 0.9
        else:
            y = y_main
            h = 1.0
        rect = plt.Rectangle((x, y), 1.6, h, facecolor=color, edgecolor="#37474f", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.8, y + h / 2, label, ha="center", va="center", fontsize=9)
    arrow_kwargs = dict(arrowstyle="->", color="#37474f", linewidth=1.4)
    arrows = [
        ((1.8, 1.7), (1.95, 1.7)),
        ((3.55, 1.7), (3.7, 1.7)),
        ((5.3, 1.7), (5.45, 1.95)),  # to spectral
        ((5.3, 1.7), (5.45, 0.65)),  # to mcmc
        ((7.05, 1.95), (7.55, 1.7)),  # spectral -> metrics
        ((7.05, 0.65), (7.55, 1.7)),  # mcmc -> metrics
        ((9.15, 1.7), (9.45, 1.7)),
    ]
    for src, dst in arrows:
        ax.annotate("", xy=dst, xytext=src, arrowprops=arrow_kwargs)
    ax.set_title("Pipeline: data → graph → partition → samplers → metrics → outlier analysis", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "pipeline.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", p)

    log.info("Done. 4 figures in %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
