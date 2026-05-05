"""Full-pipeline smoke test on a tiny synthetic state. Usage: python scripts/smoke_test.py"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import networkx as nx
import numpy as np

from gerrydetect import mcmc
from gerrydetect.analysis import compute_metrics_on_ensemble, outlier_analysis
from gerrydetect.metrics import all_metrics
from gerrydetect.partition import Partition
from gerrydetect.spectral import recursive_bisect
from gerrydetect.viz import plot_histogram

REPO_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke")


def make_synthetic_state(side: int = 10, seed: int = 1) -> nx.Graph:
    """A `side`x`side` grid graph with random pop/vote attributes."""
    rng = np.random.default_rng(seed)
    g = nx.grid_2d_graph(side, side)
    g = nx.convert_node_labels_to_integers(g)
    for n in g.nodes:
        g.nodes[n]["pop"] = float(rng.integers(800, 1200))
        d = float(rng.uniform(200, 600))
        r = float(rng.uniform(200, 600))
        g.nodes[n]["votes_d"] = d
        g.nodes[n]["votes_r"] = r
    for u, v in g.edges:
        g.edges[u, v]["weight"] = 1.0
    return g


def main() -> int:
    log.info("Building 10x10 synthetic state ...")
    graph = make_synthetic_state(side=10, seed=1)
    log.info("  %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    log.info("Running spectral bisection (k=4) ...")
    seed_assign = recursive_bisect(graph, k=4, pop_tol=0.05)
    seed_partition = Partition(graph, seed_assign)
    log.info("  district pops: %s", dict(seed_partition.district_pop))

    log.info("Running MCMC (200 saved plans, lag=20, burn-in=200) ...")
    samples, stats = mcmc.run(
        graph,
        seed_assign,
        n_steps=200,
        pop_tol=0.05,
        lag=20,
        burn_in=200,
        seed=42,
        show_progress=False,
    )
    log.info(
        "  proposed=%d, accepted=%d, accept_rate=%.3f",
        stats.proposed, stats.accepted, stats.acceptance_rate(),
    )

    log.info("Computing metrics on ensemble ...")
    df = compute_metrics_on_ensemble(samples)
    log.info("  ensemble cut_edge_ratio: mean=%.4f std=%.4f",
             df["cut_edge_ratio"].mean(), df["cut_edge_ratio"].std())

    enacted_metrics = all_metrics(seed_partition)  # treat spectral as "enacted"
    results = outlier_analysis(enacted_metrics, df)
    log.info("Outlier analysis vs spectral baseline:")
    for r in results:
        log.info(
            "  %-18s  enacted=%.4f  pct=%.1f  p=%.3f",
            r.metric, r.enacted_value, r.percentile, r.p_value_two_sided,
        )

    fig_dir = REPO_ROOT / "output" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "smoke_cut_edge_ratio.png"
    plot_histogram(
        df["cut_edge_ratio"].to_numpy(),
        enacted_metrics["cut_edge_ratio"],
        title="smoke test: cut edge ratio across MCMC ensemble",
        xlabel="cut edge ratio",
        savepath=fig_path,
        p_value=results[0].p_value_two_sided if results else None,
    )
    log.info("Wrote %s", fig_path)
    log.info("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
