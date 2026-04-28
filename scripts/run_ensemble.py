"""Run the spectral baseline + MCMC ensemble for a state.

Usage:
    python scripts/run_ensemble.py pa --n 1000 --lag 100 --burn-in 10000

Writes to data/ensembles/<state>_*.parquet:
  - assignments per saved plan (one row per plan, one column per node)
  - per-plan metric values (one row per plan, one column per metric)

The seed partition is the recursive spectral bisection of the precinct
graph (k = number of districts in the enacted plan), so the chain starts
from a population-balanced compact baseline.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("run_ensemble")


def main(args) -> int:
    from gerrydetect import mcmc
    from gerrydetect.analysis import compute_metrics_on_ensemble
    from gerrydetect.partition import Partition
    from gerrydetect.spectral import recursive_bisect

    state = args.state.lower()
    proc_dir = REPO_ROOT / "data" / "processed"
    ens_dir = REPO_ROOT / "data" / "ensembles"
    ens_dir.mkdir(parents=True, exist_ok=True)

    graph_path = proc_dir / f"{state}_graph.pkl"
    gdf_path = proc_dir / f"{state}_precincts.parquet"
    if not graph_path.exists():
        log.error("Run scripts/build_graph.py %s first; missing %s", state, graph_path)
        return 1

    log.info("Loading graph from %s", graph_path)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    import geopandas as gpd
    gdf = gpd.read_parquet(gdf_path) if gdf_path.exists() else None

    enacted_assignment = {n: int(graph.nodes[n].get("district", 0)) for n in graph.nodes}
    k = len(set(enacted_assignment.values()))
    log.info("Enacted plan has %d districts", k)

    log.info("Running spectral bisection (k=%d) ...", k)
    seed_assignment = recursive_bisect(graph, k=k, pop_tol=args.pop_tol)

    log.info(
        "Running MCMC: n=%d, lag=%d, burn-in=%d, pop_tol=%g",
        args.n, args.lag, args.burn_in, args.pop_tol,
    )
    samples, stats = mcmc.run(
        graph,
        seed_assignment=seed_assignment,
        n_steps=args.n,
        pop_tol=args.pop_tol,
        lag=args.lag,
        burn_in=args.burn_in,
        seed=args.seed,
    )
    log.info(
        "MCMC done: %d samples, accepted=%d, rejected_pop=%d, rejected_contig=%d, accept_rate=%.3f",
        len(samples), stats.accepted, stats.rejected_pop, stats.rejected_contig,
        stats.acceptance_rate(),
    )

    # Persist assignments as wide parquet (rows = samples, cols = node ids).
    node_order = list(graph.nodes)
    assignments_array = np.array([s.to_array(node_order) for s in samples], dtype=np.int32)
    assign_df = pd.DataFrame(assignments_array, columns=[str(n) for n in node_order])
    assign_path = ens_dir / f"{state}_mcmc_assignments.parquet"
    assign_df.to_parquet(assign_path)
    log.info("Wrote %s", assign_path)

    # Compute and persist per-plan metrics (skip geometric metrics if no gdf).
    log.info("Computing metrics on ensemble ...")
    metrics_df = compute_metrics_on_ensemble(samples, gdf=gdf)
    metrics_path = ens_dir / f"{state}_mcmc_metrics.parquet"
    metrics_df.to_parquet(metrics_path)
    log.info("Wrote %s", metrics_path)

    # Also persist the enacted partition's metrics + the seed (spectral) plan
    # so analyze.py can compare.
    enacted_partition = Partition(graph, enacted_assignment)
    seed_partition = Partition(graph, seed_assignment)
    side_df = compute_metrics_on_ensemble(
        [enacted_partition, seed_partition], gdf=gdf
    )
    side_df.insert(0, "plan", ["enacted", "spectral"])
    side_path = ens_dir / f"{state}_baselines.parquet"
    side_df.to_parquet(side_path)
    log.info("Wrote %s", side_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="two-letter state code (e.g., pa)")
    parser.add_argument("--n", type=int, default=1000, help="saved samples")
    parser.add_argument("--lag", type=int, default=100, help="steps between saves")
    parser.add_argument("--burn-in", type=int, default=10_000, help="burn-in steps")
    parser.add_argument("--pop-tol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(main(args))
