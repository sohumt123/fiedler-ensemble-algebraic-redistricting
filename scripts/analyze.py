"""Outlier analysis + figure generation for a state.

Reads the ensemble metrics and baseline metrics produced by run_ensemble.py
and writes per-metric histograms, the choropleth of the enacted plan, and
a CSV summary table.

Usage:
    python scripts/analyze.py pa
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("analyze")


def main(state: str) -> int:
    from gerrydetect.analysis import (
        composite_severity_score,
        outlier_analysis,
        summary_table,
    )
    from gerrydetect.viz import plot_district_map, plot_histogram

    state = state.lower()
    proc = REPO_ROOT / "data" / "processed"
    ens = REPO_ROOT / "data" / "ensembles"
    fig_dir = REPO_ROOT / "output" / "figures"
    tab_dir = REPO_ROOT / "output" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_parquet(ens / f"{state}_mcmc_metrics.parquet")
    baselines = pd.read_parquet(ens / f"{state}_baselines.parquet").set_index("plan")
    enacted_metrics = baselines.loc["enacted"].to_dict()

    results = outlier_analysis(enacted_metrics, metrics_df)
    severity = composite_severity_score(results)
    log.info("Composite severity score for %s: %.3f", state.upper(), severity)

    # Histograms
    pretty_xlabel = {
        "cut_edge_ratio": "fraction of edges crossing district boundaries",
        "mst_diameter": "mean MST diameter (edges)",
        "modularity": "Newman–Girvan modularity",
        "polsby_popper": "mean Polsby-Popper score",
        "reock": "mean Reock score",
        "efficiency_gap": "efficiency gap",
        "mean_median": "mean − median D-share",
    }
    for r in results:
        plot_histogram(
            metrics_df[r.metric].to_numpy(),
            r.enacted_value,
            title=f"{state.upper()}: {r.metric}",
            xlabel=pretty_xlabel.get(r.metric, r.metric),
            savepath=fig_dir / f"{state}_hist_{r.metric}.png",
            p_value=r.p_value_two_sided,
        )

    # Choropleth of enacted plan
    gdf_path = proc / f"{state}_precincts.parquet"
    graph_path = proc / f"{state}_graph.pkl"
    if gdf_path.exists() and graph_path.exists():
        import geopandas as gpd
        gdf = gpd.read_parquet(gdf_path)
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        enacted_assignment = {n: int(graph.nodes[n].get("district", 0)) for n in graph.nodes}
        plot_district_map(
            gdf,
            enacted_assignment,
            title=f"{state.upper()} enacted congressional plan",
            savepath=fig_dir / f"{state}_enacted_map.png",
        )

    # Summary table
    summary = summary_table(results, state)
    summary.loc[len(summary)] = {
        "state": state,
        "metric": "_composite_severity",
        "enacted": severity,
        "ensemble_mean": float("nan"),
        "ensemble_std": float("nan"),
        "percentile": float("nan"),
        "p_two_sided": float("nan"),
        "direction": "summary",
    }
    out_csv = tab_dir / f"{state}_summary.csv"
    summary.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="two-letter state code (e.g., pa)")
    args = parser.parse_args()
    sys.exit(main(args.state))
