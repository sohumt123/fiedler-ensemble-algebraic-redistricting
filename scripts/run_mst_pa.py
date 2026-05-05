"""Run the full PA analysis with mst_diameter included in the ensemble.

The optimized _kruskal_mst (node-iteration edge collection) makes this feasible
for PA (~9K VTDs, 18 districts, 900-plan ensemble).

Usage:
    python scripts/run_mst_pa.py

Outputs (overwrite PA files only):
    docs/figures/real_pa/pa_real_panel.png   (updated, MST in summary table)
    output/tables/pa_real_long.csv           (5 metrics, including mst_diameter)
    output/tables/pa_mst_summary.csv         (MST-specific outlier result)
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gerrydetect.analysis import (
    bootstrap_p_value,
    composite_severity_score,
    outlier_analysis,
)
from gerrydetect.data_mggg import load_mggg_pa
from gerrydetect.graph import build_graph, graph_summary
from gerrydetect.metrics import (
    all_metrics,
    cut_edge_ratio,
    efficiency_gap,
    mean_median,
    modularity,
    mst_diameter,
    seats_votes_curve,
)
from gerrydetect.multichain import run_multichain
from gerrydetect.partition import Partition
from gerrydetect.spectral import recursive_bisect

REPO_ROOT = Path(__file__).resolve().parent.parent
TAB_DIR = REPO_ROOT / "output" / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("run_mst_pa")

N_CHAINS = 3
N_STEPS_PER_CHAIN = 300
LAG = 30
BURN_IN = 3000
POP_TOL = 0.05

METRIC_FNS = {
    "cut_edge_ratio": cut_edge_ratio,
    "modularity": modularity,
    "efficiency_gap": efficiency_gap,
    "mean_median": mean_median,
    "mst_diameter": mst_diameter,
}

METRIC_LABELS = {
    "cut_edge_ratio": "cut edge ratio",
    "modularity": "modularity",
    "efficiency_gap": "efficiency gap",
    "mean_median": "mean – median",
    "mst_diameter": "MST diameter",
}


def _safe_hist(ax, values: np.ndarray, bins: int = 30) -> None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        ax.text(0.5, 0.5, "(no samples)", ha="center", va="center",
                transform=ax.transAxes)
        return
    if float(values.std()) == 0.0:
        v = float(values[0])
        ax.bar([v], [len(values)], width=max(abs(v) * 0.02, 0.001),
               color="#a8c5e6", edgecolor="white")
        return
    ax.hist(values, bins=bins, color="#a8c5e6", edgecolor="white")


def _plot_precinct_map(gdf, assignment: dict, ax, title: str) -> None:
    n_d = max(assignment.values()) + 1
    cmap = plt.colormaps["tab20"].resampled(n_d)
    xs = gdf.geometry.centroid.x.values
    ys = gdf.geometry.centroid.y.values
    cs = np.array([cmap(assignment.get(i, 0)) for i in range(len(gdf))])
    ax.scatter(xs, ys, c=cs, s=1.5, edgecolor="none", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def render_pa_panel(
    gdf,
    enacted_partition: Partition,
    spectral_partition: Partition,
    ensemble_df: pd.DataFrame,
    enacted_metrics: dict,
    spectral_metrics: dict,
    out_path: Path,
) -> None:
    """10-panel figure: 2 maps + 5 metric histograms + seats-votes + summary."""
    fig = plt.figure(figsize=(18, 10))
    # 2 rows x 5 cols
    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.28)

    # Maps
    ax_map1 = fig.add_subplot(gs[0, 0])
    _plot_precinct_map(gdf, enacted_partition.assignment, ax_map1,
                       "Pennsylvania\nenacted: 2018 Remedial Plan")
    ax_map2 = fig.add_subplot(gs[0, 1])
    _plot_precinct_map(gdf, spectral_partition.assignment, ax_map2,
                       "Pennsylvania\nspectral baseline")

    # 5 metric histograms — positions across both rows
    panel_metrics = ["cut_edge_ratio", "modularity", "efficiency_gap",
                     "mean_median", "mst_diameter"]
    positions = [(0, 2), (0, 3), (0, 4), (1, 0), (1, 1)]
    for (row, col), metric in zip(positions, panel_metrics):
        ax = fig.add_subplot(gs[row, col])
        values = ensemble_df[metric].to_numpy()
        ev = enacted_metrics[metric]
        sv = spectral_metrics[metric]
        _safe_hist(ax, values, bins=30)
        ax.axvline(ev, color="#c0392b", lw=2, label=f"enacted = {ev:.3f}")
        ax.axvline(sv, color="#2980b9", lw=1.5, ls="--", label=f"spectral = {sv:.3f}")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    # Seats-votes curve
    ax_sv = fig.add_subplot(gs[1, 2])
    enacted_curve = seats_votes_curve(enacted_partition, swing_range=0.15, n_points=41)
    spectral_curve = seats_votes_curve(spectral_partition, swing_range=0.15, n_points=41)
    n_seats = enacted_partition.num_districts
    xs = enacted_curve.statewide_d_share
    ideal = np.clip(n_seats * (xs - 0.5) + n_seats / 2, 0, n_seats)
    ax_sv.plot(xs, ideal, "k--", lw=1.2, label="symmetric ideal")
    ax_sv.plot(spectral_curve.statewide_d_share, spectral_curve.expected_d_seats,
               color="#2980b9", lw=2, label="spectral")
    ax_sv.plot(enacted_curve.statewide_d_share, enacted_curve.expected_d_seats,
               color="#c0392b", lw=2, label="enacted")
    ax_sv.set_xlabel("statewide D vote share", fontsize=9)
    ax_sv.set_ylabel("expected D seats", fontsize=9)
    ax_sv.set_title("seats–votes curve", fontsize=10)
    ax_sv.legend(fontsize=7)
    ax_sv.grid(True, alpha=0.3)

    # Summary table (all 5 metrics)
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    lines = ["Metric         Enact  Ens.μ  Pctile"]
    for m in panel_metrics:
        ev = enacted_metrics[m]
        em = float(ensemble_df[m].mean())
        pctile = float((ensemble_df[m].to_numpy() <= ev).mean() * 100)
        lbl = METRIC_LABELS[m][:14]
        lines.append(f"{lbl:14s} {ev:+.3f} {em:+.3f} {pctile:.1f}%")
    ax_text.text(
        0.03, 0.95, "\n".join(lines), transform=ax_text.transAxes,
        fontsize=8, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    # Blank last panel
    fig.add_subplot(gs[1, 4]).axis("off")

    fig.suptitle(
        f"Pennsylvania (real data, with MST diameter): 2018 Remedial Plan vs. "
        f"{len(ensemble_df)}-plan MCMC ensemble (k=18, n={len(gdf)} VTDs)",
        fontsize=11, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load PA data
    t0 = time.time()
    log.info("Loading PA MGGG shapefile ...")
    gdf = load_mggg_pa(REPO_ROOT / "data/raw/pa_mggg", election="2016")
    log.info("Loaded %d VTDs in %.1fs", len(gdf), time.time() - t0)

    # 2. Build adjacency graph
    graph = build_graph(gdf)
    log.info("Graph: %s", graph_summary(graph))

    # 3. Enacted plan
    enacted_assign = {i: int(gdf.loc[i, "district"]) for i in range(len(gdf))}
    unique_dists = sorted(set(enacted_assign.values()))
    remap = {d: i for i, d in enumerate(unique_dists)}
    enacted_assign = {n: remap[d] for n, d in enacted_assign.items()}
    k = len(unique_dists)
    log.info("Enacted plan: k=%d districts", k)

    enacted_partition = Partition(graph, enacted_assign)
    enacted_metrics = all_metrics(enacted_partition)
    log.info("Enacted metrics: %s", {m: round(v, 4) for m, v in enacted_metrics.items()})

    # 4. Spectral baseline
    log.info("Spectral bisection k=%d ...", k)
    t0 = time.time()
    spectral_assign = recursive_bisect(graph, k=k, pop_tol=POP_TOL)
    spectral_partition = Partition(graph, spectral_assign)
    spectral_metrics = all_metrics(spectral_partition)
    log.info("Spectral done in %.1fs", time.time() - t0)

    # 5. MCMC ensemble
    log.info("MCMC: %d chains × %d steps (lag=%d burn=%d) ...",
             N_CHAINS, N_STEPS_PER_CHAIN, LAG, BURN_IN)
    t0 = time.time()
    result = run_multichain(
        graph,
        seed_assignment=spectral_assign,
        n_chains=N_CHAINS,
        n_steps=N_STEPS_PER_CHAIN,
        lag=LAG,
        burn_in=BURN_IN,
        pop_tol=POP_TOL,
        seeds=[42, 55, 68],
        show_progress=True,
    )
    log.info("MCMC done in %.1fs", time.time() - t0)

    pooled = result.pooled_samples()
    log.info("Pooled: %d plans", len(pooled))

    # 6. Compute all 5 metrics on ensemble (MST included)
    log.info("Computing metrics on %d ensemble plans (includes MST — may take ~1-3 min) ...",
             len(pooled))
    t0 = time.time()
    ensemble_rows = []
    for i, p in enumerate(pooled):
        row = {name: fn(p) for name, fn in METRIC_FNS.items()}
        ensemble_rows.append(row)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pooled) - i - 1) / rate
            log.info("  %d/%d plans done (%.1f plans/s, ETA %.0fs)",
                     i + 1, len(pooled), rate, eta)
    ensemble_df = pd.DataFrame(ensemble_rows)
    log.info("Metric computation done in %.1fs", time.time() - t0)

    # 7. Outlier analysis
    enacted_for_compare = {m: enacted_metrics[m] for m in METRIC_FNS}
    out_results = outlier_analysis(enacted_for_compare, ensemble_df)
    rows_out = []
    for r in out_results:
        p_pt, p_lo, p_hi = bootstrap_p_value(
            r.enacted_value,
            ensemble_df[r.metric].to_numpy(),
            n_boot=500,
            seed=42,
        )
        rows_out.append({
            "state": "PA",
            "name": "Pennsylvania",
            "metric": r.metric,
            "enacted": r.enacted_value,
            "spectral": spectral_metrics[r.metric],
            "ensemble_mean": r.ensemble_mean,
            "ensemble_std": r.ensemble_std,
            "percentile": r.percentile,
            "p_value": p_pt,
            "p_lo_95": p_lo,
            "p_hi_95": p_hi,
            "direction": r.direction,
        })

    severity = composite_severity_score(out_results)
    log.info("PA composite severity (5 metrics): %.3f", severity)

    long_df = pd.DataFrame(rows_out)
    long_df.to_csv(TAB_DIR / "pa_real_long.csv", index=False)
    log.info("Saved pa_real_long.csv")

    # Print MST result clearly
    mst_row = long_df[long_df.metric == "mst_diameter"].iloc[0]
    log.info("=== MST DIAMETER RESULT ===")
    log.info("  Enacted:       %.4f", mst_row.enacted)
    log.info("  Spectral:      %.4f", mst_row.spectral)
    log.info("  Ensemble mean: %.4f ± %.4f", mst_row.ensemble_mean, mst_row.ensemble_std)
    log.info("  Percentile:    %.1f%%", mst_row.percentile)
    log.info("  p-value:       %.4f", mst_row.p_value)
    log.info("  Direction:     %s", mst_row.direction)

    # 8. Updated figure (5 metrics)
    fig_path = REPO_ROOT / "output" / "figures" / "real_pa" / "pa_real_panel.png"
    docs_path = REPO_ROOT / "docs" / "figures" / "real_pa" / "pa_real_panel.png"
    render_pa_panel(
        gdf, enacted_partition, spectral_partition,
        ensemble_df, enacted_metrics, spectral_metrics,
        out_path=fig_path,
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, docs_path)
    log.info("Figure committed to %s", docs_path)

    log.info("Done. PA MST analysis complete.")


if __name__ == "__main__":
    main()
