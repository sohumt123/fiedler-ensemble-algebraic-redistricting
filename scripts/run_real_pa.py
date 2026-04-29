"""End-to-end analysis of real Pennsylvania redistricting data.

Uses the MGGG PA VTD shapefile (real precincts, real elections, real enacted
district plans) to run the exact same pipeline that `run_full_analysis.py`
runs on synthetic data.

This script:
  1. Loads real PA VTDs (~9000 precincts) from MGGG data
  2. Builds the precinct adjacency graph
  3. Computes the spectral bisection baseline plan (Fiedler vector)
  4. Runs multi-chain MCMC to generate the ensemble null distribution
  5. Computes compactness + partisan metrics on every plan
  6. Tests the real enacted plan as an outlier vs. the ensemble
  7. Produces figures + tables

Data source: https://github.com/mggg-states/PA-shapefiles (MIT license)
  - 2010 Census VTDs with demographics
  - 2016 Presidential election returns
  - 2018 Remedial Congressional District plan (the "enacted" plan)

Usage:
    python scripts/run_real_pa.py
"""

from __future__ import annotations

import json
import logging
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
from gerrydetect.diagnostics import effective_sample_size, gelman_rubin
from gerrydetect.graph import build_graph, graph_summary
from gerrydetect.metrics import (
    all_metrics,
    cut_edge_ratio,
    efficiency_gap,
    mean_median,
    modularity,
    mst_diameter,
    polsby_popper,
    reock,
    seats_votes_curve,
)
from gerrydetect.multichain import run_multichain
from gerrydetect.partition import Partition
from gerrydetect.spectral import recursive_bisect

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "pa_mggg"
FIG_DIR = REPO_ROOT / "output" / "figures" / "real_pa"
TAB_DIR = REPO_ROOT / "output" / "tables"
DOCS_FIG_DIR = REPO_ROOT / "docs" / "figures" / "real_pa"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("real_pa")

# MCMC parameters — conservative for real data.
N_CHAINS = 3
N_STEPS_PER_CHAIN = 200   # real PA is large, each step is slower
LAG = 30
BURN_IN = 3000
POP_TOL = 0.05   # 5% population tolerance (legal standard)

N_DISTRICTS = 18  # PA 2018 remedial plan has 18 congressional districts


def plot_precinct_map(gdf, assignment, ax, title: str, cmap_name: str = "tab20"):
    """Plot VTD centroids colored by district assignment."""
    cmap = plt.colormaps[cmap_name].resampled(max(assignment.values()) + 1)
    xs = gdf.geometry.centroid.x.values
    ys = gdf.geometry.centroid.y.values
    cs = np.array([cmap(assignment.get(i, 0)) for i in range(len(gdf))])
    ax.scatter(xs, ys, c=cs, s=1.5, edgecolor="none", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_hist(ax, values: np.ndarray, bins: int = 30) -> None:
    """matplotlib's hist throws on a zero-range input."""
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


def render_real_pa_panel(
    gdf,
    graph,
    enacted_partition: Partition,
    spectral_partition: Partition,
    ensemble_metrics: pd.DataFrame,
    enacted_metrics: dict,
    spectral_metrics: dict,
    out_path: Path,
):
    """8-panel figure for real PA analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)

    # Row 1: maps + compactness histograms
    ax_map1 = fig.add_subplot(gs[0, 0])
    plot_precinct_map(gdf, enacted_partition.assignment, ax_map1,
                      "PA — enacted (2018 remedial)")
    ax_map2 = fig.add_subplot(gs[0, 1])
    plot_precinct_map(gdf, spectral_partition.assignment, ax_map2,
                      "PA — spectral baseline")

    metric_panels = [
        ("cut_edge_ratio", "cut edge ratio"),
        ("modularity", "Newman–Girvan modularity"),
        ("efficiency_gap", "efficiency gap"),
        ("mean_median", "mean – median"),
    ]

    for i, (metric_name, label) in enumerate(metric_panels[:2]):
        ax = fig.add_subplot(gs[0, 2 + i])
        values = ensemble_metrics[metric_name].to_numpy()
        enacted_v = enacted_metrics[metric_name]
        spectral_v = spectral_metrics[metric_name]
        _safe_hist(ax, values, bins=30)
        ax.axvline(enacted_v, color="#c0392b", linewidth=2,
                   label=f"enacted = {enacted_v:.4f}")
        ax.axvline(spectral_v, color="#2980b9", linewidth=1.5, linestyle="--",
                   label=f"spectral = {spectral_v:.4f}")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    # Row 2: partisan histograms + seats-votes
    for i, (metric_name, label) in enumerate(metric_panels[2:]):
        ax = fig.add_subplot(gs[1, i])
        values = ensemble_metrics[metric_name].to_numpy()
        enacted_v = enacted_metrics[metric_name]
        spectral_v = spectral_metrics[metric_name]
        _safe_hist(ax, values, bins=30)
        ax.axvline(enacted_v, color="#c0392b", linewidth=2,
                   label=f"enacted = {enacted_v:.4f}")
        ax.axvline(spectral_v, color="#2980b9", linewidth=1.5, linestyle="--",
                   label=f"spectral = {spectral_v:.4f}")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    # Seats-votes curves
    ax_sv = fig.add_subplot(gs[1, 2])
    enacted_curve = seats_votes_curve(enacted_partition, swing_range=0.15, n_points=41)
    spectral_curve = seats_votes_curve(spectral_partition, swing_range=0.15, n_points=41)
    n_seats = enacted_partition.num_districts
    xs = enacted_curve.statewide_d_share
    ideal = np.clip(n_seats * (xs - 0.5) + n_seats / 2, 0, n_seats)
    ax_sv.plot(xs, ideal, color="black", linestyle="--", linewidth=1.2,
               label="symmetric ideal")
    ax_sv.plot(spectral_curve.statewide_d_share, spectral_curve.expected_d_seats,
               color="#2980b9", linewidth=2, label="spectral")
    ax_sv.plot(enacted_curve.statewide_d_share, enacted_curve.expected_d_seats,
               color="#c0392b", linewidth=2, label="enacted")
    ax_sv.set_xlabel("statewide D vote share", fontsize=9)
    ax_sv.set_ylabel("expected D seats", fontsize=9)
    ax_sv.set_title("seats–votes curve", fontsize=10)
    ax_sv.legend(fontsize=7)
    ax_sv.grid(True, alpha=0.3)

    # P-value summary text
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    summary_lines = ["Metric          Enacted  Ens.Mean  Pctile"]
    for m, label in metric_panels:
        ev = enacted_metrics[m]
        em = float(ensemble_metrics[m].mean())
        pctile = float(
            (ensemble_metrics[m].to_numpy() <= ev).mean() * 100
        )
        summary_lines.append(f"{label[:16]:16s} {ev:+.4f}  {em:+.4f}  {pctile:.1f}%")
    ax_text.text(0.05, 0.95, "\n".join(summary_lines), transform=ax_text.transAxes,
                 fontsize=8, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(
        f"Pennsylvania (REAL DATA): enacted 2018 remedial plan vs. "
        f"{len(ensemble_metrics)}-plan MCMC ensemble (k={N_DISTRICTS})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure to %s", out_path)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load real PA data ----
    log.info("=== Loading real PA VTD data (MGGG) ===")
    t0 = time.time()
    gdf = load_mggg_pa(DATA_DIR, election="2016")
    log.info("Loaded %d VTDs in %.1fs", len(gdf), time.time() - t0)

    # ---- 2. Build precinct adjacency graph ----
    log.info("Building adjacency graph ...")
    t0 = time.time()
    graph = build_graph(gdf)
    gs_dict = graph_summary(graph)
    log.info("Graph built in %.1fs: %s", time.time() - t0, gs_dict)

    # ---- 3. Extract the real enacted plan from the data ----
    enacted_assign = {i: int(gdf.loc[i, "district"]) for i in range(len(gdf))}
    # Remap district IDs to 0-based contiguous IDs for consistency.
    unique_dists = sorted(set(enacted_assign.values()))
    dist_remap = {d: i for i, d in enumerate(unique_dists)}
    enacted_assign = {n: dist_remap[d] for n, d in enacted_assign.items()}
    k = len(unique_dists)
    log.info("Enacted plan has %d districts", k)

    enacted_partition = Partition(graph, enacted_assign)
    # Skip geometric metrics (Polsby-Popper, Reock) — they need shapely >= 2.1
    # for minimum_bounding_circle. Graph-theoretic + partisan metrics suffice.
    enacted_metrics = all_metrics(enacted_partition)
    log.info("Enacted metrics: %s", {m: round(v, 4) for m, v in enacted_metrics.items()})

    # ---- 4. Spectral bisection baseline ----
    log.info("Running spectral bisection (k=%d) ...", k)
    t0 = time.time()
    spectral_assign = recursive_bisect(graph, k=k, pop_tol=POP_TOL)
    spectral_partition = Partition(graph, spectral_assign)
    spectral_metrics = all_metrics(spectral_partition)
    log.info("Spectral bisection done in %.1fs", time.time() - t0)
    log.info("Spectral metrics: %s", {m: round(v, 4) for m, v in spectral_metrics.items()})

    # ---- 5. MCMC ensemble ----
    log.info("Running %d-chain MCMC (steps=%d, lag=%d, burn=%d) ...",
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
        seeds=[42 + i * 13 for i in range(N_CHAINS)],
        show_progress=True,
    )
    dt = time.time() - t0
    log.info("MCMC done in %.1fs", dt)

    for i, st in enumerate(result.stats):
        log.info("  chain %d: proposed=%d accepted=%d rate=%.3f",
                 i, st.proposed, st.accepted, st.acceptance_rate())

    pooled = result.pooled_samples()
    log.info("Pooled ensemble: %d plans", len(pooled))

    # ---- 6. Compute metrics on ensemble ----
    metric_fns = {
        "cut_edge_ratio": cut_edge_ratio,
        "mst_diameter": mst_diameter,
        "modularity": modularity,
        "efficiency_gap": efficiency_gap,
        "mean_median": mean_median,
    }
    ensemble_rows = []
    for i, p in enumerate(pooled):
        row = {name: fn(p) for name, fn in metric_fns.items()}
        ensemble_rows.append(row)
        if (i + 1) % 50 == 0:
            log.info("  computed metrics for %d/%d ensemble plans", i + 1, len(pooled))
    ensemble_df = pd.DataFrame(ensemble_rows)

    # ---- 7. Diagnostics ----
    # R-hat / ESS require re-evaluating each metric on every chain sample.
    # For cheap metrics (cut ratio, EG, MM) that's fine; for expensive ones
    # (mst_diameter on 9253 nodes) it would take hours. We compute R-hat
    # only for the cheap metrics and report N/A for the expensive ones.
    cheap_metric_fns = {
        "cut_edge_ratio": cut_edge_ratio,
        "efficiency_gap": efficiency_gap,
        "mean_median": mean_median,
    }
    diagnostics = {}
    for name, fn in cheap_metric_fns.items():
        log.info("  computing R-hat for %s ...", name)
        diagnostics[f"{name}_rhat"] = result.rhat(fn)
        diagnostics[f"{name}_ess"] = result.effective_sample_size(fn)
    # For expensive metrics, report NaN.
    for name in metric_fns:
        if name not in cheap_metric_fns:
            diagnostics[f"{name}_rhat"] = float("nan")
            diagnostics[f"{name}_ess"] = float("nan")
    log.info("R-hat: %s", {k: round(v, 3) for k, v in diagnostics.items()
                           if "_rhat" in k and not np.isnan(v)})

    # ---- 8. Outlier analysis ----
    enacted_for_compare = {mn: enacted_metrics[mn] for mn in metric_fns}
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
            "state": "PA_REAL",
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
            "rhat": diagnostics[f"{r.metric}_rhat"],
            "ess": diagnostics[f"{r.metric}_ess"],
        })

    severity = composite_severity_score(out_results)
    log.info("Composite severity score: %.3f", severity)

    # Write per-metric table.
    state_table = pd.DataFrame(rows_out)
    csv_path = TAB_DIR / "pa_real_summary.csv"
    state_table.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    # ---- 9. Figures ----
    panel_path = FIG_DIR / "pa_real_panel.png"
    render_real_pa_panel(
        gdf, graph,
        enacted_partition, spectral_partition,
        ensemble_df, enacted_metrics, spectral_metrics,
        out_path=panel_path,
    )
    import shutil
    shutil.copy2(panel_path, DOCS_FIG_DIR / "pa_real_panel.png")

    # ---- 10. Run config sidecar ----
    run_meta = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "MGGG PA-shapefiles (https://github.com/mggg-states/PA-shapefiles)",
        "data_license": "MIT",
        "election_year": "2016 Presidential",
        "enacted_plan": "2018 Remedial Congressional Plan",
        "n_vtds": len(gdf),
        "n_districts": k,
        "total_pop": int(gdf["pop"].sum()),
        "n_chains": N_CHAINS,
        "n_steps_per_chain": N_STEPS_PER_CHAIN,
        "lag": LAG,
        "burn_in": BURN_IN,
        "pop_tol": POP_TOL,
        "ensemble_size": len(pooled),
        "severity_score": severity,
    }
    meta_path = FIG_DIR / "run_config_real_pa.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    log.info("Wrote %s", meta_path)

    # Print summary.
    print("\n" + "=" * 60)
    print("  REAL PENNSYLVANIA ANALYSIS — SUMMARY")
    print("=" * 60)
    print(f"  VTDs:          {len(gdf)}")
    print(f"  Districts:     {k}")
    print(f"  Ensemble size: {len(pooled)}")
    print(f"  Severity:      {severity:.3f}")
    print()
    print(state_table.to_string(index=False))
    print()
    print(f"  Figures: {panel_path}")
    print(f"  Tables:  {csv_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
