"""End-to-end analysis on real MGGG shapefiles for PA, NC, MD, WI.

Download data first (one-time, ~100 MB total):
    python scripts/download_mggg_states.py all

Then run:
    python scripts/run_real_all_states.py

Per-state outputs (figures committed to docs/figures/real_<state>/):
  docs/figures/real_<state>/<state>_real_panel.png

Table outputs (gitignored, in output/tables/):
  output/tables/<state>_real_summary.csv
  output/tables/all_states_real_summary.csv
  output/tables/all_states_real_long.csv
"""

from __future__ import annotations

import json
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
from gerrydetect.data_mggg import load_mggg_md, load_mggg_nc, load_mggg_pa, load_mggg_wi
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
log = logging.getLogger("run_real_all")

# ---------------------------------------------------------------------------
# Per-state configuration
# ---------------------------------------------------------------------------

STATE_CONFIG: dict[str, dict] = {
    "pa": {
        "name": "Pennsylvania",
        "loader": lambda: load_mggg_pa(REPO_ROOT / "data/raw/pa_mggg", election="2016"),
        "enacted_plan": "2018 Remedial Plan (court-ordered)",
        "election": "2016 Presidential",
    },
    "nc": {
        "name": "North Carolina",
        "loader": lambda: load_mggg_nc(REPO_ROOT / "data/raw/nc_mggg"),
        "enacted_plan": "2016 Enacted Plan",
        "election": "2016 Presidential",
    },
    "md": {
        "name": "Maryland",
        "loader": lambda: load_mggg_md(REPO_ROOT / "data/raw/md_mggg"),
        "enacted_plan": "2011 Enacted Plan",
        "election": "2016 Presidential",
    },
    "wi": {
        "name": "Wisconsin",
        "loader": lambda: load_mggg_wi(REPO_ROOT / "data/raw/wi_mggg"),
        "enacted_plan": "2011 Enacted Plan",
        "election": "2016 Presidential",
    },
}

# MCMC parameters (same for all states for comparability)
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
}
# mst_diameter omitted from ensemble computation — Kruskal on 9K-node graphs
# across 900 plans is too slow in pure Python. Still reported on the enacted
# and spectral plans individually (fast) but not on the full ensemble.

METRIC_LABELS = {
    "cut_edge_ratio": "cut edge ratio",
    "modularity": "modularity",
    "efficiency_gap": "efficiency gap",
    "mean_median": "mean – median",
}

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


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


def render_state_panel(
    state_code: str,
    cfg: dict,
    gdf,
    enacted_partition: Partition,
    spectral_partition: Partition,
    ensemble_df: pd.DataFrame,
    enacted_metrics: dict,
    spectral_metrics: dict,
    out_path: Path,
) -> None:
    """8-panel figure: 2 maps + 4 metric histograms + seats-votes + summary."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)
    name = cfg["name"]

    # Maps
    ax_map1 = fig.add_subplot(gs[0, 0])
    _plot_precinct_map(gdf, enacted_partition.assignment, ax_map1,
                       f"{name}\nenacted: {cfg['enacted_plan']}")
    ax_map2 = fig.add_subplot(gs[0, 1])
    _plot_precinct_map(gdf, spectral_partition.assignment, ax_map2,
                       f"{name}\nspectral baseline")

    # Histograms — 4 metrics in positions (0,2), (0,3), (1,0), (1,1)
    panel_metrics = ["cut_edge_ratio", "modularity", "efficiency_gap", "mean_median"]
    positions = [(0, 2), (0, 3), (1, 0), (1, 1)]
    for (row, col), metric in zip(positions, panel_metrics):
        ax = fig.add_subplot(gs[row, col])
        values = ensemble_df[metric].to_numpy()
        ev = enacted_metrics[metric]
        sv = spectral_metrics[metric]
        _safe_hist(ax, values, bins=30)
        ax.axvline(ev, color="#c0392b", lw=2, label=f"enacted = {ev:.4f}")
        ax.axvline(sv, color="#2980b9", lw=1.5, ls="--", label=f"spectral = {sv:.4f}")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
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

    # Summary table panel
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    lines = ["Metric           Enacted  Ens.Mean Pctile"]
    for m in panel_metrics:
        ev = enacted_metrics[m]
        em = float(ensemble_df[m].mean())
        pctile = float((ensemble_df[m].to_numpy() <= ev).mean() * 100)
        lbl = METRIC_LABELS.get(m, m)[:16]
        lines.append(f"{lbl:16s} {ev:+.4f}  {em:+.4f} {pctile:.1f}%")
    ax_text.text(
        0.05, 0.95, "\n".join(lines), transform=ax_text.transAxes,
        fontsize=8, fontfamily="monospace", va="top",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )

    fig.suptitle(
        f"{name} (real data): {cfg['enacted_plan']} vs. "
        f"{len(ensemble_df)}-plan MCMC ensemble "
        f"(k={enacted_partition.num_districts}, n={len(gdf)} VTDs)",
        fontsize=12, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Per-state analysis
# ---------------------------------------------------------------------------


def analyze_state(state_code: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    log.info("=== %s (%s) ===", state_code.upper(), cfg["name"])

    # 1. Load data
    t0 = time.time()
    gdf = cfg["loader"]()
    log.info("Loaded %d VTDs in %.1fs", len(gdf), time.time() - t0)

    # 2. Build adjacency graph
    graph = build_graph(gdf)
    log.info("Graph: %s", graph_summary(graph))

    # 3. Extract the enacted assignment from the data
    enacted_assign = {i: int(gdf.loc[i, "district"]) for i in range(len(gdf))}
    unique_dists = sorted(set(enacted_assign.values()))
    remap = {d: i for i, d in enumerate(unique_dists)}
    enacted_assign = {n: remap[d] for n, d in enacted_assign.items()}
    k = len(unique_dists)
    log.info("Enacted plan: k=%d districts", k)

    enacted_partition = Partition(graph, enacted_assign)
    enacted_metrics = all_metrics(enacted_partition)
    log.info("Enacted metrics: %s", {m: round(v, 4) for m, v in enacted_metrics.items()})

    # 4. Spectral bisection baseline
    log.info("Spectral bisection k=%d ...", k)
    t0 = time.time()
    spectral_assign = recursive_bisect(graph, k=k, pop_tol=POP_TOL)
    spectral_partition = Partition(graph, spectral_assign)
    spectral_metrics = all_metrics(spectral_partition)
    log.info("Spectral done in %.1fs. Metrics: %s",
             time.time() - t0, {m: round(v, 4) for m, v in spectral_metrics.items()})

    # 5. Multi-chain MCMC
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
        seeds=[42 + i * 13 for i in range(N_CHAINS)],
        show_progress=True,
    )
    log.info("MCMC done in %.1fs", time.time() - t0)
    for i, st in enumerate(result.stats):
        log.info("  chain %d: accepted=%d / proposed=%d (%.1f%%)",
                 i, st.accepted, st.proposed, st.acceptance_rate() * 100)

    pooled = result.pooled_samples()
    log.info("Pooled: %d plans", len(pooled))

    # 6. Compute metrics on ensemble
    ensemble_rows = []
    for p in pooled:
        ensemble_rows.append({name: fn(p) for name, fn in METRIC_FNS.items()})
    ensemble_df = pd.DataFrame(ensemble_rows)

    # 7. Convergence diagnostics (skip mst_diameter — too slow on large graphs)
    cheap = {
        "cut_edge_ratio": cut_edge_ratio,
        "efficiency_gap": efficiency_gap,
        "mean_median": mean_median,
    }
    diagnostics: dict[str, float] = {}
    for name, fn in cheap.items():
        diagnostics[f"{name}_rhat"] = result.rhat(fn)
        diagnostics[f"{name}_ess"] = result.effective_sample_size(fn)
    for name in METRIC_FNS:
        if name not in cheap:
            diagnostics[f"{name}_rhat"] = float("nan")
            diagnostics[f"{name}_ess"] = float("nan")
    log.info("R-hat: %s", {
        k: round(v, 3) for k, v in diagnostics.items()
        if k.endswith("_rhat") and not np.isnan(v)
    })

    # 8. Outlier analysis with bootstrap CIs
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
            "state": state_code.upper(),
            "name": cfg["name"],
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
            "rhat": diagnostics.get(f"{r.metric}_rhat", float("nan")),
            "ess": diagnostics.get(f"{r.metric}_ess", float("nan")),
        })

    severity = composite_severity_score(out_results)
    state_table = pd.DataFrame(rows_out)
    state_table.to_csv(TAB_DIR / f"{state_code}_real_summary.csv", index=False)
    log.info("%s composite severity: %.3f", state_code.upper(), severity)

    # 9. Figures
    fig_dir = REPO_ROOT / "output" / "figures" / f"real_{state_code}"
    docs_dir = REPO_ROOT / "docs" / "figures" / f"real_{state_code}"
    panel_path = fig_dir / f"{state_code}_real_panel.png"
    render_state_panel(
        state_code, cfg, gdf,
        enacted_partition, spectral_partition,
        ensemble_df, enacted_metrics, spectral_metrics,
        out_path=panel_path,
    )
    docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(panel_path, docs_dir / f"{state_code}_real_panel.png")
    log.info("Figure committed to %s", docs_dir)

    valid_rhats = [
        v for k, v in diagnostics.items()
        if k.endswith("_rhat") and not np.isnan(v)
    ]
    return {
        "state": state_code.upper(),
        "name": cfg["name"],
        "n_vtds": len(gdf),
        "n_districts": k,
        "n_ensemble": len(pooled),
        "severity_score": severity,
        "enacted_eg": enacted_metrics["efficiency_gap"],
        "enacted_mm": enacted_metrics["mean_median"],
        "enacted_cut_ratio": enacted_metrics["cut_edge_ratio"],
        "enacted_modularity": enacted_metrics["modularity"],
        "spectral_eg": spectral_metrics["efficiency_gap"],
        "spectral_cut_ratio": spectral_metrics["cut_edge_ratio"],
        "max_rhat": max(valid_rhats) if valid_rhats else float("nan"),
    }, state_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_tables = []

    for state_code, cfg in STATE_CONFIG.items():
        data_dir = REPO_ROOT / "data" / "raw" / f"{state_code}_mggg"
        has_shp = data_dir.exists() and any(data_dir.glob("*.shp"))
        if not has_shp:
            log.warning(
                "Skipping %s — no shapefile in %s. "
                "Run: python scripts/download_mggg_states.py %s",
                state_code.upper(), data_dir, state_code,
            )
            continue
        try:
            row, table = analyze_state(state_code, cfg)
            summary_rows.append(row)
            all_tables.append(table)
        except Exception:
            log.error("Failed on %s", state_code.upper(), exc_info=True)

    if not summary_rows:
        log.error(
            "No states completed. Download data first:\n"
            "  python scripts/download_mggg_states.py all"
        )
        return 1

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("severity_score", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(TAB_DIR / "all_states_real_summary.csv", index=False)

    pd.concat(all_tables, ignore_index=True).to_csv(
        TAB_DIR / "all_states_real_long.csv", index=False
    )

    meta = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "states_completed": [r["state"] for r in summary_rows],
        "n_chains": N_CHAINS,
        "n_steps_per_chain": N_STEPS_PER_CHAIN,
        "lag": LAG,
        "burn_in": BURN_IN,
        "pop_tol": POP_TOL,
    }
    with open(REPO_ROOT / "output" / "run_config_real.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 65)
    print("  REAL-DATA CROSS-STATE SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
