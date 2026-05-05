"""Outlier analysis: percentile, p-value, and composite severity score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gerrydetect.metrics import all_metrics
from gerrydetect.partition import Partition

# Metrics whose direction-of-concern is "lower = more compact, fair."
# Higher cut ratio / MST diameter = less compact.
LOWER_IS_BETTER = {"cut_edge_ratio", "mst_diameter"}
HIGHER_IS_BETTER = {"modularity", "polsby_popper", "reock"}

# Partisan metrics: sign carries direction (positive EG = R-favoring).
SIGNED_METRICS = {"efficiency_gap", "mean_median"}


@dataclass
class OutlierResult:
    metric: str
    enacted_value: float
    ensemble_mean: float
    ensemble_std: float
    percentile: float       # in [0, 100]; where the enacted falls in the ensemble
    p_value_two_sided: float
    direction: str          # "more_extreme" / "less_extreme" / "neutral"


def compute_metrics_on_ensemble(
    samples: list[Partition], gdf=None
) -> pd.DataFrame:
    """Compute every metric on every sample. Returns a DataFrame, one row
    per sample, columns = metric names.
    """
    rows = [all_metrics(p, gdf=gdf) for p in samples]
    return pd.DataFrame(rows)


def percentile_of(value: float, distribution: np.ndarray) -> float:
    """Where in [0, 100] does `value` sit in `distribution`?"""
    if len(distribution) == 0:
        return float("nan")
    return float((np.sum(distribution < value) + 0.5 * np.sum(distribution == value))
                 / len(distribution) * 100)


def two_sided_p_value(value: float, distribution: np.ndarray) -> float:
    """Fraction of `distribution` at least as extreme (in either tail) as
    `value`, measured by distance from the ensemble median.
    """
    if len(distribution) == 0:
        return float("nan")
    median = float(np.median(distribution))
    deviation = abs(value - median)
    return float(np.mean(np.abs(distribution - median) >= deviation))


def bootstrap_p_value(
    value: float,
    distribution: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI around the two-sided p-value.

    We resample the ensemble with replacement `n_boot` times, recompute the
    p-value on each resample, and report (point_estimate, lower, upper) where
    the CI is the central `ci`-quantile interval of the bootstrap distribution.

    The point estimate is `two_sided_p_value` on the original distribution.

    For a constant ensemble (zero variance), the p-value is 1 if the enacted
    value matches the constant, 0 otherwise; the CI collapses to that value.
    """
    distribution = np.asarray(distribution, dtype=float)
    n = len(distribution)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = two_sided_p_value(value, distribution)
    if n < 2 or float(np.std(distribution)) == 0.0:
        return point, point, point

    rng = np.random.default_rng(seed)
    boot_ps = np.empty(n_boot)
    for b in range(n_boot):
        sample = distribution[rng.integers(0, n, size=n)]
        boot_ps[b] = two_sided_p_value(value, sample)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_ps, alpha))
    hi = float(np.quantile(boot_ps, 1.0 - alpha))
    return point, lo, hi


def outlier_analysis(
    enacted_metrics: dict[str, float],
    ensemble_df: pd.DataFrame,
) -> list[OutlierResult]:
    """For every metric column in `ensemble_df`, compute outlier stats."""
    out: list[OutlierResult] = []
    for metric in ensemble_df.columns:
        if metric not in enacted_metrics:
            continue
        ev = enacted_metrics[metric]
        dist = ensemble_df[metric].to_numpy()
        pct = percentile_of(ev, dist)
        p = two_sided_p_value(ev, dist)
        direction = _interpret_direction(metric, ev, dist)
        out.append(
            OutlierResult(
                metric=metric,
                enacted_value=float(ev),
                ensemble_mean=float(np.mean(dist)),
                ensemble_std=float(np.std(dist)),
                percentile=pct,
                p_value_two_sided=p,
                direction=direction,
            )
        )
    return out


def _interpret_direction(metric: str, enacted: float, dist: np.ndarray) -> str:
    """Human-readable label for which tail the enacted plan falls in."""
    if len(dist) == 0:
        return "neutral"
    pct = percentile_of(enacted, dist)
    if metric in LOWER_IS_BETTER:
        return "less_compact" if pct > 95 else ("more_compact" if pct < 5 else "neutral")
    if metric in HIGHER_IS_BETTER:
        return "less_compact" if pct < 5 else ("more_compact" if pct > 95 else "neutral")
    if metric in SIGNED_METRICS:
        if pct > 95:
            return "republican_favoring"
        if pct < 5:
            return "democratic_favoring"
        return "neutral"
    return "neutral"


def composite_severity_score(results: list[OutlierResult]) -> float:
    """Average of -log10(p) across metrics, capped to avoid infinities.

    Higher score = more extreme overall outlier. This is a heuristic; the
    report should also present per-metric p-values directly.
    """
    if not results:
        return 0.0
    scores = []
    for r in results:
        p = max(r.p_value_two_sided, 1e-4)  # cap
        scores.append(-np.log10(p))
    return float(np.mean(scores))


def summary_table(
    results: list[OutlierResult], state: str
) -> pd.DataFrame:
    """Tidy DataFrame for export to CSV / inclusion in the report."""
    return pd.DataFrame(
        [
            {
                "state": state,
                "metric": r.metric,
                "enacted": r.enacted_value,
                "ensemble_mean": r.ensemble_mean,
                "ensemble_std": r.ensemble_std,
                "percentile": r.percentile,
                "p_two_sided": r.p_value_two_sided,
                "direction": r.direction,
            }
            for r in results
        ]
    )
