"""Plotting helpers: histogram, district map, seats-votes curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    ensemble_values: np.ndarray,
    enacted_value: float,
    title: str,
    xlabel: str,
    savepath: Path | str | None = None,
    p_value: float | None = None,
    bins: int = 40,
):
    """Histogram of an ensemble metric with the enacted plan marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ensemble_values, bins=bins, color="#a8c5e6", edgecolor="white")
    ax.axvline(enacted_value, color="#c0392b", linewidth=2.5, label=f"enacted = {enacted_value:.4f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ensemble plans")
    if p_value is not None:
        ax.text(
            0.02,
            0.97,
            f"p = {p_value:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"),
        )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150)
    return fig


def plot_district_map(
    gdf,
    assignment: dict,
    title: str,
    savepath: Path | str | None = None,
):
    """Choropleth: color each precinct by district id."""
    plot_gdf = gdf.copy()
    plot_gdf["district"] = plot_gdf.index.map(assignment)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_gdf.plot(
        column="district",
        ax=ax,
        cmap="tab20",
        categorical=True,
        linewidth=0.05,
        edgecolor="white",
        legend=True,
        legend_kwds={"loc": "upper right", "fontsize": 8, "title": "district"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150)
    return fig


def plot_seats_votes(
    enacted_curve,
    ensemble_curves: list,
    title: str,
    savepath: Path | str | None = None,
):
    """Seats-votes plot: ensemble in light grey, enacted in red."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for c in ensemble_curves:
        ax.plot(
            c.statewide_d_share,
            c.expected_d_seats,
            color="lightgray",
            linewidth=0.5,
            alpha=0.4,
        )
    ax.plot(
        enacted_curve.statewide_d_share,
        enacted_curve.expected_d_seats,
        color="#c0392b",
        linewidth=2.5,
        label="enacted",
    )
    ax.set_xlabel("statewide D vote share")
    ax.set_ylabel("expected D seats")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150)
    return fig
