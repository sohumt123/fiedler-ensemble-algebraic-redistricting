# FiedlerEnsemble: Algebraic Redistricting & MCMC Outlier Detection

NETS 1500 final project, Spring 2026.
Sohum Trivedi · Ronit Kapoor · Aditi Ghosh · Maurya Bonu

We model U.S. states as precinct adjacency graphs and apply two complementary
graph-partitioning algorithms — recursive **spectral bisection via the Fiedler
vector** (the eigenvector of the second-smallest eigenvalue of the graph
Laplacian) and a **single-flip Metropolis–Hastings MCMC ensemble** — to test
whether enacted congressional district maps are statistical outliers relative
to a large family of algorithmically neutral plans. Pilot state:
**Pennsylvania**. Future states: NC, WI, OH, MD.

The full design lives in
[`docs/specs/2026-04-28-gerrymandering-detection-design.md`](docs/specs/2026-04-28-gerrymandering-detection-design.md).

## Pipeline

![pipeline](docs/figures/pipeline.png)

Raw shapefiles + election returns become a precinct adjacency graph; the
graph is partitioned (deterministically by spectral bisection, stochastically
by single-flip MCMC); each partition is scored by compactness and
partisan-fairness metrics; the enacted plan's metric values are compared
against the ensemble distribution to produce p-values.

## What's implemented

`gerrydetect` is a from-scratch Python package — no GerryChain dependency.
The graph algorithms (spectral bisection via the Fiedler vector, Newman–Girvan
modularity, MST-based compactness, BFS contiguity, single-flip Metropolis MCMC,
efficiency gap, mean–median, seats–votes curves) are all written by hand. We
use NetworkX as a graph container, SciPy for sparse eigensolvers, and
geopandas/shapely for shapefile I/O.

| Module | Responsibility |
| --- | --- |
| `gerrydetect.partition` | `Partition` / `MutablePartition` — the central abstraction |
| `gerrydetect.contiguity` | BFS-based district connectedness check |
| `gerrydetect.graph` | Build the precinct adjacency graph from a GeoDataFrame |
| `gerrydetect.data` | Load PA precincts + enacted districts |
| `gerrydetect.metrics` | Cut ratio, MST diameter, modularity, Polsby–Popper, Reock, efficiency gap, mean-median, seats-votes |
| `gerrydetect.spectral` | Recursive spectral bisection (Fiedler vector) |
| `gerrydetect.mcmc` | Single-flip Metropolis ensemble sampler |
| `gerrydetect.analysis` | Per-metric outlier p-values, composite severity score |
| `gerrydetect.viz` | Histograms, district choropleth maps, seats–votes plots |

## Example results (synthetic 14×14 state)

The figures below were produced by `scripts/generate_readme_figures.py` —
no real-data download required. They illustrate how the pipeline tells a
"baseline plan vs. ensemble" story; the same plots will be regenerated on
real Pennsylvania data once the team runs the full ensemble.

### District maps: spectral baseline vs. MCMC ensemble draws

![district maps](docs/figures/district_maps.png)

The spectral bisection produces clean, near-rectangular districts (left).
After ~8K accepted MCMC moves, the ensemble draws have organically jagged
boundaries that respect contiguity and population balance but explore a much
wider region of the plan space.

### Outlier histograms

![histograms panel](docs/figures/histograms_panel.png)

Three metrics on the same ensemble: the spectral baseline (red line) is more
compact than every MCMC sample by both **cut edge ratio** (lower is better)
and **modularity** (higher is better), giving p ≈ 0 in both tails. The
**efficiency gap** sits squarely inside the ensemble (p = 0.89), as expected:
spectral bisection optimizes geometric compactness, not partisan fairness.

### Seats–votes curves

![seats votes](docs/figures/seats_votes.png)

The spectral baseline curve (red) and 25 ensemble curves (grey) overlaid on
the symmetric ideal (dashed). Spectral compactness produces noticeable
"step plateaus" — districts flip in clusters as the statewide swing crosses
key thresholds — which is the geometric signature any human map-drawer would
need to overcome to achieve a smooth seats-votes response.

## Setup

```bash
# Python 3.11+ required.
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick smoke test (no data download required)

```bash
python scripts/smoke_test.py
```

Runs the full pipeline (build graph → spectral bisect → MCMC → metrics →
histogram) on a synthetic 10×10 grid "state" with 4 districts. Should finish
in well under a minute and write a figure to `output/figures/smoke_*.png`.

To regenerate the figures embedded in this README:

```bash
python scripts/generate_readme_figures.py
```

## Running on real Pennsylvania data

1. Download PA precinct shapefile + 2020 election results, ACS population, and
   the enacted congressional district shapefile. URLs and instructions live in
   `scripts/download_data.py` (sources: Redistricting Data Hub VEST 2020 PA,
   ACS 5-year via `census.gov`, RDH enacted PA congressional plan).

   ```bash
   python scripts/download_data.py pa
   ```

   Files land in `data/raw/pa/`. (This step is gitignored — each user fetches
   their own copy.)

2. Build the precinct graph:

   ```bash
   python scripts/build_graph.py pa
   ```

   Produces `data/processed/pa_graph.pkl` and `data/processed/pa_precincts.parquet`.

3. Run the ensemble (spectral seed + MCMC):

   ```bash
   python scripts/run_ensemble.py pa --n 1000 --lag 100 --burn-in 10000
   ```

   Writes `data/ensembles/pa_mcmc.parquet` (assignments) and
   `data/ensembles/pa_mcmc_metrics.parquet` (per-plan metrics).

4. Outlier analysis and figures:

   ```bash
   python scripts/analyze.py pa
   ```

   Writes histograms and maps to `output/figures/`, summary table to
   `output/tables/pa_summary.csv`.

## Tests

```bash
pytest
```

Unit tests cover hand-computed metrics on small graphs, partition invariants,
spectral bisection on synthetic planar graphs, and MCMC step-by-step
contiguity/population invariants.

## Repo layout

```
gerrydetect/        # importable Python package
scripts/            # CLI entry points
tests/              # pytest unit tests
notebooks/          # exploratory + walkthrough
report/             # LaTeX final report scaffold
data/               # gitignored — raw, processed, ensembles
output/             # gitignored — figures, tables
docs/specs/         # design specs
docs/figures/       # README figures (committed)
```

## Team responsibilities

- **Sohum** — pipeline architecture, MCMC, spectral bisection, outlier analysis
- **Ronit** — data acquisition, graph construction, preprocessing
- **Aditi** — metrics, statistical analysis, seats–votes curves
- **Maurya** — visualization, report writing, presentation

## License

All datasets used are public-domain or permissively licensed (Census Bureau
TIGER/Line, Redistricting Data Hub VEST, MIT Election Data + Science Lab,
American Community Survey). Attributions in the final report.
