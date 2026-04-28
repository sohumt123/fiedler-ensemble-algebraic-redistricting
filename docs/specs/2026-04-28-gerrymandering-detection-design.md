# Gerrymandering Detection via Graph Partitioning — Design Spec

**Date:** 2026-04-28
**Project:** NETS 1500 final project (Spring 2026)
**Team:** Sohum Trivedi, Ronit Kapoor, Aditi Ghosh, Maurya Bonu
**Status:** approved

## Context

Gerrymandering — drawing electoral district boundaries to favor a party — is hard
to detect because there is no single definition of a "fair" map. The modern
computational approach treats districting as a **graph partitioning** problem:
model a state as a graph of voting precincts (nodes) connected by geographic
adjacency (edges); a districting plan is a partition of this graph into k
connected, population-balanced subgraphs. By generating a large ensemble of
algorithmically neutral partitions and comparing the enacted plan against that
distribution, we get a principled outlier test.

This project implements that pipeline from scratch in Python for **Pennsylvania
as the pilot state** (most-cited gerrymandering case, ~9,200 precincts, 17
districts). The pipeline is designed to scale to the four other focal states
(NC, WI, OH, MD) once PA is working.

## Goals & non-goals

**Goals (v1):**
- End-to-end PA pipeline: data ingest → graph → metrics → ensemble → outlier analysis → figures
- Implement spectral bisection, single-flip MCMC, modularity, MST diameter,
  efficiency gap, mean-median, seats-votes by hand
- Produce histograms of every metric with the enacted plan marked, plus
  per-metric p-values
- Publish a clean, reusable Python package (`gerrydetect`)
- Scaffold the LaTeX report with PA section drafted

**Non-goals (v1):**
- Automated download of all 5 states (PA only; document procedure for the rest)
- Beating production samplers on mixing speed (single-flip is known slow; we
  acknowledge in the report)
- Web UI / interactive visualization
- ReCom (recombination) sampler — too complex for a 6-week project from scratch

## Decisions made during brainstorming

1. **Build from scratch, not GerryChain.** Spectral bisection, MCMC, modularity,
   MST diameter, contiguity checking — all our code. SciPy (`eigsh`), NetworkX
   (graph container only), geopandas, shapely are infrastructure, not algorithms.
2. **Pilot state: Pennsylvania.** Replicate to other states after PA works.
3. **Library boundary "C" (NETS 1500-appropriate):** NetworkX for storage,
   SciPy for matrix primitives, everything algorithmic by hand.
4. **Single-flip MCMC** per the proposal. Realistic v1 target is **1,000 saved
   plans** (lag=100, 100K post-burn-in steps); architecture supports longer runs.
5. **Build scope: end-to-end.** All four teammates' modules.

## System architecture

```
shapefiles + election + ACS population
        |
        v
[1] data ingest      -> precinct GeoDataFrame (geometry, pop, votes_d, votes_r, district)
        |
        v
[2] graph builder    -> nx.Graph (nodes = precincts, edges = rook-adjacency, weighted)
        |
        v
[3] partition layer  -> Partition(graph, assignment) — central abstraction
        |
        +--> enacted partition (from district shapefile)
        +--> spectral.recursive_bisect(G, k)         (deterministic baseline)
        +--> mcmc.run(G, seed_partition, n_steps)    (single-flip Metropolis chain)
        |
        v
[4] metrics          -> per-Partition: cut ratio, MST diameter, modularity,
                        Polsby-Popper, Reock, efficiency gap, mean-median,
                        seats-votes curve
        |
        v
[5] analysis         -> histograms, p-values (enacted vs. ensemble),
                        composite gerrymandering severity score
        |
        v
[6] visualization    -> matplotlib histograms, geopandas choropleth maps,
                        seats-votes plots
        |
        v
deliverables: output/figures, output/tables, report/report.tex
```

The **`Partition` class** is the central abstraction. It wraps
`(graph, assignment: dict[node -> district_id])` and lazily computes per-district
aggregates (population, D-votes, R-votes, boundary edges, node sets). Every
metric and every sampler operates on `Partition` instances. This isolates each
module: metrics don't care how the partition was generated; samplers don't care
which metrics will run later.

## Data layer (`gerrydetect/data.py`)

Sources:
- **Precincts + 2020 election results:** Redistricting Data Hub VEST PA
  shapefile. ~9,200 precincts with `pres_d`, `pres_r`.
- **Population:** ACS 5-year tract estimates joined to precincts via
  area-weighted spatial overlay (precinct gets sum of overlapping tract pop
  weighted by overlap area).
- **Enacted districts:** PA congressional districts (post-2020, Act 2022-19),
  shapefile from RDH.

Files land in `data/raw/pa/` (gitignored). `scripts/download_data.py` documents
URLs and pulls archives; processed precinct GeoDataFrame written as
`data/processed/pa_precincts.parquet`.

## Graph construction (`gerrydetect/graph.py`)

`build_graph(gdf) -> nx.Graph`:
- For each precinct polygon, find candidate neighbors via `shapely.STRtree`.
- Edge between u, v iff their polygons share a boundary segment of length > ε
  (drops point-touches at corners).
- Edge weight = shared border length.
- Node attrs: `pop`, `votes_d`, `votes_r`, `district` (enacted), `centroid`.
- Geometry stored in the GeoDataFrame, referenced by node ID; not stored in graph.

Preprocessing:
- Compute connected components; keep largest.
- For islands (e.g., precincts on islands in Lake Erie), merge into nearest
  non-island neighbor by centroid distance and add a synthetic edge.
- Log dropped/merged counts for the report.

Persisted as `data/processed/pa_graph.pkl` (NetworkX pickle).

## Partition abstraction (`gerrydetect/partition.py`)

```python
class Partition:
    graph: nx.Graph
    assignment: dict[NodeId, int]      # node -> district id
    gdf: GeoDataFrame | None           # optional; needed for geometric metrics

    # Lazy properties (cached, invalidated on mutation):
    districts: dict[int, set[NodeId]]
    district_pop: dict[int, int]
    district_votes_d: dict[int, int]
    district_votes_r: dict[int, int]
    boundary_edges: set[tuple]         # cross-district edges
    cut_size: int

    def flip(self, node: NodeId, new_district: int) -> "Partition":
        """Return new partition with node reassigned. Pure / immutable."""
```

For MCMC speed we will additionally provide a mutable variant
(`MutablePartition`) with O(degree(v)) incremental updates to boundary edges and
district aggregates on flip. Tests verify the two stay in sync.

## Contiguity (`gerrydetect/contiguity.py`)

`is_district_connected_after_flip(P, node, new_district) -> bool`:
- Source district = `P.assignment[node]`.
- Run BFS over `P.districts[source] - {node}` starting from any other node in
  source.
- Returns whether all of `source - {node}` are reachable.
- O(|district|). Used inside MCMC inner loop, must be fast — implemented
  iteratively with a `set` visited check.

## Metrics (`gerrydetect/metrics.py`)

All take a `Partition`, return scalar (or dict for seats-votes).

- `cut_edge_ratio(P)` — `len(P.boundary_edges) / G.number_of_edges()`.
- `mst_diameter(P)` — for each district subgraph, build MST (Kruskal by hand),
  compute longest path with two BFS. Return mean across districts.
- `modularity(P)` — Newman-Girvan formula computed by hand from per-district
  edge counts and degree sums:
  `Q = sum_c (e_c / m - (d_c / 2m)^2)` where `e_c` = within-district edges,
  `d_c` = sum of degrees in district c.
- `polsby_popper(P)` — needs polygon union per district from `gdf`.
  `4*pi*A / per^2`. Returns mean across districts.
- `reock(P)` — for each district polygon, smallest enclosing circle (use
  `shapely.minimum_bounding_circle`). `area / circle_area`. Mean across districts.
- `efficiency_gap(P)` — wasted votes formula. `(W_R - W_D) / total_votes` where
  losing-party wasted = all votes; winning-party wasted = votes above 50%.
- `mean_median(P)` — `mean(D_share_per_district) - median(D_share_per_district)`.
- `seats_votes_curve(P)` — uniform partisan swing: shift D-share by delta in
  [-0.2, +0.2], count districts where D-share > 0.5; returns array of
  (statewide_share, expected_seats).

## Samplers

### Spectral bisection (`gerrydetect/spectral.py`)

`recursive_bisect(G, k, pop_tol=0.02) -> assignment`:
- Compute graph Laplacian L = D - A as `scipy.sparse.csr_matrix`.
- Get Fiedler vector via `scipy.sparse.linalg.eigsh(L, k=2, which='SM',
  sigma=0)` and take the second smallest eigenvector.
- Sort nodes by Fiedler value, find the cut index that best balances
  population.
- Recurse on each half; stop when k districts produced.
- Repair contiguity: for each district subgraph, if disconnected, reassign
  small fragments to neighboring districts.

### Single-flip MCMC (`gerrydetect/mcmc.py`)

`run(G, seed_partition, n_steps, pop_tol=0.02, lag=100, burn_in=10000,
seed=42) -> list[Partition]`:

```
P = MutablePartition(G, seed_partition.assignment)
samples = []
boundary = list(P.boundary_edges)        # rebuilt periodically

for step in range(burn_in + n_steps * lag):
    (u, v) = random.choice(boundary)
    # propose flipping the smaller-district endpoint to the larger
    src_d, dst_d = P.assignment[u], P.assignment[v]
    if P.district_pop[src_d] < P.district_pop[dst_d]:
        flip_node, target = v, src_d
    else:
        flip_node, target = u, dst_d

    # population check
    new_src_pop = P.district_pop[P.assignment[flip_node]] - P.node_pop[flip_node]
    new_dst_pop = P.district_pop[target] + P.node_pop[flip_node]
    if not within_tolerance(new_src_pop, new_dst_pop, pop_tol):
        continue

    # contiguity check
    if not is_district_connected_after_flip(P, flip_node, target):
        continue

    P.flip(flip_node, target)             # incremental update
    if step >= burn_in and (step - burn_in) % lag == 0:
        samples.append(P.snapshot())      # frozen Partition

return samples
```

**Realism:** PA at 9K nodes, lag=100 → 100K + 10K steps post-/pre-burn-in.
Estimated 30-90 min on a laptop in pure Python. v1 target: 1,000 saved
plans. The `Partition` snapshot copies `assignment` only (~70KB); 1K plans
~70MB on disk as parquet of assignment vectors.

## Analysis (`gerrydetect/analysis.py`)

- `compute_metrics_on_ensemble(samples, metric_fns) -> DataFrame` — one row
  per plan, one column per metric.
- `outlier_p_value(enacted_value, ensemble_values, direction='two-sided')` —
  fraction of ensemble more extreme than enacted.
- `composite_severity_score(p_values_per_state) -> float` — mean of
  `-log10(p)` across metrics, signed by gerrymander direction.

## Visualization (`gerrydetect/viz.py`)

- `plot_histogram(values, enacted, title, xlabel, savepath)` — matplotlib hist
  + vertical red line at enacted value; annotate p-value.
- `plot_district_map(gdf, assignment, title, savepath)` — geopandas choropleth
  colored by district id.
- `plot_seats_votes(curves, enacted_curve, title, savepath)` — overlay
  ensemble curves (light) and enacted curve (bold).

## Repository layout

```
nets-project/
|-- README.md
|-- pyproject.toml              # uv-managed, pinned deps
|-- .gitignore                  # excludes data/raw, data/ensembles, output/
|-- data/
|   |-- raw/        (gitignored)
|   |-- processed/  (gitignored)
|   `-- ensembles/  (gitignored)
|-- output/
|   |-- figures/
|   `-- tables/
|-- gerrydetect/                # the library
|   |-- __init__.py
|   |-- data.py
|   |-- graph.py
|   |-- partition.py
|   |-- contiguity.py
|   |-- metrics.py
|   |-- spectral.py
|   |-- mcmc.py
|   |-- analysis.py
|   `-- viz.py
|-- scripts/                    # CLI entry points
|   |-- download_data.py
|   |-- build_graph.py
|   |-- run_ensemble.py
|   `-- analyze.py
|-- notebooks/
|   `-- 01_pa_walkthrough.ipynb
|-- report/
|   |-- report.tex              # LaTeX scaffold
|   `-- figures/                # symlink/copy from output/figures
`-- tests/
    |-- test_graph.py
    |-- test_partition.py
    |-- test_contiguity.py
    |-- test_metrics.py
    |-- test_spectral.py
    `-- test_mcmc.py
```

## Implementation order

1. Repo scaffolding: `pyproject.toml`, `.gitignore`, `README.md`.
2. Core abstractions: `Partition`, `MutablePartition`, contiguity check + tests.
3. Metrics on synthetic graphs + tests (4-cycle, 6-grid hand-computed values).
4. Graph builder (works on any GeoDataFrame; tested with synthetic polygons).
5. Spectral bisection + tests (50-node random planar graph).
6. MCMC sampler + tests (small graph, verify invariants hold every step).
7. Data ingestion script (PA-specific, with download instructions).
8. Analysis + viz modules.
9. Notebook walkthrough + LaTeX report scaffold.
10. End-to-end smoke test on a tiny synthetic state (10 precincts, 3 districts).

## Verification (how to test end-to-end)

**Unit tests** (`pytest tests/`):
- Metrics on hand-computed graphs.
- Partition invariants: `assert sum(district_pop.values()) == total_pop` after
  every flip.
- Spectral bisection: produces 2 connected, population-balanced halves on a
  50-node random planar graph.
- MCMC: 1K steps on a 20-node graph preserve contiguity and population
  balance every step (asserted in property test).

**Smoke test** (`scripts/smoke_test.py`): synthetic 10-precinct, 3-district
state. Runs full pipeline: build graph, spectral bisect, MCMC for 200 saved
plans, compute all metrics, generate one histogram. Should run end-to-end in
< 30 sec.

**Real-data verification** (manual, requires data download):
```
python scripts/download_data.py pa
python scripts/build_graph.py pa
python scripts/run_ensemble.py pa --n 1000
python scripts/analyze.py pa
```
Expected outputs: figures in `output/figures/pa_*.png`, summary in
`output/tables/pa_summary.csv`.

## Open risks

1. **MCMC mixing.** Single-flip is known slow. We will run multiple chains
   from different seeds, compute per-chain metric distributions, and report
   inter-chain variance. If mixing is poor, the report honestly states that
   p-values are conditional on the chain region explored.
2. **Data licensing.** All sources are public-domain or CC-BY; we will
   include attributions in the report.
3. **Polsby-Popper / Reock for non-simply-connected districts.** District
   polygon unions can have holes; we use the union's exterior perimeter.
