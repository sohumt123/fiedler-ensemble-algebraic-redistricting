# FiedlerEnsemble: Using Math to Detect Gerrymandering

Sohum Trivedi, Ronit Kapoor, Aditi Ghosh, Maurya Bonu
NETS 1500, Spring 2026

---

## Project Description

FiedlerEnsemble is a computational tool for detecting gerrymandering in U.S. congressional district maps using graph algorithms and statistical inference. We model each state as a precinct adjacency graph, where voting precincts are nodes and shared borders are edges, then generate an ensemble of thousands of neutral random maps using a Metropolis-Hastings Markov chain Monte Carlo sampler. Each map is scored on five metrics covering compactness and partisan fairness, producing a null distribution against which the real enacted map is tested as a statistical outlier. We apply this pipeline to four states with real precinct shapefiles and 2016 Presidential election returns from the Metric Geometry and Gerrymandering Group: Pennsylvania, North Carolina, Maryland, and Wisconsin. Our results quantify, with p-values and bootstrap confidence intervals, which maps are statistically anomalous and in which direction.

## Concepts Covered

**Choice 1: Graphs and Graph Algorithms.** The core of the project. We build precinct adjacency graphs from geographic shapefiles and implement spectral partitioning (Fiedler vector bisection), minimum spanning tree diameter, Newman-Girvan modularity, BFS-based contiguity checking, and single-flip Metropolis MCMC entirely from scratch. NetworkX handles graph storage; all algorithms are hand-written.

**Choice 2: Social Networks.** Precinct adjacency networks exhibit community structure and homophily (neighboring precincts tend to share demographics and partisan lean). Gerrymandering is modeled as deliberate manipulation of community boundaries, and our modularity metric directly measures how well district boundaries respect natural graph communities.

**Choice 6: Game Theory, Auctions, Matching Markets.** Gerrymandering is a strategic optimization game: mapmakers maximize their party's seat count by manipulating district boundaries. The efficiency gap metric directly connects to mechanism design and fair outcome design, measuring the structural advantage one party gains through the allocation of wasted votes across districts.

## Work Breakdown

- **Sohum Trivedi:** Pipeline architecture, MCMC ensemble generation, spectral partitioning implementation, and outlier analysis.
- **Ronit Kapoor:** Data acquisition and cleaning (shapefiles, election returns, population data), graph construction, and preprocessing.
- **Aditi Ghosh:** Compactness and partisan fairness metric implementation, statistical analysis, and seats-votes curve generation.
- **Maurya Bonu:** Geographic and histogram visualizations, report writing, and final presentation preparation.

## AI Usage

Claude Code (Anthropic) was used extensively throughout this project as a coding assistant. Specifically: Claude wrote and debugged the MST diameter optimization (replacing Kruskal's algorithm with a direct two-BFS subgraph diameter approach, achieving a 170x speedup on Pennsylvania's 9,253-precinct graph), helped author the LaTeX report sections on results and conclusions, assisted with the multi-state MCMC pipeline scripts, and helped identify and fix bugs in the partition aggregation and boundary-edge update logic. All algorithmic ideas, experimental design, and interpretation of results were done by the team; Claude was used to accelerate implementation and writing.

## Pipeline

![pipeline](docs/figures/pipeline.png)

Raw shapefiles and election returns become a precinct adjacency graph. The graph is partitioned deterministically by spectral bisection and stochastically by single-flip MCMC. Each partition is scored by compactness and partisan-fairness metrics. The enacted plan's metric values are compared against the ensemble distribution to produce p-values.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download real shapefiles (PA, NC, MD, WI)
python scripts/download_mggg_states.py all

# Run full analysis
python scripts/run_real_all_states.py
```

## Repo Layout

```
gerrydetect/    importable Python package (all algorithms hand-written)
scripts/        CLI entry points for data download and analysis
tests/          pytest unit tests (61 tests)
report/         LaTeX report
docs/figures/   committed output figures
data/           gitignored raw and processed data
output/         gitignored figures and tables
```
