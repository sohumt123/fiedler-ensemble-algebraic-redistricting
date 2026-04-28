"""Build the precinct adjacency graph for a state and pickle it.

Usage:
    python scripts/build_graph.py pa
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("build_graph")


def main(state: str) -> int:
    from gerrydetect.data import load_state
    from gerrydetect.graph import build_graph, graph_summary

    state = state.lower()
    raw_dir = REPO_ROOT / "data" / "raw"
    proc_dir = REPO_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s precincts ...", state)
    gdf = load_state(state, raw_dir=raw_dir)
    log.info("Loaded %d precincts", len(gdf))

    log.info("Building adjacency graph ...")
    graph = build_graph(gdf)
    summary = graph_summary(graph)
    log.info("Graph: %s", summary)

    # Persist precinct gdf as parquet (geometry serialized via pyarrow ext type).
    gdf_path = proc_dir / f"{state}_precincts.parquet"
    gdf.to_parquet(gdf_path)
    log.info("Wrote %s", gdf_path)

    graph_path = proc_dir / f"{state}_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    log.info("Wrote %s", graph_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="two-letter state code (e.g., pa)")
    args = parser.parse_args()
    sys.exit(main(args.state))
