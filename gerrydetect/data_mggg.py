"""Load MGGG-format PA VTD shapefile into the gerrydetect pipeline.

The MGGG PA shapefile (from https://github.com/mggg-states/PA-shapefiles)
uses column names from the 2010 census / VRDI processing. This module
maps those to our internal schema.

The shapefile contains ~9,000 VTDs with:
  - TOTPOP: total population (2010 census)
  - T16PRESD / T16PRESR: 2016 presidential D/R votes
  - PRES12D / PRES12R: 2012 presidential D/R votes
  - REMEDIAL: 2018 remedial congressional district assignment
  - CD_2011: 2011 enacted congressional district assignment

We use the 2016 presidential election and the 2018 remedial plan by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_CRS = "EPSG:5070"  # NAD83 / Conus Albers — same as main data.py

# Column name mappings for the MGGG PA shapefile.
MGGG_PA_COLUMN_MAP_2016 = {
    "TOTPOP": "pop",
    "T16PRESD": "votes_d",
    "T16PRESR": "votes_r",
    "REMEDIAL": "district",  # 2018 remedial congressional plan
}

MGGG_PA_COLUMN_MAP_2012 = {
    "TOTPOP": "pop",
    "PRES12D": "votes_d",
    "PRES12R": "votes_r",
    "CD_2011": "district",  # 2011 enacted congressional plan
}


def load_mggg_pa(
    raw_dir: str | Path = "data/raw/pa_mggg",
    election: str = "2016",
) -> gpd.GeoDataFrame:
    """Load the MGGG PA VTD shapefile.

    Args:
        raw_dir: path to the extracted MGGG PA data.
        election: "2016" or "2012" — which presidential election's vote
            totals to use.

    Returns:
        GeoDataFrame with columns: geometry, pop, votes_d, votes_r, district.
        Projected to EPSG:5070. Indexed 0..n-1.
    """
    raw_dir = Path(raw_dir)

    # Find the shapefile.
    candidates = list(raw_dir.glob("*.shp"))
    if not candidates:
        raise FileNotFoundError(
            f"No .shp file found in {raw_dir}. "
            "Run `python scripts/download_mggg_pa.py` first."
        )
    shp_path = candidates[0]
    log.info("Loading MGGG PA shapefile from %s", shp_path)

    gdf = gpd.read_file(shp_path)
    log.info("Loaded %d VTDs with %d columns", len(gdf), len(gdf.columns))

    # Pick column map based on election year.
    if election == "2016":
        col_map = MGGG_PA_COLUMN_MAP_2016
    elif election == "2012":
        col_map = MGGG_PA_COLUMN_MAP_2012
    else:
        raise ValueError(f"Unknown election year: {election}. Use '2016' or '2012'.")

    # Rename columns.
    rename = {}
    for src, tgt in col_map.items():
        if src in gdf.columns:
            rename[src] = tgt
        else:
            log.warning("Expected column %s not found in shapefile", src)
    gdf = gdf.rename(columns=rename)

    # Coerce numeric.
    for col in ["pop", "votes_d", "votes_r", "district"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    # Reproject.
    if gdf.crs is None:
        log.warning("CRS missing; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)

    # Drop VTDs with zero population (water-only, etc.)
    gdf = gdf[gdf["pop"] > 0].copy()

    # Convert district to int.
    gdf["district"] = gdf["district"].astype(int)

    # Clean index.
    gdf = gdf.reset_index(drop=True)

    log.info(
        "Processed: %d VTDs, total pop=%d, D-votes=%d, R-votes=%d, %d districts",
        len(gdf),
        int(gdf["pop"].sum()),
        int(gdf["votes_d"].sum()),
        int(gdf["votes_r"].sum()),
        gdf["district"].nunique(),
    )

    return gdf
