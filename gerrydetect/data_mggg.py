"""Load MGGG-format state VTD shapefiles into the gerrydetect pipeline.

Supported states (all MIT-licensed, publicly available on GitHub — no login):
  PA — https://github.com/mggg-states/PA-shapefiles
  NC — https://github.com/mggg-states/NC-shapefiles
  MD — https://github.com/mggg-states/MD-shapefiles
  WI — https://github.com/mggg-states/WI-shapefiles

Column names sourced from each repo's README / data dictionary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_CRS = "EPSG:5070"  # NAD83 / Conus Albers — preserves area for U.S. work

# ---------------------------------------------------------------------------
# Column maps — one per (state, election) combination
# ---------------------------------------------------------------------------

MGGG_PA_COLUMN_MAP_2016 = {
    "TOTPOP": "pop",
    "T16PRESD": "votes_d",
    "T16PRESR": "votes_r",
    "REMEDIAL": "district",  # 2018 remedial congressional plan (court-ordered)
}

MGGG_PA_COLUMN_MAP_2012 = {
    "TOTPOP": "pop",
    "PRES12D": "votes_d",
    "PRES12R": "votes_r",
    "CD_2011": "district",  # 2011 enacted congressional plan
}

# NC: 2016 General Election presidential results, 2016 enacted CD plan (13 seats)
MGGG_NC_COLUMN_MAP = {
    "TOTPOP": "pop",
    "EL16G_PR_D": "votes_d",
    "EL16G_PR_R": "votes_r",
    "CD": "district",
}

# MD: 2016 presidential results, 2011 enacted CD plan (8 seats)
MGGG_MD_COLUMN_MAP = {
    "TOTPOP": "pop",
    "PRES16D": "votes_d",
    "PRES16R": "votes_r",
    "CD": "district",
}

# WI: 2011 wards with 2016 presidential results, 2011 enacted CD plan (8 seats)
# PERSONS is a fallback in case TOTPOP is absent in some releases.
MGGG_WI_COLUMN_MAP = {
    "TOTPOP": "pop",
    "PERSONS": "pop",
    "PREDEM16": "votes_d",
    "PREREP16": "votes_r",
    "CON": "district",
}

# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------


def _find_shp(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.shp"))
    if not candidates:
        raise FileNotFoundError(f"No .shp file found in {raw_dir}.")
    return candidates[0]


def _load_mggg(
    raw_dir: Path,
    col_map: dict[str, str],
    state: str,
) -> gpd.GeoDataFrame:
    """Generic MGGG shapefile loader — rename columns, reproject, clean index."""
    raw_dir = Path(raw_dir)
    try:
        shp_path = _find_shp(raw_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No shapefile in {raw_dir}. "
            f"Run `python scripts/download_mggg_states.py {state}` first."
        ) from None

    log.info("Loading %s from %s", state.upper(), shp_path)
    gdf = gpd.read_file(shp_path)
    log.info("Loaded %d rows × %d cols", len(gdf), len(gdf.columns))

    # Rename columns: first matching source wins per target.
    seen: set[str] = set()
    rename: dict[str, str] = {}
    for src, tgt in col_map.items():
        if src in gdf.columns and tgt not in seen:
            rename[src] = tgt
            seen.add(tgt)
    if missing := ({"pop", "votes_d", "votes_r", "district"} - seen):
        log.warning("%s: columns not found in shapefile: %s", state.upper(), missing)
    gdf = gdf.rename(columns=rename)

    for col in ("pop", "votes_d", "votes_r", "district"):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    if gdf.crs is None:
        log.warning("CRS missing on %s; assuming EPSG:4326", state.upper())
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)

    gdf = gdf[gdf["pop"] > 0].copy()
    if "district" in gdf.columns:
        gdf["district"] = gdf["district"].astype(int)
    gdf = gdf.reset_index(drop=True)

    log.info(
        "%s: %d VTDs, pop=%d, D-votes=%d, R-votes=%d, %d districts",
        state.upper(), len(gdf),
        int(gdf["pop"].sum()),
        int(gdf.get("votes_d", 0).sum()) if "votes_d" in gdf.columns else 0,
        int(gdf.get("votes_r", 0).sum()) if "votes_r" in gdf.columns else 0,
        gdf["district"].nunique() if "district" in gdf.columns else 0,
    )
    return gdf


# ---------------------------------------------------------------------------
# Per-state public loaders
# ---------------------------------------------------------------------------


def load_mggg_pa(
    raw_dir: str | Path = "data/raw/pa_mggg",
    election: str = "2016",
) -> gpd.GeoDataFrame:
    """Load Pennsylvania MGGG VTD shapefile (2016 or 2012 election)."""
    col_map = MGGG_PA_COLUMN_MAP_2016 if election == "2016" else MGGG_PA_COLUMN_MAP_2012
    return _load_mggg(Path(raw_dir), col_map, "pa")


def load_mggg_nc(raw_dir: str | Path = "data/raw/nc_mggg") -> gpd.GeoDataFrame:
    """Load North Carolina MGGG VTD shapefile (2016 presidential, 2016 enacted CD)."""
    return _load_mggg(Path(raw_dir), MGGG_NC_COLUMN_MAP, "nc")


def load_mggg_md(raw_dir: str | Path = "data/raw/md_mggg") -> gpd.GeoDataFrame:
    """Load Maryland MGGG precinct shapefile (2016 presidential, 2011 enacted CD)."""
    return _load_mggg(Path(raw_dir), MGGG_MD_COLUMN_MAP, "md")


def load_mggg_wi(raw_dir: str | Path = "data/raw/wi_mggg") -> gpd.GeoDataFrame:
    """Load Wisconsin MGGG ward shapefile (2016 presidential, 2011 enacted CD)."""
    return _load_mggg(Path(raw_dir), MGGG_WI_COLUMN_MAP, "wi")


MGGG_LOADERS = {
    "pa": load_mggg_pa,
    "nc": load_mggg_nc,
    "md": load_mggg_md,
    "wi": load_mggg_wi,
}
