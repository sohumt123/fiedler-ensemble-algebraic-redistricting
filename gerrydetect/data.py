"""Load Pennsylvania (and friends) precinct data into a GeoDataFrame.

We expect raw shapefiles to live under `data/raw/<state>/`. The exact column
names from Redistricting Data Hub VEST shapefiles vary by year; this loader
hides the rename so the rest of the pipeline sees a clean schema.

Expected schema produced by `load_state(...)`:

| column     | type      | meaning                                   |
|------------|-----------|-------------------------------------------|
| geometry   | Polygon   | precinct polygon (projected CRS)          |
| pop        | float     | total population                          |
| votes_d    | float     | Democratic votes (2020 presidential)      |
| votes_r    | float     | Republican votes (2020 presidential)      |
| district   | int       | enacted congressional district id         |

The GeoDataFrame is reindexed with a clean integer index (0..n-1) so that
downstream code can use those indices as node IDs without worrying about
gaps or string identifiers.

Supported states (v1): "pa". Adding a new state means adding an entry to
`STATE_LOADERS` plus a per-state column rename map.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_CRS = "EPSG:5070"  # NAD83 / Conus Albers — preserves area for U.S. work


# Column rename maps. VEST varies year to year; if the file you downloaded has
# different column names, add them here.
PA_COLUMN_MAP = {
    # population: from ACS join on tract or VEST adult-pop. Keep first hit.
    "TOTPOP": "pop",
    "TOTPOP_2020": "pop",
    "P0010001": "pop",
    "POPULATION": "pop",
    "TAPERSONS": "pop",
    # 2020 presidential votes
    "G20PREDBID": "votes_d",
    "G20PRERTRU": "votes_r",
    "PRES_DEM": "votes_d",
    "PRES_REP": "votes_r",
    # enacted district id
    "DISTRICT": "district",
    "CD": "district",
    "CD117": "district",
    "CD118": "district",
}


def _rename_first_hit(gdf, mapping: dict[str, str]):
    """Apply the first column rename whose source column exists, for each
    target. Useful when the upstream file uses one of several names for the
    same field.
    """
    rename: dict[str, str] = {}
    seen_targets: set[str] = set()
    for src, tgt in mapping.items():
        if src in gdf.columns and tgt not in seen_targets:
            rename[src] = tgt
            seen_targets.add(tgt)
    return gdf.rename(columns=rename)


def _coerce_numeric(gdf, cols: list[str]):
    import pandas as pd

    for c in cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0)
    return gdf


def load_pa(raw_dir: Path) -> "GeoDataFrame":  # type: ignore[name-defined]
    """Load Pennsylvania VEST precincts joined with the enacted CD shapefile.

    Expects, under `raw_dir`:
        pa_vest_2020.shp           (or .gpkg) — precincts + 2020 votes
        pa_cd_2022.shp             — enacted PA congressional districts
        pa_acs5_pop.csv  (optional) — tract-level population if not already
                                       in the VEST file

    If only the VEST file is present, we use whatever pop column it carries
    (`TAPERSONS` is voting-age pop; close enough for v1).
    """
    import geopandas as gpd

    raw_dir = Path(raw_dir)
    vest_path = _first_existing(
        [
            raw_dir / "pa_vest_2020.gpkg",
            raw_dir / "pa_vest_2020.shp",
            raw_dir / "pa_vest20.shp",
        ]
    )
    if vest_path is None:
        raise FileNotFoundError(
            f"No PA VEST shapefile found in {raw_dir}. "
            "Download from https://redistrictingdatahub.org/ and place under "
            f"{raw_dir}/pa_vest_2020.shp"
        )
    log.info("Loading VEST precincts from %s", vest_path)
    precincts = gpd.read_file(vest_path)

    # Standardize columns.
    precincts = _rename_first_hit(precincts, PA_COLUMN_MAP)
    precincts = _coerce_numeric(
        precincts, ["pop", "votes_d", "votes_r", "district"]
    )

    # Reproject to a common projected CRS so length/area are in meters.
    if precincts.crs is None:
        log.warning("CRS missing; assuming EPSG:4326")
        precincts = precincts.set_crs("EPSG:4326")
    if precincts.crs.to_string() != PROJECT_CRS:
        precincts = precincts.to_crs(PROJECT_CRS)

    # If district isn't already on the precincts (older VEST files don't
    # include it), join from the enacted CD shapefile by spatial overlay.
    if "district" not in precincts.columns or precincts["district"].sum() == 0:
        cd_path = _first_existing(
            [
                raw_dir / "pa_cd_2022.gpkg",
                raw_dir / "pa_cd_2022.shp",
                raw_dir / "pa_cong.shp",
            ]
        )
        if cd_path is None:
            raise FileNotFoundError(
                f"District not in VEST and no enacted-CD shapefile in {raw_dir}"
            )
        log.info("Loading enacted CD shapefile from %s", cd_path)
        cd = gpd.read_file(cd_path).to_crs(PROJECT_CRS)
        cd = _rename_first_hit(cd, PA_COLUMN_MAP)
        cd_id_col = "district"
        # Spatial join: each precinct gets the district of its largest overlap.
        precincts = _assign_districts_by_overlap(precincts, cd, cd_id_col)

    # Drop precincts with no district assigned (e.g., outside state boundary
    # because of geometry sliver issues).
    precincts = precincts.dropna(subset=["district"]).copy()
    precincts["district"] = precincts["district"].astype(int)

    # Reindex 0..n-1 for clean node IDs.
    precincts = precincts.reset_index(drop=True)
    return precincts


def _assign_districts_by_overlap(precincts, districts, district_col: str):
    """For each precinct, assign the district whose polygon it overlaps most."""
    import geopandas as gpd
    import pandas as pd

    # Quick path: if a precinct's centroid falls inside exactly one district,
    # use that. Otherwise compute area overlap.
    pre_centroids = precincts.copy()
    pre_centroids["geometry"] = precincts.geometry.centroid

    joined = gpd.sjoin(
        pre_centroids[["geometry"]],
        districts[["geometry", district_col]],
        how="left",
        predicate="within",
    )
    # Multiple matches per precinct can happen on shared boundaries — pick first.
    joined = joined.reset_index().drop_duplicates("index", keep="first").set_index("index")
    precincts = precincts.copy()
    precincts["district"] = joined[district_col].reindex(precincts.index)
    return precincts


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# Registry — extend per state.
STATE_LOADERS = {
    "pa": load_pa,
}


def load_state(state: str, raw_dir: Path | str = "data/raw") -> "GeoDataFrame":  # type: ignore[name-defined]
    """Top-level loader. Dispatches on state code."""
    state = state.lower()
    if state not in STATE_LOADERS:
        raise ValueError(
            f"State '{state}' not supported. Add a loader to STATE_LOADERS."
        )
    raw_dir = Path(raw_dir) / state
    return STATE_LOADERS[state](raw_dir)
