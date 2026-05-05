"""Download MGGG state shapefiles (PA/NC/MD/WI). Usage: python scripts/download_mggg_states.py all"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCES: dict[str, dict] = {
    "pa": {
        "dir": "pa_mggg",
        "url": "https://github.com/mggg-states/PA-shapefiles/raw/master/PA.zip",
        "label": "PA VTD shapefile — MGGG (2016 presidential, 2018 remedial CD)",
    },
    "nc": {
        "dir": "nc_mggg",
        "url": "https://github.com/mggg-states/NC-shapefiles/raw/master/NC_VTD.zip",
        "label": "NC VTD shapefile — MGGG (2016 presidential, 2016 enacted CD)",
    },
    "md": {
        "dir": "md_mggg",
        "url": "https://github.com/mggg-states/MD-shapefiles/raw/master/MD_precincts.zip",
        "label": "MD precinct shapefile — MGGG (2016 presidential, 2011 enacted CD)",
    },
    "wi": {
        "dir": "wi_mggg",
        "url": "https://github.com/mggg-states/WI-shapefiles/raw/master/WI_2011_wards.zip",
        "label": "WI 2011 ward shapefile — MGGG (2016 presidential, 2011 enacted CD)",
    },
}


def _download_and_extract(state: str) -> int:
    cfg = SOURCES[state]
    dest_dir = REPO_ROOT / "data" / "raw" / cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Skip if shapefile already present.
    existing = list(dest_dir.glob("*.shp"))
    if existing:
        print(f"  {state.upper()}: shapefile already exists at {existing[0].name} — skipping.")
        return 0

    print(f"\nDownloading {cfg['label']} ...")
    try:
        resp = requests.get(cfg["url"], stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ERROR downloading {state.upper()}: {exc}", file=sys.stderr)
        return 1

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=state.upper()) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 14):
            if chunk:
                buf.write(chunk)
                bar.update(len(chunk))
    buf.seek(0)

    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)
        print(f"  Extracted {len(zf.namelist())} files → {dest_dir}")

    for p in sorted(dest_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:50s} {size_kb:8.1f} KB")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"Available states: {list(SOURCES)}")
        return 1

    targets = [a.lower() for a in sys.argv[1:]]
    if "all" in targets:
        states = list(SOURCES)
    else:
        states = targets

    for s in states:
        if s not in SOURCES:
            print(f"Unknown state: {s}. Available: {list(SOURCES)}", file=sys.stderr)
            return 1

    for s in states:
        rc = _download_and_extract(s)
        if rc != 0:
            return rc

    print("\nDone. Run the analysis:")
    print("  python scripts/run_real_all_states.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
