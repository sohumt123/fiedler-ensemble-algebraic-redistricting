"""Download MGGG PA VTD shapefile from GitHub (MIT license). Usage: python scripts/download_mggg_pa.py"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "pa_mggg"

# Direct download URL for the PA VTD shapefile from MGGG's public GitHub repo.
PA_ZIP_URL = "https://github.com/mggg-states/PA-shapefiles/raw/master/PA.zip"

# Supplementary election CSV with 2014-2020 results.
PA_CSV_URL = "https://github.com/mggg-states/PA-shapefiles/raw/master/14-20csv.csv"


def download_with_progress(url: str, label: str) -> bytes:
    """Stream-download with a progress bar, return content bytes."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=label) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 14):
            if chunk:
                buf.write(chunk)
                bar.update(len(chunk))
    return buf.getvalue()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download and extract the PA shapefile zip.
    print(f"Downloading PA VTD shapefile from MGGG GitHub ...")
    content = download_with_progress(PA_ZIP_URL, "PA.zip")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(DATA_DIR)
        print(f"Extracted {len(zf.namelist())} files to {DATA_DIR}")

    # Download the supplementary election CSV.
    print(f"\nDownloading supplementary election CSV ...")
    csv_content = download_with_progress(PA_CSV_URL, "14-20csv.csv")
    csv_path = DATA_DIR / "14-20csv.csv"
    csv_path.write_bytes(csv_content)
    print(f"Saved to {csv_path}")

    # List what we got.
    print(f"\nFiles in {DATA_DIR}:")
    for p in sorted(DATA_DIR.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:40s} {size_mb:8.2f} MB")

    print("\nDone! Run `python scripts/run_real_pa.py` to analyze.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
