"""Download precinct shapefiles for a state. Usage: python scripts/download_data.py pa"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"

# Direct download URLs by state. Where we don't have a stable direct link
# (RDH typically requires login), the value is None and we print a manual
# instruction instead.
SOURCES = {
    "pa": {
        "vest_zip": None,  # https://redistrictingdatahub.org/dataset/vest-2020-pennsylvania-precinct-and-election-results/
        "cd_zip": None,    # https://redistrictingdatahub.org/dataset/2022-pennsylvania-congressional-districts/
        "instructions": (
            "Manual download required for Pennsylvania:\n\n"
            "  1. Create a free account at https://redistrictingdatahub.org/\n"
            "  2. Download:\n"
            "     - 'VEST 2020 Pennsylvania precinct and election results'\n"
            "       Save to: data/raw/pa/pa_vest_2020.zip (or extract directly)\n"
            "     - '2022 Pennsylvania Congressional Districts (Adopted)'\n"
            "       Save to: data/raw/pa/pa_cd_2022.zip\n"
            "  3. Unzip both archives in data/raw/pa/\n"
            "  4. Re-run scripts/build_graph.py\n"
        ),
    },
}


def _download(url: str, dest: Path, label: str) -> None:
    """Stream-download with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=label
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 14):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest_dir)


def main(state: str) -> int:
    state = state.lower()
    if state not in SOURCES:
        print(f"Unknown state: {state}. Supported: {list(SOURCES)}", file=sys.stderr)
        return 1

    state_dir = DATA_DIR / state
    state_dir.mkdir(parents=True, exist_ok=True)
    cfg = SOURCES[state]

    any_direct = False
    for key, url in cfg.items():
        if key == "instructions" or url is None:
            continue
        any_direct = True
        archive = state_dir / f"{key}.zip"
        _download(url, archive, label=key)
        _extract(archive, state_dir)

    if not any_direct:
        print(cfg["instructions"])
    else:
        print(f"Done. Files in {state_dir}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="two-letter state code (e.g., pa)")
    args = parser.parse_args()
    sys.exit(main(args.state))
