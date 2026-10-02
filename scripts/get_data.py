"""Download the two public datasets into data/.

    data/secom/            secom.data, secom_labels.data
    data/gas/Dataset/      batch1.dat ... batch10.dat

Both come from the UCI Machine Learning Repository. See docs/LOG.md for the
citations their authors ask for.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "secom": "https://archive.ics.uci.edu/static/public/179/secom.zip",
    "gas": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
}


def main() -> int:
    for name, url in SOURCES.items():
        target = Path("data") / name
        if target.exists() and any(target.iterdir()):
            print(f"{target} already present, skipping")
            continue
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URLs above
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        target.mkdir(parents=True, exist_ok=True)
        archive.extractall(target)
        print(f"  -> {target} ({len(archive.namelist())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
