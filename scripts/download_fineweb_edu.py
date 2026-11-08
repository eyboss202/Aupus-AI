"""Download the first FineWeb-Edu ``sample-10BT`` parquet files (Milestone 7).

Why download first instead of streaming straight into ``prepare_data.py``?
The full set takes about an hour to fetch. If the connection drops during a
stream, the whole pipeline run starts over. Downloading files first makes the
network part resumable (an interrupted file picks up where it stopped when you
re-run this script) and checkable (each file's SHA-256 is compared with the
hash the Hub publishes). Processing then reads from local disk.

The files are read in the same order ``datasets`` streams them, so
``prepare_data.py --source parquet`` sees the same documents in the same
order as ``--source fineweb-edu`` would.

Sizes at the pinned revision: 13 files of 2.15 GB (~725K documents each) and
a last one of 0.54 GB. Four files (~2.9M documents, ~3B tokens before
filtering and dedup) cover the 2.5B-token budget for ``small``.

Usage:
    python scripts/download_fineweb_edu.py                 # first 4 files, 8.6 GB
    python scripts/download_fineweb_edu.py --files 1       # one file, 2.15 GB
    python scripts/download_fineweb_edu.py --list          # show the files, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from aupus.data.pipeline import FINEWEB_EDU, FINEWEB_EDU_REVISION

PREFIX = "sample/10BT/"  # the `sample-10BT` config's files in the dataset repo


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/fineweb_edu"))
    parser.add_argument("--files", type=int, default=4, help="how many files to fetch, in order")
    parser.add_argument("--list", action="store_true", help="print the file list and exit")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download  # comes with `pip install -e ".[data]"`

    info = HfApi().dataset_info(FINEWEB_EDU, revision=FINEWEB_EDU_REVISION, files_metadata=True)
    files = sorted((s for s in info.siblings if s.rfilename.startswith(PREFIX)), key=lambda s: s.rfilename)
    if args.list:
        for s in files:
            print(f"{s.rfilename}  {s.size / 1e9:.2f} GB")
        return
    chosen = files[: args.files]
    total = sum(s.size for s in chosen)
    print(f"{len(chosen)} of {len(files)} files, {total / 1e9:.2f} GB -> {args.out_dir}/{PREFIX}", flush=True)

    start = time.time()
    for s in chosen:
        path = Path(
            hf_hub_download(
                FINEWEB_EDU,
                s.rfilename,
                repo_type="dataset",
                revision=FINEWEB_EDU_REVISION,
                local_dir=args.out_dir,
            )
        )
        digest = sha256_file(path)
        if s.lfs is None or digest != s.lfs.sha256:
            path.unlink()
            raise SystemExit(f"{s.rfilename}: SHA-256 mismatch (deleted, re-run to fetch it again)")
        size_gb = path.stat().st_size / 1e9
        print(f"{s.rfilename}: {size_gb:.2f} GB, sha256 ok ({time.time() - start:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
