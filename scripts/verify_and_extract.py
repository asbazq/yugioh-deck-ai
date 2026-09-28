"""
Verify a tar.gz with a .sha256 file and extract it to a target directory.

Usage:
  python scripts/verify_and_extract.py \
    --tgz /chroma/snapshots/yugioh_256_20240901.tar.gz \
    --sha /chroma/snapshots/yugioh_256_20240901.sha256 \
    --out /chroma/imported --clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .snapshot_utils import verify_checksum, extract_snapshot
except ImportError:
    from snapshot_utils import verify_checksum, extract_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tgz", required=True, help="Path to snapshot tar.gz")
    p.add_argument("--sha", required=True, help="Path to sha256 file")
    p.add_argument("--out", required=True, help="Directory to extract to")
    p.add_argument("--clean", action="store_true", help="Clean output directory before extracting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tgz = Path(args.tgz)
    sha = Path(args.sha)
    out = Path(args.out)

    if not tgz.exists():
        raise FileNotFoundError(tgz)
    if not sha.exists():
        raise FileNotFoundError(sha)

    verify_checksum(tgz, sha)
    extract_snapshot(tgz, out, clean=args.clean)
    print(f"Extracted to {out}")


if __name__ == "__main__":
    main()
