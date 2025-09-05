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
import hashlib
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tgz", required=True, help="Path to snapshot tar.gz")
    p.add_argument("--sha", required=True, help="Path to sha256 file")
    p.add_argument("--out", required=True, help="Directory to extract to")
    p.add_argument("--clean", action="store_true", help="Clean output directory before extracting")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    tgz = Path(args.tgz)
    sha = Path(args.sha)
    out = Path(args.out)

    if not tgz.exists():
        raise FileNotFoundError(tgz)
    if not sha.exists():
        raise FileNotFoundError(sha)

    expected = sha.read_text(encoding="utf-8").strip().split()[0]
    got = sha256_file(tgz)
    if got != expected:
        raise RuntimeError(f"Checksum mismatch: expected {expected}, got {got}")

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(out)
    print(f"Extracted to {out}")


if __name__ == "__main__":
    main()

