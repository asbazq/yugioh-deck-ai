"""
Package a Chroma snapshot directory into a tar.gz and generate a SHA256 file.

Inputs (directory):
- ids.json
- metadatas.json
- embeddings.npy

Outputs:
- <name>.tar.gz
- <name>.sha256  (contains: "<sha256>  <filename>")

Usage:
  python scripts/pack_snapshot.py --src ./chroma_snapshot --out ./dist \
    --name yugioh_256_20240901
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Snapshot directory")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--name", required=True, help="Base name for the package")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / f"{args.name}.tar.gz"
    sha_path = out / f"{args.name}.sha256"

    # Create tar.gz
    with tarfile.open(tar_path, "w:gz") as tar:
        for fn in ("ids.json", "metadatas.json", "embeddings.npy"):
            p = src / fn
            if not p.exists():
                raise FileNotFoundError(f"Missing {fn} in {src}")
            tar.add(p, arcname=fn)

    # Compute sha256
    digest = sha256_file(tar_path)
    sha_path.write_text(f"{digest}  {tar_path.name}\n", encoding="utf-8")
    print(f"Wrote {tar_path} and {sha_path}")


if __name__ == "__main__":
    main()

