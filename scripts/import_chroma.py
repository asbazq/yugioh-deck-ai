"""
Import a Chroma collection from snapshot files into a local PersistentClient
or an HTTP server.

Inputs in snapshot directory:
- ids.json            : list[str]
- metadatas.json      : list[dict]
- embeddings.npy      : float32 array shape (N, D)

Usage (local persistent, recommended for edge):
  python scripts/import_chroma.py --in ./chroma_snapshot \
    --mode local --path /chroma --collection yugioh_256 --reset

Usage (remote HTTP server):
  python scripts/import_chroma.py --in ./chroma_snapshot \
    --mode http --host 1.2.3.4 --port 8000 --collection yugioh_256 --reset

If args are omitted, reads from environment or .env.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import chromadb
import dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, help="Snapshot directory")
    p.add_argument("--mode", choices=["local", "http"], help="Import mode")
    p.add_argument("--collection", help="Collection name")
    p.add_argument("--batch", type=int, default=1000, help="Batch size to add")
    p.add_argument("--reset", action="store_true", help="Drop existing collection before import")
    # local
    p.add_argument("--path", help="Persistent path (local mode)")
    # http
    p.add_argument("--host", help="HTTP host (http mode)")
    p.add_argument("--port", type=int, help="HTTP port (http mode)")
    return p.parse_args()


def getenv(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key, default)
    if v is None:
        return None
    return v.strip('"').strip("'")


def main() -> None:
    dotenv.load_dotenv(".env")
    args = parse_args()

    mode = (args.mode or getenv("chroma_mode", "local")).lower()
    collection_name = args.collection or getenv("chroma_collection", "cards")

    if mode == "http":
        host = args.host or getenv("host", "localhost")
        port = args.port or int(getenv("chroma_port", "8000"))
        client = chromadb.HttpClient(host=host, port=port)
    else:
        path = args.path or getenv("chroma_path", "/chroma")
        Path(path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=path)

    # (Re)create collection
    if args.reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    # Load snapshot
    src = Path(args.src)
    ids: List[str] = json.loads((src / "ids.json").read_text(encoding="utf-8"))
    metas: List[dict] = json.loads((src / "metadatas.json").read_text(encoding="utf-8"))
    embeds: np.ndarray = np.load(src / "embeddings.npy")

    if embeds.shape[0] != len(ids) or len(ids) != len(metas):
        raise RuntimeError("Snapshot parts have different lengths")

    # Import in batches
    n = len(ids)
    bsz = int(args.batch)
    for off in range(0, n, bsz):
        sl = slice(off, min(off + bsz, n))
        col.add(ids=ids[sl], metadatas=metas[sl], embeddings=embeds[sl].tolist())
        print(f"Imported {min(off+bsz, n)}/{n}")

    print("Done.")


if __name__ == "__main__":
    main()

