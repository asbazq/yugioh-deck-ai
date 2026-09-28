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
import uuid
import fcntl
from pathlib import Path

import dotenv

try:
    from .snapshot_utils import load_snapshot
except ImportError:
    from snapshot_utils import load_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, help="Snapshot directory")
    p.add_argument("--mode", choices=["local", "http"], help="Import mode")
    p.add_argument("--collection", help="Collection name")
    p.add_argument("--batch", type=int, default=1000, help="Batch size to add")
    p.add_argument("--state-dir", default=os.getenv("CONTINUOUS_STATE_DIR", "runs/continuous"))
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
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    # Validate before marking the index pending or touching the database.
    snapshot = load_snapshot(Path(args.src))
    root = Path(args.state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "cycle.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / "external-index.json"
        write_revision(marker, pending=True)
        import_snapshot(args, snapshot)
        write_revision(marker, pending=False)


def write_revision(path, *, pending):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pending": pending, "revision": uuid.uuid4().hex}))
    temporary.replace(path)


def import_snapshot(args, snapshot):
    import chromadb

    ids, metas, embeds = snapshot
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
    col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    if args.reset:
        client.delete_collection(collection_name)
        col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    # Import in batches
    n = len(ids)
    bsz = int(args.batch)
    for off in range(0, n, bsz):
        sl = slice(off, min(off + bsz, n))
        col.upsert(ids=ids[sl], metadatas=metas[sl], embeddings=embeds[sl].tolist())
        print(f"Imported {min(off+bsz, n)}/{n}")

    print("Done.")


if __name__ == "__main__":
    main()
