"""
Container startup hook:
- Optionally auto-import a Chroma snapshot into local PersistentClient before starting the API.

Controlled by env vars:
- chroma_mode: 'local' or 'http' (only 'local' triggers import)
- chroma_path: persistent path for local client (default: /chroma)
- chroma_collection: collection name
- AUTO_IMPORT: '1' to enable (default: '0')
- SNAPSHOT_TGZ: path to snapshot tar.gz (optional but required when AUTO_IMPORT=1)
- SNAPSHOT_SHA256: path to checksum file (optional). If provided, will be verified.
- IMPORT_RESET: '1' to drop collection before import (default: '0')
- IMPORT_ON_EMPTY: '1' to import only when collection empty (default: '1')
- IMPORT_BATCH: batch size for import (default: 1000)

Idempotency:
- Stores collection-specific completed/pending hashes under chroma_path.
- Resumes an interrupted import with upsert, even when IMPORT_ON_EMPTY=1.
"""
from __future__ import annotations

import os
from pathlib import Path
import hashlib

from dotenv import load_dotenv

try:
    from .snapshot_utils import verify_checksum, extract_snapshot, load_snapshot
except ImportError:  # python scripts/entrypoint.py
    from snapshot_utils import verify_checksum, extract_snapshot, load_snapshot


def getenv(key: str, default: str | None = None) -> str:
    v = os.getenv(key, default)
    return (v or "").strip('"').strip("'")


def main() -> None:
    load_dotenv(".env")

    mode = getenv("chroma_mode", "local").lower()
    if mode != "local":
        print("[entrypoint] chroma_mode != local; skipping auto-import")
        return

    auto = getenv("AUTO_IMPORT", "0") == "1"
    if not auto:
        print("[entrypoint] AUTO_IMPORT disabled; skipping")
        return

    chroma_path = Path(getenv("chroma_path", "/chroma"))
    chroma_path.mkdir(parents=True, exist_ok=True)
    collection = getenv("chroma_collection", "cards")
    batch = int(getenv("IMPORT_BATCH", "1000"))
    if batch < 1:
        raise ValueError("IMPORT_BATCH must be positive")
    import_on_empty = getenv("IMPORT_ON_EMPTY", "1") == "1"
    do_reset = getenv("IMPORT_RESET", "0") == "1"

    snap_tgz = getenv("SNAPSHOT_TGZ", "")
    snap_sha = getenv("SNAPSHOT_SHA256", "")
    if not snap_tgz:
        raise ValueError("SNAPSHOT_TGZ is required when AUTO_IMPORT=1")

    tgz = Path(snap_tgz)
    if not tgz.exists():
        raise FileNotFoundError(tgz)

    # compute current package hash (from file or .sha256)
    pkg_hash = verify_checksum(tgz, Path(snap_sha) if snap_sha else None)

    collection_key = hashlib.sha256(collection.encode()).hexdigest()[:16]
    hash_file = chroma_path / f".snapshot_hash_{collection_key}"
    pending_file = chroma_path / f".snapshot_pending_{collection_key}"
    resuming = pending_file.exists() and pending_file.read_text().strip() == pkg_hash
    if hash_file.exists() and not do_reset and not resuming:
        last = hash_file.read_text(encoding="utf-8").strip()
        if last == pkg_hash:
            print("[entrypoint] Snapshot already imported; skipping")
            return

    # connect to local persistent client
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    # skip if only-on-empty and collection has data
    if import_on_empty and not do_reset and not resuming:
        cnt = col.count()
        if cnt > 0:
            print(f"[entrypoint] Collection not empty (count={cnt}); skipping import")
            return

    # Extract in memory and import using scripts/import_chroma-like logic
    import tempfile

    with tempfile.TemporaryDirectory() as tmpd:
        extract_snapshot(tgz, Path(tmpd))
        ids, metas, embeds = load_snapshot(Path(tmpd))
        if do_reset:
            client.delete_collection(collection)
            col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

        pending_file.write_text(pkg_hash, encoding="utf-8")
        n = len(ids)
        for off in range(0, n, batch):
            sl = slice(off, min(off + batch, n))
            col.upsert(ids=ids[sl], metadatas=metas[sl], embeddings=embeds[sl].tolist())
            print(f"[entrypoint] Imported {min(off+batch, n)}/{n}")

    hash_file.write_text(pkg_hash, encoding="utf-8")
    pending_file.unlink(missing_ok=True)
    print("[entrypoint] Import complete")


if __name__ == "__main__":
    main()
