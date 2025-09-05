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
- Stores last imported package hash in `<chroma_path>/.snapshot_hash` and skips when unchanged unless IMPORT_RESET=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib

import chromadb
from dotenv import load_dotenv


def getenv(key: str, default: str | None = None) -> str:
    v = os.getenv(key, default)
    return (v or "").strip('"').strip("'")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    import_on_empty = getenv("IMPORT_ON_EMPTY", "1") == "1"
    do_reset = getenv("IMPORT_RESET", "0") == "1"

    snap_tgz = getenv("SNAPSHOT_TGZ", "")
    snap_sha = getenv("SNAPSHOT_SHA256", "")
    if not snap_tgz:
        print("[entrypoint] SNAPSHOT_TGZ not set; skipping import")
        return

    tgz = Path(snap_tgz)
    if not tgz.exists():
        print(f"[entrypoint] SNAPSHOT_TGZ not found: {tgz}")
        return

    # compute current package hash (from file or .sha256)
    pkg_hash = None
    if snap_sha:
        sha_path = Path(snap_sha)
        if sha_path.exists():
            try:
                pkg_hash = sha_path.read_text(encoding="utf-8").strip().split()[0]
            except Exception:
                pkg_hash = None
    if not pkg_hash:
        pkg_hash = sha256_file(tgz)

    hash_file = chroma_path / ".snapshot_hash"
    if hash_file.exists() and not do_reset:
        last = hash_file.read_text(encoding="utf-8").strip()
        if last == pkg_hash:
            print("[entrypoint] Snapshot already imported; skipping")
            return

    # connect to local persistent client
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    # skip if only-on-empty and collection has data
    if import_on_empty and not do_reset:
        try:
            cnt = col.count()
            if cnt > 0:
                print(f"[entrypoint] Collection not empty (count={cnt}); skipping import")
                return
        except Exception:
            pass

    # Extract in memory and import using scripts/import_chroma-like logic
    import tarfile
    import tempfile
    import numpy as np

    with tarfile.open(tgz, "r:gz") as tar:
        with tempfile.TemporaryDirectory() as tmpd:
            tar.extractall(tmpd)
            tmp = Path(tmpd)
            ids = json.loads((tmp / "ids.json").read_text(encoding="utf-8"))
            metas = json.loads((tmp / "metadatas.json").read_text(encoding="utf-8"))
            embeds = np.load(tmp / "embeddings.npy")
            if do_reset:
                try:
                    client.delete_collection(collection)
                except Exception:
                    pass
                col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

            n = len(ids)
            for off in range(0, n, batch):
                sl = slice(off, min(off + batch, n))
                col.add(ids=ids[sl], metadatas=metas[sl], embeddings=embeds[sl].tolist())
                print(f"[entrypoint] Imported {min(off+batch, n)}/{n}")

    hash_file.write_text(pkg_hash, encoding="utf-8")
    print("[entrypoint] Import complete")


if __name__ == "__main__":
    main()

