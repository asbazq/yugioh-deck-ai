"""
Export a Chroma collection (ids, embeddings, metadatas) to portable files.

Outputs:
- ids.json            : list[str]
- metadatas.json      : list[dict]
- embeddings.npy      : float32 array shape (N, D)

Usage:
  python scripts/export_chroma.py --out ./chroma_snapshot \
    --host 1.2.3.4 --port 8000 --collection yugioh_256

If args are omitted, reads from environment or .env:
  host (default: localhost)
  chroma_port (default: 8000)
  chroma_collection (default: cards)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for snapshot")
    parser.add_argument("--host", help="Chroma HTTP host")
    parser.add_argument("--port", type=int, help="Chroma HTTP port")
    parser.add_argument("--collection", help="Collection name")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size for paging")
    return parser.parse_args()


def getenv(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key, default)
    if v is None:
        return None
    return v.strip('"').strip("'")


def main() -> None:
    dotenv.load_dotenv(".env")
    args = parse_args()

    host = args.host or getenv("host", "localhost")
    port = args.port or int(getenv("chroma_port", "8001"))
    collection_name = args.collection or getenv("chroma_collection", "cards")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.HttpClient(host=host, port=port)
    col = client.get_or_create_collection(collection_name)

    total = col.count()
    print(f"Exporting collection '{collection_name}' from {host}:{port} (count={total})")

    all_ids: List[str] = []
    all_metas: List[dict] = []
    emb_list: List[np.ndarray] = []

    offset = 0
    batch = int(args.batch)
    while offset < total:
        # 'ids' are always returned; include controls optional fields
        got = col.get(include=["metadatas", "embeddings"], limit=batch, offset=offset)
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        embs = got.get("embeddings") or []

        if not ids:
            break

        # Chroma returns nested lists when multiple queries; for get() it's flat
        all_ids.extend(ids)
        all_metas.extend(metas)
        if embs:
            emb_arr = np.asarray(embs, dtype=np.float32)
            emb_list.append(emb_arr)

        offset += len(ids)
        print(f"  fetched {offset}/{total}")

    if not emb_list:
        raise RuntimeError("No embeddings fetched. Ensure server allows returning embeddings and collection is not empty.")

    embeds = np.concatenate(emb_list, axis=0)
    if embeds.shape[0] != len(all_ids):
        raise RuntimeError(f"Count mismatch: embeddings={embeds.shape[0]} ids={len(all_ids)}")

    (out_dir / "ids.json").write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metadatas.json").write_text(json.dumps(all_metas, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "embeddings.npy", embeds)

    print(f"Done. Wrote snapshot to '{out_dir}'.")


if __name__ == "__main__":
    main()
