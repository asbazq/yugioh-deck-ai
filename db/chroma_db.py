from typing import List
import os

import chromadb
import dotenv


config = dotenv.dotenv_values(".env")


class ChromaDBConnection:
    def __init__(self):
        # Mode: 'http' connects to remote chroma server, 'local' uses embedded persistent client
        mode = os.getenv("chroma_mode", config.get("chroma_mode", "local")).lower()
        collection_name = os.getenv("chroma_collection", config.get("chroma_collection", "cards"))

        self._mode = mode
        if mode == "http":
            host = os.getenv("host", config.get("host", "localhost"))
            port = int(os.getenv("chroma_port", str(config.get("chroma_port", "8000"))))
            self.client = chromadb.HttpClient(host, port)
        else:
            path = os.getenv("chroma_path", config.get("chroma_path", "/chroma"))
            # Use an embedded persistent client so each client server can run locally without external service
            self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def remove_none_values(self, metadata: List[dict]):
        return [self.remove_none_value(data) for data in metadata]

    def remove_none_value(self, metadata: dict):
        for key in metadata:
            if metadata[key] is None:
                metadata[key] = ""
        return metadata

    def update_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.update(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )

    def insert_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.add(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )

    def insert(self, ids, embeds, metadata: List[dict]):
        metadata = self.remove_none_values(metadata)
        return self.collection.add(ids=ids, metadatas=metadata, embeddings=embeds)

    def search_by_embed(self, embed, n_result=1):
        """Query by embedding and return normalized metadatas.
        Always request metadatas, and avoid empty filters. For HTTP servers that
        validate filters strictly, add a harmless where_document.
        """
        if self._mode == "http":
            try:
                result = self.collection.query(
                    query_embeddings=[embed],
                    n_results=n_result,
                    include=["metadatas"],
                    where_document={"$not_contains": "\u0000"},
                )
            except Exception:
                result = self.collection.query(
                    query_embeddings=[embed],
                    n_results=n_result,
                    include=["metadatas"],
                    where={"id": {"$gte": -1}},
                    where_document={"$not_contains": "\u0000"},
                )
        else:
            # Local persistent client usually doesn't require any filter
            result = self.collection.query(
                query_embeddings=[embed],
                n_results=n_result,
                include=["metadatas"],
            )

        raw = result.get("metadatas", []) or []
        normalized = []
        for items in raw:
            norm_items = []
            for m in items or []:
                m = m or {}
                idv = m.get("id") if m.get("id") is not None else m.get("ygopro_id")
                name = m.get("name") or m.get("kor_name") or ""
                # keep original fields but ensure id/name keys exist
                out = {**m}
                out.setdefault("id", idv)
                out.setdefault("name", name)
                norm_items.append(out)
            normalized.append(norm_items)
        return normalized
