import json
from pathlib import Path

import numpy as np

from scientific_rag.embeddings.local_embedder import LocalEmbedder


class SimpleRetriever:
    """Retrieve relevant chunks using cosine similarity over local embeddings."""

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        embedder: LocalEmbedder,
    ) -> None:
        
        self.embeddings = np.load(embeddings_path)
        self.metadata = self._load_metadata(metadata_path)
        self.embedder = embedder

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                "Embeddings and metadata length mismatch."
                )
        
    def _load_metadata(self, metadata_path: Path) -> list[dict]:
        records = []

        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        
        return records
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_texts([query])[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []

        for idx in top_indices:
            record = self.metadata[int(idx)].copy()
            record["score"] = float(scores[idx])
            results.append(record)

        return results

