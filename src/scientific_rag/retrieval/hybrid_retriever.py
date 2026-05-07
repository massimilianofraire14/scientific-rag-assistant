from pathlib import Path

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.retrieval.bm25_retriever import BM25Retriever
from scientific_rag.retrieval.simple_retriever import SimpleRetriever


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to the [0, 1] range."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


class HybridRetriever:
    """Combine dense retrieval and BM25 retrieval using weighted score fusion."""

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        embedder: LocalEmbedder,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.dense_retriever = SimpleRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            embedder=embedder,
        )
        self.bm25_retriever = BM25Retriever(metadata_path=metadata_path)
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[dict]:
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        combined: dict[tuple[str, int, int], dict] = {}

        dense_scores = [result["score"] for result in dense_results]
        bm25_scores = [result["score"] for result in bm25_results]

        dense_scores_norm = min_max_normalize(dense_scores)
        bm25_scores_norm = min_max_normalize(bm25_scores)

        for result, normalized_score in zip(
            dense_results, 
            dense_scores_norm,
            strict=True,
        ):
            key = (
                result["paper_id"],
                result["page_number"],
                result["chunk_id"],
            )

            combined[key] = result.copy()
            combined[key]["dense_score"] = normalized_score
            combined[key]["bm25_score"] = 0.0

        for result, normalized_score in zip(
            bm25_results, 
            bm25_scores_norm,
            strict=True,
        ):
            key = (
                result["paper_id"],
                result["page_number"],
                result["chunk_id"],
            )

            if key not in combined:
                combined[key] = result.copy()
                combined[key]["dense_score"] = 0.0

            combined[key]["bm25_score"] = normalized_score

        for record in combined.values():
            record["score"] = (
                self.alpha * record["dense_score"]
                + (1.0 - self.alpha) * record["bm25_score"]
            )

        ranked_results = sorted(
            combined.values(),
            key=lambda record: record["score"],
            reverse=True,
        )

        return ranked_results[:top_k]