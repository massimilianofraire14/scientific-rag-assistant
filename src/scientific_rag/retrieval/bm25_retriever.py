import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, and remove simple English stopwords."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


class BM25Retriever:
    """Retrieve chunks using BM25 keyword-based search."""

    def __init__(self, metadata_path: Path) -> None:
        self.metadata = self._load_metadata(metadata_path)
        self.tokenized_corpus = [
            tokenize(record["text"]) for record in self.metadata
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_metadata(self, metadata_path: Path) -> list[dict]:
        records: list[dict] = []

        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        return records

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[::-1][:top_k]

        results: list[dict] = []

        for index in top_indices:
            record = self.metadata[int(index)].copy()
            record["score"] = float(scores[index])
            results.append(record)

        return results