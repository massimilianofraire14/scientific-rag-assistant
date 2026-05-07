import json
from pathlib import Path

import numpy as np
import pytest

from scientific_rag.retrieval.simple_retriever import SimpleRetriever


class DummyEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "physics": np.array([[1.0, 0.0]]),
            "math": np.array([[0.0, 1.0]]),
        }
        return vectors[texts[0]]


def write_metadata(metadata_path: Path) -> None:
    rows = [
        {
            "paper_id": "paper_a",
            "page_number": 1,
            "chunk_id": 0,
            "text": "physics text",
        },
        {
            "paper_id": "paper_b",
            "page_number": 2,
            "chunk_id": 0,
            "text": "math text",
        },
    ]

    metadata_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_retriever_returns_most_similar_chunk(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.jsonl"

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    np.save(embeddings_path, embeddings)
    write_metadata(metadata_path)

    retriever = SimpleRetriever(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedder=DummyEmbedder(),
    )

    results = retriever.retrieve("physics", top_k=1)

    assert len(results) == 1
    assert results[0]["paper_id"] == "paper_a"
    assert results[0]["score"] == pytest.approx(1.0)


def test_retriever_rejects_mismatched_embeddings_and_metadata(
    tmp_path: Path,
) -> None:
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.jsonl"

    np.save(embeddings_path, np.array([[1.0, 0.0]]))
    write_metadata(metadata_path)

    with pytest.raises(ValueError):
        SimpleRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            embedder=DummyEmbedder(),
        )