from pathlib import Path

from scientific_rag.retrieval.bm25_retriever import BM25Retriever, tokenize


def test_tokenize_removes_stopwords() -> None:
    tokens = tokenize("What is the role of collocation points in PINNs?")

    assert tokens == ["role", "collocation", "points", "pinns"]


def test_bm25_retriever_returns_keyword_match(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"

    metadata_path.write_text(
        '{"paper_id": "paper_a", "page_number": 1, "chunk_id": 0, "text": "automatic differentiation and neural networks"}\n'
        '{"paper_id": "paper_b", "page_number": 2, "chunk_id": 0, "text": "collocation points enforce physical constraints"}\n',
        encoding="utf-8",
    )

    retriever = BM25Retriever(metadata_path=metadata_path)

    results = retriever.retrieve("collocation points", top_k=1)

    assert len(results) == 1
    assert results[0]["paper_id"] == "paper_b"