import json
from pathlib import Path
from typing import Any

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.evaluation.retrieval_metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from scientific_rag.retrieval.bm25_retriever import BM25Retriever
from scientific_rag.retrieval.hybrid_retriever import HybridRetriever
from scientific_rag.retrieval.simple_retriever import SimpleRetriever


BENCHMARK_PATH = Path("data/processed/retrieval_benchmark.json")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load retrieval benchmark examples from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retriever(
    name: str,
    retriever: Any,
    benchmark: list[dict[str, Any]],
    top_k: int,
) -> tuple[float, float]:
    """
    Evaluate one retriever on the benchmark.

    Returns:
    - recall@k
    - mean reciprocal rank@k
    """
    hits: list[bool] = []
    reciprocal_ranks: list[float] = []

    print("=" * 80)
    print(f"Evaluating: {name}")
    print("=" * 80)

    for example in benchmark:
        results = retriever.retrieve(example["question"], top_k=top_k)

        hit = hit_at_k(
            retrieved_results=results,
            expected_paper_id=example["expected_paper_id"],
            expected_page_numbers=example["expected_page_numbers"],
        )

        rr = reciprocal_rank(
            retrieved_results=results,
            expected_paper_id=example["expected_paper_id"],
            expected_page_numbers=example["expected_page_numbers"],
        )

        hits.append(hit)
        reciprocal_ranks.append(rr)

        status = "HIT" if hit else "MISS"
        retrieved_pages = [result["page_number"] for result in results]

        print(f"{status} | RR={rr:.3f} | {example['question']}")
        print(f"Expected pages: {example['expected_page_numbers']}")
        print(f"Retrieved pages: {retrieved_pages}")
        print()

    recall = recall_at_k(hits)
    mrr = mean_reciprocal_rank(reciprocal_ranks)

    print(f"{name} Recall@{top_k}: {recall:.3f} ({sum(hits)}/{len(hits)})")
    print(f"{name} MRR@{top_k}: {mrr:.3f}")
    print()

    return recall, mrr


def main() -> None:
    top_k = 5

    benchmark = load_benchmark(BENCHMARK_PATH)

    embedder = LocalEmbedder()

    dense_retriever = SimpleRetriever(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedder=embedder,
    )

    bm25_retriever = BM25Retriever(
        metadata_path=METADATA_PATH,
    )

    hybrid_retriever = HybridRetriever(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedder=embedder,
        alpha=0.3,
    )

    scores = {
        "Dense": evaluate_retriever(
            name="Dense",
            retriever=dense_retriever,
            benchmark=benchmark,
            top_k=top_k,
        ),
        "BM25": evaluate_retriever(
            name="BM25",
            retriever=bm25_retriever,
            benchmark=benchmark,
            top_k=top_k,
        ),
        "Hybrid alpha=0.3": evaluate_retriever(
            name="Hybrid alpha=0.3",
            retriever=hybrid_retriever,
            benchmark=benchmark,
            top_k=top_k,
        ),
    }

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for name, (recall, mrr) in scores.items():
        print(f"{name}: Recall@{top_k} = {recall:.3f}, MRR@{top_k} = {mrr:.3f}")


if __name__ == "__main__":
    main()