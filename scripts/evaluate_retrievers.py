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

REPORTS_DIR = Path("reports")
RESULTS_PATH = REPORTS_DIR / "retrieval_benchmark_results.json"

TOP_K = 5
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def save_report(
    scores: dict[str, tuple[float, float]],
    best_hybrid_name: str,
    best_hybrid_score: tuple[float, float],
    top_k: int,
    output_path: Path,
) -> None:
    """Save retrieval benchmark results to a JSON report."""
    report = {
        "top_k": top_k,
        "default_retriever": "Dense",
        "best_hybrid": {
            "name": best_hybrid_name,
            "recall_at_k": best_hybrid_score[0],
            "mrr_at_k": best_hybrid_score[1],
        },
        "results": {
            name: {
                "recall_at_k": recall,
                "mrr_at_k": mrr,
            }
            for name, (recall, mrr) in scores.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Saved benchmark report to {output_path}")


def main() -> None:
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

    scores = {
        "Dense": evaluate_retriever(
            name="Dense",
            retriever=dense_retriever,
            benchmark=benchmark,
            top_k=TOP_K,
        ),
        "BM25": evaluate_retriever(
            name="BM25",
            retriever=bm25_retriever,
            benchmark=benchmark,
            top_k=TOP_K,
        ),
    }

    for alpha in ALPHAS:
        hybrid_retriever = HybridRetriever(
            embeddings_path=EMBEDDINGS_PATH,
            metadata_path=METADATA_PATH,
            embedder=embedder,
            alpha=alpha,
        )

        name = f"Hybrid alpha={alpha:.1f}"

        scores[name] = evaluate_retriever(
            name=name,
            retriever=hybrid_retriever,
            benchmark=benchmark,
            top_k=TOP_K,
        )

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for name, (recall, mrr) in scores.items():
        print(f"{name}: Recall@{TOP_K} = {recall:.3f}, MRR@{TOP_K} = {mrr:.3f}")

    hybrid_scores = {
        name: score
        for name, score in scores.items()
        if name.startswith("Hybrid alpha=")
    }

    best_hybrid_name, best_hybrid_score = max(
        hybrid_scores.items(),
        key=lambda item: (item[1][0], item[1][1]),
    )

    best_recall, best_mrr = best_hybrid_score

    print()
    print("=" * 80)
    print("Best hybrid configuration")
    print("=" * 80)
    print(
        f"{best_hybrid_name}: "
        f"Recall@{TOP_K} = {best_recall:.3f}, "
        f"MRR@{TOP_K} = {best_mrr:.3f}"
    )

    save_report(
        scores=scores,
        best_hybrid_name=best_hybrid_name,
        best_hybrid_score=best_hybrid_score,
        top_k=TOP_K,
        output_path=RESULTS_PATH,
    )


if __name__ == "__main__":
    main()