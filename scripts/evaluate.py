import json
from pathlib import Path

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.evaluation.retrieval_metrics import hit_at_k, recall_at_k
from scientific_rag.retrieval.simple_retriever import SimpleRetriever


BENCHMARK_PATH = Path("data/processed/retrieval_benchmark.json")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    top_k = 5

    benchmark = load_benchmark(BENCHMARK_PATH)

    embedder = LocalEmbedder()
    retriever = SimpleRetriever(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedder=embedder,
    )

    hits: list[bool] = []

    for example in benchmark:
        results = retriever.retrieve(example["question"], top_k=top_k)

        hit = hit_at_k(
            retrieved_results=results,
            expected_paper_id=example["expected_paper_id"],
            expected_page_number=example["expected_page_numbers"],
        )

        hits.append(hit)

        status = "HIT" if hit else "MISS"
        print("=" * 80)
        print(f"Question: {example['question']}")
        print(
            f"Expected: {example['expected_paper_id']}, "
            f"pages {example['expected_page_numbers']}"
        )
        print(f"Result: {status}")
        print("Top retrieved pages:")
        for result in results:
            print(
                f"- {result['paper_id']}, page {result['page_number']}, "
                f"score={result['score']:.4f}"
            )

    score = recall_at_k(hits)

    print("=" * 80)
    print(f"Recall@{top_k}: {score:.3f} ({sum(hits)}/{len(hits)})")


if __name__ == "__main__":
    main()