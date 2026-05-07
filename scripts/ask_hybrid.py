import argparse
from pathlib import Path

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.retrieval.hybrid_retriever import HybridRetriever

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid dense + BM25 retrieval.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()

    embedder = LocalEmbedder()

    retriever = HybridRetriever(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedder=embedder,
        alpha=args.alpha,
    )

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Paper: {result['paper_id']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Hybrid score: {result['score']:.4f}")
        print(f"Dense score norm: {result['dense_score']:.4f}")
        print(f"BM25 score norm: {result['bm25_score']:.4f}")
        print("-" * 80)
        print(result["text"][:1000])
        print()


if __name__ == "__main__":
    main()