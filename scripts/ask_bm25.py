import argparse
from pathlib import Path

from scientific_rag.retrieval.bm25_retriever import BM25Retriever

METADATA_PATH = Path("data/processed/metadata.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 keyword retrieval.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    retriever = BM25Retriever(metadata_path=METADATA_PATH)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Paper: {result['paper_id']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"BM25 score: {result['score']:.4f}")
        print("-" * 80)
        print(result["text"][:1000])
        print()


if __name__ == "__main__":
    main()