import argparse
from pathlib import Path

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.retrieval.simple_retriever import SimpleRetriever

DEFAULT_EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
DEFAULT_METADATA_PATH = Path("data/processed/metadata.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question to the scientific RAG retrieval system."
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to ask.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )

    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to the embeddings .npy file.",
    )

    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the metadata .jsonl file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedder = LocalEmbedder()

    retriever = SimpleRetriever(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        embedder=embedder,
    )

    results = retriever.retrieve(args.question, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {rank}")
        print(f"Paper: {result['paper_id']}")
        print(f"Page: {result['page_number']}")

        if "chunk_id" in result:
            print(f"Chunk: {result['chunk_id']}")

        if "score" in result:
            print(f"Score: {result['score']:.4f}")

        print("-" * 80)
        print(result["text"])
        print()

    if not results:
        print("No results found.")


if __name__ == "__main__":
    main()