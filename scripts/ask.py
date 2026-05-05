from pathlib import Path
import argparse

from scientific_rag.embeddings.local_embedder import LocalEmbedder
from scientific_rag.generation.grounded_generator import generate_grounded_answer
from scientific_rag.retrieval.simple_retriever import SimpleRetriever


EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over scientific papers.")
    parser.add_argument("query", type=str, help="Question or search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks before the generated answer",
    )

    args = parser.parse_args()

    embedder = LocalEmbedder()

    retriever = SimpleRetriever(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        embedder=embedder,
    )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    if args.show_context:
        for i, result in enumerate(results, start=1):
            print("=" * 80)
            print(f"Retrieved chunk {i}")
            print(f"Paper: {result['paper_id']}")
            print(f"Page: {result['page_number']}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Score: {result['score']:.4f}")
            print("-" * 80)
            print(result["text"][:1200])
            print()

    answer = generate_grounded_answer(args.query, results)

    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()