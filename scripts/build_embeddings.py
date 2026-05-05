import json
from pathlib import Path

import numpy as np
from scientific_rag.embeddings.local_embedder import LocalEmbedder

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.jsonl")   

def load_chunks(chunks_path: Path)-> tuple[list[str], list[dict]]:
    """Load chunk texts and metadata from a JSONL file."""
    texts = []
    metadata = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            
            texts.append(record["text"])
            metadata.append(
                {
                "paper_id": record["paper_id"],
                "page_number": record["page_number"],
                "chunk_id": record["chunk_id"],
                "text": record["text"]
                }
            )
    return texts, metadata

def save_metadata(metadata: list[dict], output_path: Path) -> None:
    """Save metadata records to a JSONL file."""
    with output_path.open("w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}. Run scripts/build_chunks.py first."
        )
    
    texts, metadata = load_chunks(CHUNKS_PATH)

    print(f"Loaded {len(texts)} chunks. Computing embeddings...")

    embedder = LocalEmbedder()
    embeddings = embedder.embed_texts(texts)

    np.save(EMBEDDINGS_PATH, embeddings)
    save_metadata(metadata, METADATA_PATH)

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
    
    

