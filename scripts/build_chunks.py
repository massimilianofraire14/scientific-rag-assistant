import json
from pathlib import Path

from scientific_rag.chunking.text_chunker import split_text_into_chunks

INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

# Minimum number of characters required to keep a chunk

MIN_CHUNK_CHARS = 300


def main() -> None:
    json_files = sorted(INPUT_DIR.glob("*_pages.json"))

    if not json_files:
        print("No processed page files found.")
        return

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_file:
        for json_file in json_files:
            print(f"Processing {json_file}")

            with json_file.open("r", encoding="utf-8") as f:
                pages = json.load(f)

            for page in pages:
                chunks = split_text_into_chunks(page["text"])

                kept_chunk_id = 0

                for chunk in chunks:
                    chunk = chunk.strip()

                    if len(chunk) < MIN_CHUNK_CHARS:
                        continue

                    record = {
                        "paper_id": page["paper_id"],
                        "page_number": page["page_number"],
                        "chunk_id": kept_chunk_id,
                        "text": chunk,
                    }

                    out_file.write(json.dumps(record) + "\n")
                    kept_chunk_id += 1

    print(f"Saved chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()