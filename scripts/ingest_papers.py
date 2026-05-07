from pathlib import Path

from scientific_rag.ingestion.pdf_extractor import (
    extract_pdf_pages,
    save_pages_json,
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {RAW_DIR}. Add at least one PDF first.")
        return

    for pdf_path in pdf_paths:
        print(f"Extracting: {pdf_path}")

        pages = extract_pdf_pages(pdf_path)
        output_path = PROCESSED_DIR / f"{pdf_path.stem}_pages.json"

        save_pages_json(pages, output_path)

        print(f"Saved {len(pages)} pages to {output_path}")


if __name__ == "__main__":
    main()