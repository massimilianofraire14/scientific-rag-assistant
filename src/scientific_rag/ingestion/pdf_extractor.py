import json
import re
from pathlib import Path

import fitz  # PyMuPDF


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = text.replace("\x00", " ")            #Remove null characters
    text = re.sub(r"[ \t]+", " ", text)         #Collapse multiple spaces/tabs into one
    text = re.sub(r"\n{3,}", "\n\n", text)      #Collapse multiple newlines into two
    text = text.strip()                         #Remove leading/trailing whitespace

    return text

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text page by page from a PDF.

    Return a List of dictionariaries with keys:
    - paper_id
    - file_name
    - page_number
    - text
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Use the file name without extension as paper_id
    paper_id = pdf_path.stem                     
    pages: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            raw_text = page.get_text()
            text = normalize_text(raw_text)
            pages.append({
                "paper_id": paper_id,
                "file_name": pdf_path.name,
                "page_number": page_index + 1,   # Page numbers start at 1
                "text": text
            })

    return pages


def save_pages_json(pages: list[dict], output_path: Path) -> None:
    """Save extracted pages to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)  
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

