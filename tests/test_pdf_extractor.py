from scientific_rag.ingestion.pdf_extractor import normalize_text


def test_normalize_text_removes_extra_spaces() -> None:
    raw_text = "This   is\t\tmessy text."
    cleaned_text = normalize_text(raw_text)

    assert cleaned_text == "This is messy text."


def test_normalize_text_reduces_many_newlines() -> None:
    raw_text = "First paragraph.\n\n\n\nSecond paragraph."
    cleaned_text = normalize_text(raw_text)

    assert cleaned_text == "First paragraph.\n\nSecond paragraph."