import pytest

from scientific_rag.chunking.text_chunker import split_text_into_chunks


def test_split_text_into_chunks_does_not_cut_words() -> None:
    text = "alpha beta gamma delta epsilon"

    chunks = split_text_into_chunks(text, chunk_size=18, overlap=0)

    assert chunks == [
        "alpha beta gamma",
        "delta epsilon",
    ]


def test_overlap_property() -> None:
    text = "alpha beta gamma delta epsilon"

    chunks = split_text_into_chunks(text, chunk_size=15, overlap=6)

    for i in range(len(chunks) - 1):
        assert chunks[i].split()[-1] == chunks[i + 1].split()[0]


def test_split_text_into_chunks_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text_into_chunks("some text", chunk_size=100, overlap=100)