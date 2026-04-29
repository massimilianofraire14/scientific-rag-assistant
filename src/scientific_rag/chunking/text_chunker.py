from typing import List


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks without cutting words.

    Strategy:
    - Split text into words
    - Build chunks by accumulating words up to chunk_size
    - When a chunk is full, keep a portion of its ending (overlap)
      and reuse it at the beginning of the next chunk

    Parameters:
    - text: input text to split
    - chunk_size: approximate max number of characters per chunk
    - overlap: approximate number of overlapping characters between chunks

    Returns:
    - List of text chunks (strings)
    """

    # Safety check: overlap must be smaller than chunk size
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split text into words (prevents breaking words mid-way)
    words = text.split()

    # Final list of chunks
    chunks: List[str] = []

    # Current chunk (list of words) and its approximate length
    current_chunk: list[str] = []
    current_length = 0

    # Iterate over all words
    for word in words:
        # Approximate length of word (+1 for space)
        word_length = len(word) + 1

        # If adding this word exceeds chunk size → finalize current chunk
        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk as a string
            chunks.append(" ".join(current_chunk))

            # Build overlap: take last words from current chunk
            overlap_words: list[str] = []
            overlap_length = 0

            # Iterate backwards to collect overlap from the end
            for previous_word in reversed(current_chunk):
                previous_word_length = len(previous_word) + 1

                # Stop when overlap size is reached
                if overlap_length + previous_word_length > overlap:
                    break

                # Insert at beginning to preserve original order
                overlap_words.insert(0, previous_word)
                overlap_length += previous_word_length

            # Start new chunk with overlap words
            current_chunk = overlap_words
            current_length = overlap_length

        # Add current word to the chunk
        current_chunk.append(word)
        current_length += word_length

    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks