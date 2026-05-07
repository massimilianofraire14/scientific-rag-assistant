

def format_citation(result: dict)-> str:
    """Format citation from retrived chunk metadata. """

    return f"{result['paper_id']}, page {result['page_number']}"

def generate_grounded_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate a simple grounded answer based on retrieved chunks.

    The answer shows the most relevant retrieved evidence without using an LLM.
    """
    
    if not retrieved_chunks:
        return "No relevant information found in the retrieved chunks."
    
    answer_lines = []

    answer_lines.append(f"Query: {query}")
    answer_lines.append("")
    answer_lines.append("Grounded Answer:")
    answer_lines.append(
        "Based on the retrieved chunks, the most relevant evidence is:"
    )

    answer_lines.append("")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        citation = format_citation(chunk)
        text_preview = chunk["text"][:700].strip()

        answer_lines.append(f"{i}. {citation}")
        answer_lines.append(f"{text_preview}")
        answer_lines.append("")

    answer_lines.append(
        "Note: This answer is generated based on the retrieved chunks "" "
        "and does not involve any additional reasoning or synthesis."
    )

    return "\n".join(answer_lines)
