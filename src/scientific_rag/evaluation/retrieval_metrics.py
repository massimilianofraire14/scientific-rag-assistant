def hit_at_k(
    retrieved_results: list[dict],
    expected_paper_id: str,
    expected_page_number: int,
) -> bool:
    """Return True if the expected paper/page appears in retrieved results."""
    for result in retrieved_results:
        if (
            result["paper_id"] == expected_paper_id
            and result["page_number"] == expected_page_number
        ):
            return True

    return False


def recall_at_k(hits: list[bool]) -> float:
    """Compute recall@k over a list of benchmark examples."""
    if not hits:
        return 0.0

    return sum(hits) / len(hits)