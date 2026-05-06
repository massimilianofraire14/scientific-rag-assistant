def hit_at_k(
    retrieved_results: list[dict],
    expected_paper_id: str,
    expected_page_numbers: list[int],
) -> bool:
    """Return True if any expected paper/page appears in retrieved results."""
    expected_pages = set(expected_page_numbers)

    for result in retrieved_results:
        if (
            result["paper_id"] == expected_paper_id
            and result["page_number"] in expected_pages
        ):
            return True

    return False


def reciprocal_rank(
    retrieved_results: list[dict],
    expected_paper_id: str,
    expected_page_numbers: list[int],
) -> float:
    """Return reciprocal rank of the first relevant result."""
    expected_pages = set(expected_page_numbers)

    for rank, result in enumerate(retrieved_results, start=1):
        if (
            result["paper_id"] == expected_paper_id
            and result["page_number"] in expected_pages
        ):
            return 1.0 / rank

    return 0.0


def recall_at_k(hits: list[bool]) -> float:
    """Compute recall@k over a list of benchmark examples."""
    if not hits:
        return 0.0

    return sum(hits) / len(hits)


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """Compute mean reciprocal rank over benchmark examples."""
    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)