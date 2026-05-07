from scientific_rag.evaluation.retrieval_metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_returns_true_when_expected_page_is_retrieved() -> None:
    results = [
        {"paper_id": "neural_odes", "page_number": 1},
        {"paper_id": "PINN_Raissi", "page_number": 3},
    ]

    assert hit_at_k(
        retrieved_results=results,
        expected_paper_id="PINN_Raissi",
        expected_page_numbers=[3],
    )


def test_hit_at_k_returns_false_when_expected_page_is_missing() -> None:
    results = [
        {"paper_id": "neural_odes", "page_number": 1},
        {"paper_id": "PINN_Raissi", "page_number": 5},
    ]

    assert not hit_at_k(
        retrieved_results=results,
        expected_paper_id="PINN_Raissi",
        expected_page_numbers=[3],
    )


def test_reciprocal_rank_returns_inverse_rank_of_first_match() -> None:
    results = [
        {"paper_id": "neural_odes", "page_number": 1},
        {"paper_id": "PINN_Raissi", "page_number": 3},
        {"paper_id": "PINN_Raissi", "page_number": 5},
    ]

    rr = reciprocal_rank(
        retrieved_results=results,
        expected_paper_id="PINN_Raissi",
        expected_page_numbers=[3],
    )

    assert rr == 0.5


def test_reciprocal_rank_returns_zero_when_no_match_exists() -> None:
    results = [
        {"paper_id": "neural_odes", "page_number": 1},
        {"paper_id": "PINN_Raissi", "page_number": 5},
    ]

    rr = reciprocal_rank(
        retrieved_results=results,
        expected_paper_id="PINN_Raissi",
        expected_page_numbers=[3],
    )

    assert rr == 0.0


def test_recall_at_k_computes_fraction_of_hits() -> None:
    hits = [True, False, True, True]

    assert recall_at_k(hits) == 0.75


def test_mean_reciprocal_rank_computes_average_rr() -> None:
    reciprocal_ranks = [1.0, 0.5, 0.0]

    assert mean_reciprocal_rank(reciprocal_ranks) == 0.5