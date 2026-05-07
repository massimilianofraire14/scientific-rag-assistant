from scientific_rag.generation.grounded_generator import (
    format_citation,
    generate_grounded_answer,
)


def test_format_citation() -> None:
    result = {
        "paper_id": "PINN_Raissi",
        "page_number": 3,
    }

    citation = format_citation(result)

    assert citation == "PINN_Raissi, page 3"


def test_generate_grounded_answer_with_context() -> None:
    query = "How do PINNs use automatic differentiation?"
    chunks = [
        {
            "paper_id": "PINN_Raissi",
            "page_number": 3,
            "text": (
                "PINNs use automatic differentiation to compute derivatives"
                "with respect to input coordinates."
            ),
        }
    ]

    answer = generate_grounded_answer(query, chunks)

    assert "How do PINNs use automatic differentiation?" in answer
    assert "PINN_Raissi, page 3" in answer
    assert "automatic differentiation" in answer


def test_generate_grounded_answer_without_context() -> None:
    answer = generate_grounded_answer("Unknown question", [])

    assert "no relevant information" in answer.lower()