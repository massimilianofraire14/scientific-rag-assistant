# Scientific RAG Assistant

A lightweight Retrieval-Augmented Generation-style system for querying scientific papers.

The project parses scientific PDFs, extracts page-level text, chunks the content, computes local dense embeddings, and evaluates dense, BM25, and hybrid retrieval on a manually curated benchmark.

The current focus is on the retrieval layer of a scientific RAG pipeline: reliable document ingestion, chunking, retrieval, evaluation, and command-line querying.

## Features

- PDF ingestion and page-level text extraction
- Text chunking with short-chunk filtering
- Local dense embeddings using sentence-transformers
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid dense/BM25 retrieval with alpha grid search
- Page-level retrieval benchmark
- Recall@k and MRR@k evaluation
- Benchmark results exported to `reports/`
- CLI scripts for querying the corpus
- Unit tests with pytest

## Corpus

The current corpus contains four scientific papers:

- Physics-Informed Neural Networks
- Neural Ordinary Differential Equations
- Attention Is All You Need
- Fourier Neural Operator

The raw PDFs are expected in:

    data/raw/

The processed files are written to:

    data/processed/

## Retrieval Results

Current benchmark results with `top_k = 5`:

| Retriever | Recall@5 | MRR@5 |
|---|---:|---:|
| Dense | 1.000 | 0.892 |
| BM25 | 1.000 | 0.821 |
| Hybrid alpha=0.0 | 1.000 | 0.821 |
| Hybrid alpha=0.1 | 1.000 | 0.846 |
| Hybrid alpha=0.2 | 1.000 | 0.829 |
| Hybrid alpha=0.3 | 1.000 | 0.829 |
| Hybrid alpha=0.4 | 1.000 | 0.829 |
| Hybrid alpha=0.5 | 1.000 | 0.842 |
| Hybrid alpha=0.6 | 1.000 | 0.867 |
| Hybrid alpha=0.7 | 1.000 | 0.867 |
| Hybrid alpha=0.8 | 1.000 | 0.858 |
| Hybrid alpha=0.9 | 1.000 | 0.883 |
| Hybrid alpha=1.0 | 1.000 | 0.892 |

Dense retrieval is used as the default retriever because it obtains the best MRR@5 on the current benchmark.

The benchmark report is saved in:

    reports/retrieval_benchmark_results.json

## Project Structure

    scientific-rag-assistant/
      src/
        scientific_rag/
          chunking/
            text_chunker.py
          embeddings/
            local_embedder.py
          evaluation/
            retrieval_metrics.py
          generation/
            grounded_generator.py
          ingestion/
            pdf_extractor.py
          retrieval/
            bm25_retriever.py
            hybrid_retriever.py
            simple_retriever.py

      scripts/
        ask.py
        ask_bm25.py
        ask_hybrid.py
        ask_simple.py
        build_chunks.py
        build_embeddings.py
        evaluate.py
        evaluate_retrievers.py
        ingest_papers.py
        run_pipeline.py

      data/
        raw/
        processed/

      reports/
        retrieval_benchmark_results.json

      tests/
        test_bm25_retriever.py
        test_grounded_generator.py
        test_pdf_extractor.py
        test_retrieval_metrics.py
        test_simple_retriever.py
        test_text_chunker.py

      pyproject.toml
      uv.lock
      README.md

## Installation

This project uses `uv`.

    uv sync

To install development dependencies:

    uv sync --group dev

## Usage

Run the full pipeline:

    uv run python scripts/run_pipeline.py

Ask a question using the default dense retriever:

    uv run python scripts/ask.py "What are neural ordinary differential equations?"

Ask with a custom number of retrieved chunks:

    uv run python scripts/ask.py "How do PINNs use automatic differentiation?" --top-k 5

Run the BM25 retriever:

    uv run python scripts/ask_bm25.py "What is scaled dot-product attention?"

Run the hybrid retriever:

    uv run python scripts/ask_hybrid.py "How does the Fourier layer work?"

Evaluate the retrievers and run the alpha grid search:

    uv run python scripts/evaluate_retrievers.py

Run tests:

    uv run pytest

Run linting:

    uv run ruff check .

## Benchmark

The retrieval benchmark is stored in:

    data/processed/retrieval_benchmark.json

Each benchmark example has the following structure:

    {
      "question": "What are neural ordinary differential equations?",
      "expected_paper_id": "neural_odes",
      "expected_page_numbers": [1]
    }

Evaluation is page-level: a retrieval is counted as correct if the expected paper and one of the expected pages appear in the top-k retrieved chunks.

The evaluation reports:

- Recall@k
- Mean Reciprocal Rank@k

## Design Decisions

### Dense retrieval as default

Dense retrieval achieves the best MRR@5 on the current benchmark. BM25 and hybrid retrieval are still implemented and evaluated as baselines.

### Page-level evaluation

The ingestion pipeline extracts text from PDFs with page-level metadata. Page-level retrieval is a practical first target for helping users navigate scientific papers.

### Chunk filtering

Very short chunks often contain headers, footers, incomplete sentences, or PDF extraction artifacts. Filtering short chunks helps reduce noisy retrieval results.

### Hybrid retrieval

Hybrid retrieval combines dense and BM25 scores using an alpha parameter. A grid search over alpha values showed that `alpha=1.0` performs best on the current benchmark, which is equivalent to dense retrieval.

## Current Limitations

- Small manually curated benchmark
- Small paper corpus
- Page-level evaluation instead of exact answer-span evaluation
- No reranking step yet
- No web interface yet
- Answer generation is still minimal compared with a full production RAG system

## Future Work

- Add answer generation with stronger citation formatting
- Add reranking
- Add a larger benchmark
- Add exact-span or passage-level evaluation
- Add a Streamlit or FastAPI interface
- Add support for more robust PDF parsing
- Add configurable retriever settings from the command line

## Development Notes

Generated artifacts such as processed data, embeddings, reports, caches, and virtual environments should not be committed unless explicitly needed.

The repository is intended to track source code, tests, configuration, and lightweight placeholder files such as `.gitkeep`.
