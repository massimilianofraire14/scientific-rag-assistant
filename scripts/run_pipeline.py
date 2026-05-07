import subprocess
import sys

PIPELINE_COMMANDS = [
    ["uv", "run", "python", "scripts/ingest_papers.py"],
    ["uv", "run", "python", "scripts/build_chunks.py"],
    ["uv", "run", "python", "scripts/build_embeddings.py"],
    ["uv", "run", "python", "scripts/evaluate_retrievers.py"],
]


def run_command(command: list[str]) -> None:
    print("=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    for command in PIPELINE_COMMANDS:
        run_command(command)

    print("=" * 80)
    print("Pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        raise