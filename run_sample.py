"""Run the submission intake against sample-input/sample.pdf."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from active.submission import create_submission


def main():
    project_directory = Path(__file__).resolve().parent
    source_pdf = project_directory / "sample-input" / "sample.pdf"

    if not source_pdf.is_file():
        raise SystemExit(
            f"Sample PDF not found: {source_pdf}\n"
            "Place your PDF there and name it sample.pdf."
        )

    now = datetime.now(timezone.utc)
    submission_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    output_directory = project_directory / "runs" / submission_id

    start_time = perf_counter()
    try:
        result = create_submission(
            source_path=str(source_pdf),
            submission_id=submission_id,
            output_directory=str(output_directory),
            created_at=now.isoformat(),
        )
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit(f"Submission intake failed: {error}") from error

    elapsed_seconds = perf_counter() - start_time
    print(f"Submission status: {result['manifest']['status']}")
    print(f"Pages recorded: {len(result['manifest']['pages'])}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Quality notes: {result['quality_notes_path']}")
    print(f"Elapsed: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()