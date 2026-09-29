"""Function contracts for submission intake and baseline page evaluation.

The submission stage validates one source PDF, renders its pages, evaluates
each page, and saves the first workflow checkpoint. The manifest records
operational state; quality notes preserve the evidence behind decisions.
"""

import copy
import json
import os
import tempfile


# 1. Source PDF intake

def normalize_submission_path(raw_path: str) -> str:
    """Return a filesystem path with surrounding whitespace and quotes removed.

    Preconditions:
        - raw_path is a string supplied by the caller.

    Postconditions:
        - The returned value is suitable for filesystem validation.
        - This function does not check that the path exists or open the file.
    """
    if not isinstance(raw_path, str):
        raise TypeError("Submission path must be a string")

    return raw_path.strip().strip("\"'").strip()


def validate_submission_pdf(path: str) -> str:
    """Validate and return the path to one readable, existing PDF file.

    Raises:
        ValueError: The path is missing, is not a file, or is not a PDF.

    Postconditions:
        - The returned path identifies the validated source PDF.
        - No page images or pipeline records have been created.
    """
    normalized_path = normalize_submission_path(path)

    if not os.path.isfile(normalized_path):
        raise ValueError("Invalid file path")

    _, extension = os.path.splitext(normalized_path)
    if extension.lower() != ".pdf":
        raise ValueError("File format not accepted; expected a PDF")

    try:
        with open(normalized_path, "rb") as source_file:
            header = source_file.read(1024)
    except OSError as error:
        raise ValueError("PDF file cannot be read") from error

    if b"%PDF-" not in header:
        raise ValueError("File does not contain a valid PDF header")

    return normalized_path


# 2. Page rendering

def render_pdf_pages(source_pdf: str, image_directory: str) -> list[dict[str, object]]:
    """Render source pages and return one ordered record for each PDF page.

    Successfully rendered records contain:
        - page_number: the original, one-based PDF page number
        - image_path: the path to that page's rendered image

    A page that cannot be rendered contains page_number, image_path set to
    None, and render_error with the failure detail.

    Preconditions:
        - source_pdf has passed validate_submission_pdf.
        - image_directory is reserved for this submission's rendered pages.

    Postconditions:
        - Every source page has exactly one record in original order.
        - A failed page is not omitted or represented as a successful image.

    Raises:
        ValueError: The PDF contains no pages.
        OSError: The PDF cannot be opened or the image directory cannot be created.
    """
    try:
        import pymupdf
    except ImportError as error:
        raise ImportError(
            "PyMuPDF is required to render PDF pages. Install it with: pip install pymupdf"
        ) from error

    validated_pdf = validate_submission_pdf(source_pdf)
    os.makedirs(image_directory, exist_ok=True)

    try:
        document = pymupdf.open(validated_pdf)
    except Exception as error:
        raise OSError(f"Could not open PDF: {validated_pdf}") from error

    try:
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("PDF contains no pages")

        page_records = []
        for page_index in range(page_count):
            page_number = page_index + 1
            image_path = os.path.join(
                image_directory,
                f"page_{page_number:03d}.png",
            )

            try:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pixmap.save(image_path)
            except Exception as error:
                page_records.append({
                    "page_number": page_number,
                    "image_path": None,
                    "render_error": str(error) or type(error).__name__,
                })
                continue

            page_records.append({
                "page_number": page_number,
                "image_path": image_path,
            })

        return page_records
    finally:
        document.close()


# 3. Page quality findings and state decisions

_BLANK_PAGE_THRESHOLD = 0.999
_MIN_PAGE_WIDTH = 512
_MIN_PAGE_HEIGHT = 512
_QUALITY_SAMPLE_SIZE = 64


def evaluate_page_quality(
    page_number: int,
    image_path: str,
) -> list[dict[str, object]]:
    """Return zero or more structured quality issues found on one page.

    Each issue contains category, severity, confidence, evidence, action,
    source_stage, and a concise human-readable note. Initial findings use
    source_stage "submission". A clean page returns an empty list.

    This function records findings only. It does not overwrite or replace the
    rendered source image, prompt the user, or decide the page's final state.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            with image.convert("L") as sample:
                sample.thumbnail((_QUALITY_SAMPLE_SIZE, _QUALITY_SAMPLE_SIZE))
                pixels = sample.tobytes()

        if not pixels:
            raise ValueError("Image contains no pixels")
    except (OSError, ValueError) as error:
        return [{
            "category": "unreadable_image",
            "severity": "blocking",
            "confidence": "high",
            "evidence": {"error": str(error)},
            "action": "exclude",
            "source_stage": "submission",
            "note": "The rendered page image could not be read or evaluated.",
        }]

    issues = []
    white_ratio = sum(pixel > 250 for pixel in pixels) / len(pixels)
    if white_ratio >= _BLANK_PAGE_THRESHOLD:
        issues.append({
            "category": "blank_page",
            "severity": "blocking",
            "confidence": "high",
            "evidence": {
                "blank_ratio": round(white_ratio, 4),
                "threshold": _BLANK_PAGE_THRESHOLD,
            },
            "action": "skipped",
            "source_stage": "submission",
            "note": "The page appears blank and is not useful for transcription.",
        })

    if width < _MIN_PAGE_WIDTH or height < _MIN_PAGE_HEIGHT:
        issues.append({
            "category": "low_resolution",
            "severity": "warning",
            "confidence": "high",
            "evidence": {
                "width": width,
                "height": height,
                "minimum_width": _MIN_PAGE_WIDTH,
                "minimum_height": _MIN_PAGE_HEIGHT,
            },
            "action": "review",
            "source_stage": "submission",
            "note": "The page image is below the minimum resolution threshold.",
        })

    return issues


def decide_initial_page_state(
    issues: list[dict[str, object]],
) -> tuple[str, str | None]:
    """Choose the page's initial workflow state from its quality findings.

    Returns:
        A pair of (page_status, skip_reason). page_status is one of accepted,
        needs_review, rejected, or skipped. skip_reason is present when the
        page is skipped and otherwise is None.

    Policy:
        - Usable pages without meaningful risk are accepted.
        - Usable but risky pages may be marked needs_review.
        - Unusable pages are rejected or skipped according to their findings.
        - A skipped page remains in the manifest with its original number.
        - Quality evidence remains in quality notes; state is not a substitute
          for the issue records.
    """
    categories = {issue.get("category") for issue in issues}
    if "blank_page" in categories:
        return "skipped", "blank_page"

    if any(
        issue.get("severity") == "blocking"
        or issue.get("action") in {"exclude", "manual_verify"}
        for issue in issues
    ):
        return "rejected", None

    if any(
        issue.get("severity") == "warning"
        or issue.get("action") in {"review", "retry"}
        for issue in issues
    ):
        return "needs_review", None

    return "accepted", None


def evaluate_submission_pages(
    rendered_pages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all rendered pages and return page records plus quality notes.

    For each rendered page, collect its quality issues, choose exactly one
    initial page state, and retain the original page number and image path.

    Returns:
        A pair of (manifest_page_records, quality_page_records).

    Manifest page records contain page_number, image_path, and page_status;
    skipped records also contain skip_reason. Quality page records contain
    page_number, stage, and the page's issue list. Pages with no issues may
    have an empty issue list so the record remains page-complete.
    """
    manifest_pages = []
    quality_pages = []

    for rendered_page in rendered_pages:
        page_number = rendered_page["page_number"]
        image_path = rendered_page.get("image_path")
        if "render_error" in rendered_page:
            issues = [{
                "category": "rendering_failure",
                "severity": "blocking",
                "confidence": "high",
                "evidence": {"error": rendered_page["render_error"]},
                "action": "exclude",
                "source_stage": "submission",
                "note": "The source PDF page could not be rendered to an image.",
            }]
        else:
            issues = evaluate_page_quality(page_number, image_path)
        page_status, skip_reason = decide_initial_page_state(issues)

        manifest_page = {
            "page_number": page_number,
            "image_path": image_path,
            "page_status": page_status,
        }
        if skip_reason is not None:
            manifest_page["skip_reason"] = skip_reason
        manifest_pages.append(manifest_page)

        quality_pages.append({
            "page_number": page_number,
            "stage": "submission",
            "issues": issues,
        })

    return manifest_pages, quality_pages


# 4. Submission manifest and quality-notes sidecar

def create_submission_manifest(
    submission_id: str,
    source_pdf: str,
    created_at: str,
    pages: list[dict[str, object]],
) -> dict[str, object]:
    """Build the operational record for one submission.

    The manifest includes submission_id, source_pdf, created_at, overall
    status, and the complete ordered page list. Each page has one page_status;
    skipped pages retain their original page_number and skip_reason. Keep
    detailed evidence in quality_notes rather than duplicating it here.
    """
    if not submission_id or not isinstance(submission_id, str):
        raise ValueError("submission_id must be a non-empty string")
    if not isinstance(source_pdf, str) or not source_pdf:
        raise ValueError("source_pdf must be a non-empty string")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("created_at must be a non-empty string")
    if not isinstance(pages, list):
        raise ValueError("pages must be a list")

    valid_statuses = {"accepted", "needs_review", "rejected", "skipped"}
    manifest_pages = []
    previous_page_number = 0

    for page in pages:
        page_number = page.get("page_number")
        image_path = page.get("image_path")
        page_status = page.get("page_status")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number <= previous_page_number
        ):
            raise ValueError("pages must have unique, increasing positive page numbers")
        if page_status not in valid_statuses:
            raise ValueError(f"invalid page_status: {page_status!r}")
        if image_path is None:
            if page_status != "rejected":
                raise ValueError("only rejected pages may have no image_path")
        elif not isinstance(image_path, str) or not image_path:
            raise ValueError("image_path must be a non-empty string when present")

        manifest_page = {
            "page_number": page_number,
            "image_path": image_path,
            "page_status": page_status,
        }
        if page_status == "skipped":
            skip_reason = page.get("skip_reason")
            if not isinstance(skip_reason, str) or not skip_reason:
                raise ValueError("skipped pages must include a non-empty skip_reason")
            manifest_page["skip_reason"] = skip_reason
        elif "skip_reason" in page:
            raise ValueError("only skipped pages may include a skip_reason")

        manifest_pages.append(manifest_page)
        previous_page_number = page_number

    return {
        "submission_id": submission_id,
        "source_pdf": source_pdf,
        "created_at": created_at,
        "status": "in_progress",
        "pages": manifest_pages,
    }


def create_quality_notes(
    submission_id: str,
    generated_at: str,
    page_findings: list[dict[str, object]],
) -> dict[str, object]:
    """Build the evidence sidecar for source-stage quality findings.

    The document includes submission_id, generated_at, and page-level records.
    Each issue preserves category, severity, confidence, evidence, action,
    source_stage, and note. Findings from later stages can be appended without
    deleting earlier evidence.
    """
    if not submission_id or not isinstance(submission_id, str):
        raise ValueError("submission_id must be a non-empty string")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at must be a non-empty string")
    if not isinstance(page_findings, list):
        raise ValueError("page_findings must be a list")

    valid_stages = {"submission", "transcription", "translation", "final_review"}
    valid_severities = {"informational", "warning", "blocking"}
    valid_confidences = {"low", "medium", "high"}
    valid_actions = {"accepted", "review", "retry", "exclude", "manual_verify", "skipped"}
    required_issue_fields = {
        "category",
        "severity",
        "confidence",
        "evidence",
        "action",
        "source_stage",
        "note",
    }
    validated_pages = []

    for page in page_findings:
        page_number = page.get("page_number")
        stage = page.get("stage")
        issues = page.get("issues")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number <= 0
        ):
            raise ValueError("each finding page must have a positive page_number")
        if stage not in valid_stages:
            raise ValueError(f"invalid quality stage: {stage!r}")
        if not isinstance(issues, list):
            raise ValueError("each finding page must have an issues list")

        for issue in issues:
            if not isinstance(issue, dict) or not required_issue_fields.issubset(issue):
                raise ValueError("each issue must include all required quality fields")
            if not isinstance(issue["category"], str) or not issue["category"]:
                raise ValueError("each issue must have a non-empty category")
            if issue["severity"] not in valid_severities:
                raise ValueError(f"invalid issue severity: {issue['severity']!r}")
            if issue["confidence"] not in valid_confidences:
                raise ValueError(f"invalid issue confidence: {issue['confidence']!r}")
            if issue["action"] not in valid_actions:
                raise ValueError(f"invalid issue action: {issue['action']!r}")
            if issue["source_stage"] not in valid_stages:
                raise ValueError(f"invalid issue source_stage: {issue['source_stage']!r}")
            if issue["source_stage"] != stage:
                raise ValueError("issue source_stage must match its page stage")
            if not isinstance(issue["note"], str):
                raise ValueError("each issue note must be a string")

        validated_pages.append({
            "page_number": page_number,
            "stage": stage,
            "issues": issues,
        })

    return {
        "submission_id": submission_id,
        "generated_at": generated_at,
        "pages": copy.deepcopy(validated_pages),
    }


def write_submission_checkpoint(
    output_directory: str,
    manifest: dict[str, object],
    quality_notes: dict[str, object],
) -> tuple[str, str]:
    """Persist the manifest and quality-notes sidecar before LLM processing.

    Returns:
        A pair of (manifest_path, quality_notes_path).

    Postconditions:
        - The two JSON files are saved in the submission's output directory.
        - The manifest and sidecar refer to the same submission_id.
        - No transcription stage may begin until both files are written.

    Raises:
        OSError: Either checkpoint file cannot be written.
    """
    manifest_id = manifest.get("submission_id")
    quality_notes_id = quality_notes.get("submission_id")
    if not manifest_id or manifest_id != quality_notes_id:
        raise ValueError("manifest and quality notes must have the same submission_id")

    os.makedirs(output_directory, exist_ok=True)
    manifest_path = os.path.join(output_directory, "submission_manifest.json")
    quality_notes_path = os.path.join(output_directory, "quality_notes.json")
    pending_files = []

    try:
        for data in (manifest, quality_notes):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_directory,
                prefix=".checkpoint-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                pending_files.append(temporary_file.name)
                json.dump(data, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")

        os.replace(pending_files[0], manifest_path)
        pending_files.pop(0)
        os.replace(pending_files[0], quality_notes_path)
        pending_files.pop(0)
    finally:
        for temporary_path in pending_files:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass

    return manifest_path, quality_notes_path


# 5. Intake workflow entry point

def create_submission(
    source_path: str,
    submission_id: str,
    output_directory: str,
    created_at: str,
) -> dict[str, object]:
    """Run intake through the first saved checkpoint for a single PDF.

    Workflow:
        1. Normalize and validate source_path.
        2. Render pages into this submission's image directory.
        3. Evaluate each page and assign its initial workflow state.
        4. Build the manifest and quality-notes sidecar.
        5. Persist both files before any LLM stage can run.

    Returns a result containing the manifest, quality_notes, manifest_path,
    and quality_notes_path. The manifest retains all original page numbers,
    including pages marked skipped. If the valid PDF has no usable pages,
    persist the findings and return a manifest with status "rejected".

    The created_at value is also used as generated_at for the initial quality
    notes checkpoint.

    Raises:
        ValueError: The source is invalid or the PDF contains no pages.
        OSError: The PDF cannot be opened, the image directory cannot be
            created, or checkpoint persistence fails. Individual page-render
            failures are saved as rejected page findings instead.
    """
    validated_source = validate_submission_pdf(source_path)
    source_pdf = os.path.abspath(validated_source)
    submission_directory = os.path.abspath(output_directory)
    image_directory = os.path.join(submission_directory, "page_images")

    rendered_pages = render_pdf_pages(source_pdf, image_directory)
    manifest_pages, quality_page_findings = evaluate_submission_pages(rendered_pages)

    manifest = create_submission_manifest(
        submission_id,
        source_pdf,
        created_at,
        manifest_pages,
    )
    usable_statuses = {"accepted", "needs_review"}
    if not any(page["page_status"] in usable_statuses for page in manifest_pages):
        manifest["status"] = "rejected"

    quality_notes = create_quality_notes(
        submission_id,
        created_at,
        quality_page_findings,
    )
    manifest_path, quality_notes_path = write_submission_checkpoint(
        submission_directory,
        manifest,
        quality_notes,
    )

    return {
        "manifest": manifest,
        "quality_notes": quality_notes,
        "manifest_path": manifest_path,
        "quality_notes_path": quality_notes_path,
    }