"""Transcription workflow stage for OCR processing.

This file is intended for the transcription-stage workflow:
- load the manifest and active pages
- filter pages by workflow state
- prepare the page-level OCR job
- call the provider adapter
- validate output and update evidence

The provider-specific HTTP and request logic lives in the Gemini API module.
"""

import copy
import json
from pathlib import Path

from PIL import Image

from active.gemini_api import GeminiClient, GeminiProviderError, transcribe_gemini_page
from active.submission import write_submission_checkpoint


# ---------------------------------------------------------------------------
# Section 1: Load the submission context
# ---------------------------------------------------------------------------

def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, object]:
	"""Read a JSON object from a submission artifact path."""
	try:
		with path.open("r", encoding="utf-8") as artifact_file:
			artifact = json.load(artifact_file)
	except json.JSONDecodeError as error:
		raise ValueError(f"{artifact_name} contains invalid JSON: {path}") from error

	if not isinstance(artifact, dict):
		raise ValueError(f"{artifact_name} must contain a JSON object: {path}")
	return artifact


def _validate_manifest_pages(pages: object) -> list[dict[str, object]]:
	"""Validate and return the complete, ordered manifest page list."""
	if not isinstance(pages, list):
		raise ValueError("submission manifest pages must be a list")

	valid_statuses = {"accepted", "needs_review", "rejected", "skipped"}
	previous_page_number = 0
	for page in pages:
		if not isinstance(page, dict):
			raise ValueError("each manifest page must be an object")

		page_number = page.get("page_number")
		if (
			not isinstance(page_number, int)
			or isinstance(page_number, bool)
			or page_number <= previous_page_number
		):
			raise ValueError("manifest pages must have unique, increasing positive page numbers")

		page_status = page.get("page_status")
		if page_status not in valid_statuses:
			raise ValueError(f"invalid page_status for page {page_number}: {page_status!r}")

		image_path = page.get("image_path")
		if image_path is None:
			if page_status != "rejected":
				raise ValueError("only rejected pages may have no image_path")
		elif not isinstance(image_path, str) or not image_path:
			raise ValueError(f"image_path for page {page_number} must be a non-empty string")

		previous_page_number = page_number

	return pages


def load_submission_context(submission_directory: str | Path) -> dict[str, object]:
	"""Load matching operational manifest and quality evidence artifacts.

	The manifest remains the source of page order and workflow state. The
	quality-notes sidecar is returned separately and is never merged into pages.
	"""
	directory = Path(submission_directory)
	manifest_path = directory / "submission_manifest.json"
	quality_notes_path = directory / "quality_notes.json"

	manifest = _read_json_artifact(manifest_path, "submission_manifest.json")
	quality_notes = _read_json_artifact(quality_notes_path, "quality_notes.json")

	submission_id = manifest.get("submission_id")
	if not isinstance(submission_id, str) or not submission_id:
		raise ValueError("submission manifest must contain a non-empty submission_id")
	if quality_notes.get("submission_id") != submission_id:
		raise ValueError("manifest and quality notes must have the same submission_id")
	if not isinstance(quality_notes.get("pages"), list):
		raise ValueError("quality notes pages must be a list")

	manifest["pages"] = _validate_manifest_pages(manifest.get("pages"))
	return {
		"manifest": manifest,
		"quality_notes": quality_notes,
		"manifest_path": str(manifest_path),
		"quality_notes_path": str(quality_notes_path),
	}


# ---------------------------------------------------------------------------
# Section 2: Filter active pages
# ---------------------------------------------------------------------------

def select_active_pages(pages: list[dict[str, object]]) -> list[dict[str, object]]:
	"""Return active pages in manifest order without changing their records."""
	if not isinstance(pages, list):
		raise ValueError("pages must be a list")

	active_statuses = {"accepted", "needs_review"}
	known_statuses = active_statuses | {"rejected", "skipped"}
	active_pages = []
	for page in pages:
		if not isinstance(page, dict):
			raise ValueError("each manifest page must be an object")

		page_status = page.get("page_status")
		if not isinstance(page_status, str) or page_status not in known_statuses:
			raise ValueError(f"invalid page_status: {page_status!r}")
		if page_status in active_statuses:
			active_pages.append(page)

	return active_pages


# ---------------------------------------------------------------------------
# Section 3: Page-level preparation
# ---------------------------------------------------------------------------

def _get_page_quality_context(
	quality_notes: dict[str, object],
	page_number: int,
) -> list[dict[str, object]]:
	"""Return unchanged quality issue records associated with one page."""
	quality_pages = quality_notes.get("pages")
	if not isinstance(quality_pages, list):
		raise ValueError("quality notes pages must be a list")

	issues = []
	transcription_context_stages = {"submission", "transcription"}
	valid_stages = transcription_context_stages | {"translation", "final_review"}
	for quality_page in quality_pages:
		if not isinstance(quality_page, dict):
			raise ValueError("each quality-notes page must be an object")
		if quality_page.get("page_number") != page_number:
			continue
		quality_stage = quality_page.get("stage")
		if not isinstance(quality_stage, str) or quality_stage not in valid_stages:
			raise ValueError(f"invalid quality-notes stage: {quality_stage!r}")
		if quality_stage not in transcription_context_stages:
			continue

		page_issues = quality_page.get("issues")
		if not isinstance(page_issues, list):
			raise ValueError(f"quality issues for page {page_number} must be a list")
		if any(not isinstance(issue, dict) for issue in page_issues):
			raise ValueError(f"quality issues for page {page_number} must be objects")
		issues.extend(page_issues)

	return issues


def prepare_page_job(
	page: dict[str, object],
	quality_notes: dict[str, object],
	submission_directory: str | Path,
	*,
	revisit: bool = False,
) -> dict[str, object]:
	"""Validate a page image and build the structured provider job record.

	Relative image paths are resolved from the submission directory. Quality
	issues are attached only for explicit revisits and remain outside model input.
	"""
	if not isinstance(page, dict):
		raise ValueError("page must be an object")
	if not isinstance(quality_notes, dict):
		raise ValueError("quality_notes must be an object")
	if not isinstance(revisit, bool):
		raise ValueError("revisit must be a boolean")

	page_number = page.get("page_number")
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_number must be a positive integer")

	page_status = page.get("page_status")
	if not isinstance(page_status, str):
		raise ValueError(f"page_status for page {page_number} must be a string")

	image_path = page.get("image_path")
	if not isinstance(image_path, str) or not image_path.strip():
		raise ValueError(f"image_path for page {page_number} must be a non-empty string")

	image_file = Path(image_path)
	if not image_file.is_absolute():
		image_file = Path(submission_directory) / image_file
	image_file = image_file.resolve()
	if not image_file.is_file():
		raise ValueError(f"image for page {page_number} does not exist: {image_file}")

	try:
		with Image.open(image_file) as page_image:
			page_image.verify()
	except (OSError, SyntaxError, ValueError, EOFError) as error:
		raise ValueError(f"image for page {page_number} is not readable: {image_file}") from error

	page_job = {
		"page_number": page_number,
		"image_path": str(image_file),
		"page_status": page_status,
		"source_stage": "transcription",
	}
	if revisit:
		page_job["quality_context"] = _get_page_quality_context(
			quality_notes,
			page_number,
		)
	return page_job


# ---------------------------------------------------------------------------
# Section 4: Prompt construction
# ---------------------------------------------------------------------------

# THIS PROMPT IS TO BE HIGHLY TWEAKED IN THE NEAR FUTURE, AND MAY BE REPLACED BY A MORE SOPHISTICATED PROMPT ENGINEERING SYSTEM.

def build_transcription_prompt(page_job: dict[str, object]) -> str:
	"""Build a page-specific prompt without adding quality evidence."""
	if not isinstance(page_job, dict):
		raise ValueError("page_job must be an object")

	page_number = page_job.get("page_number")
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_job[page_number] must be a positive integer")

	return (
		f"Transcribe the visible text on page {page_number} faithfully.\n\n"
		"Preserve the original language and writing system. Follow the visible "
		"reading order and retain meaningful headings, paragraphs, lists, and "
		"line breaks where possible. Preserve spelling and punctuation as printed.\n\n"
		"Do not translate, summarize, modernize, correct, infer, or invent text. "
		"Mark any text that cannot be read confidently as [unclear].\n\n"
		"Return only the transcription. Do not add an introduction, explanation, "
		"or quality notes."
	)


# ---------------------------------------------------------------------------
# Section 5: Gemini API invocation
# ---------------------------------------------------------------------------

def transcribe_page_with_gemini(
	page_job: dict[str, object],
	prompt: str,
	*,
	client: GeminiClient | None = None,
	revisit: bool = False,
) -> str:
	"""Call the Gemini provider for a prepared page and return transcript text."""
	return transcribe_gemini_page(
		page_job,
		prompt,
		client=client,
		revisit=revisit,
	)


# ---------------------------------------------------------------------------
# Section 6: Validate transcript output
# ---------------------------------------------------------------------------

def inspect_transcription_output(transcript: str) -> list[dict[str, object]]:
	"""Return observable transcript-quality findings without deciding page state."""
	if not isinstance(transcript, str):
		raise TypeError("transcript must be a string")
	if not transcript.strip():
		raise ValueError("transcript must not be empty")

	findings = []
	unclear_marker_count = transcript.casefold().count("[unclear]")
	if unclear_marker_count:
		findings.append({
			"category": "uncertain_or_missing_content",
			"severity": "warning",
			"confidence": "high",
			"evidence": {"unclear_marker_count": unclear_marker_count},
			"action": "manual_verify",
			"source_stage": "transcription",
			"note": "The transcript marks one or more passages as unclear.",
		})

	duplicate_line_numbers = []
	previous_line = None
	for line_number, line in enumerate(transcript.splitlines(), start=1):
		normalized_line = " ".join(line.split()).casefold()
		if normalized_line and normalized_line == previous_line:
			duplicate_line_numbers.append(line_number)
		previous_line = normalized_line or None

	if duplicate_line_numbers:
		findings.append({
			"category": "repeated_text",
			"severity": "warning",
			"confidence": "medium",
			"evidence": {"duplicate_line_numbers": duplicate_line_numbers},
			"action": "manual_verify",
			"source_stage": "transcription",
			"note": "Adjacent identical non-empty lines should be checked against the page.",
		})

	return findings


# ---------------------------------------------------------------------------
# Section 7: Save transcription results
# ---------------------------------------------------------------------------

def save_transcription_result(
	submission_directory: str | Path,
	page_number: int,
	transcript: str,
) -> str:
	"""Save UTF-8 transcript text without overwriting an earlier page result."""
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_number must be a positive integer")
	if not isinstance(transcript, str):
		raise TypeError("transcript must be a string")
	if not transcript.strip():
		raise ValueError("transcript must not be empty")

	transcript_directory = Path(submission_directory) / "transcripts"
	transcript_directory.mkdir(parents=True, exist_ok=True)
	page_stem = f"page_{page_number:03}"
	transcript_path = transcript_directory / f"{page_stem}.txt"
	attempt_directory = transcript_directory / page_stem
	attempt_number = 2

	while True:
		try:
			with transcript_path.open("x", encoding="utf-8", newline="\n") as transcript_file:
				transcript_file.write(transcript)
			return str(transcript_path)
		except FileExistsError:
			attempt_directory.mkdir(parents=True, exist_ok=True)
			transcript_path = attempt_directory / f"attempt_{attempt_number:03}.txt"
			attempt_number += 1

# ---------------------------------------------------------------------------
# Section 8: Record transcription evidence
# ---------------------------------------------------------------------------

def append_transcription_findings(
	quality_notes: dict[str, object],
	page_number: int,
	findings: list[dict[str, object]],
) -> dict[str, object]:
	"""Return quality notes with findings appended to this page's transcription stage."""
	if not isinstance(quality_notes, dict):
		raise ValueError("quality_notes must be an object")
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_number must be a positive integer")
	if not isinstance(findings, list):
		raise ValueError("findings must be a list")

	valid_severities = {"informational", "warning", "blocking"}
	valid_confidences = {"low", "medium", "high"}
	valid_actions = {"accepted", "review", "retry", "exclude", "manual_verify", "skipped"}
	required_fields = {
		"category",
		"severity",
		"confidence",
		"evidence",
		"action",
		"source_stage",
		"note",
	}
	for finding in findings:
		if not isinstance(finding, dict) or not required_fields.issubset(finding):
			raise ValueError("each transcription finding must include all required quality fields")
		if not isinstance(finding["category"], str) or not finding["category"]:
			raise ValueError("each transcription finding must have a non-empty category")
		if not isinstance(finding["severity"], str) or finding["severity"] not in valid_severities:
			raise ValueError(f"invalid issue severity: {finding['severity']!r}")
		if not isinstance(finding["confidence"], str) or finding["confidence"] not in valid_confidences:
			raise ValueError(f"invalid issue confidence: {finding['confidence']!r}")
		if not isinstance(finding["evidence"], dict):
			raise ValueError("each transcription finding must include evidence as an object")
		if not isinstance(finding["action"], str) or finding["action"] not in valid_actions:
			raise ValueError(f"invalid issue action: {finding['action']!r}")
		if finding["source_stage"] != "transcription":
			raise ValueError("transcription finding source_stage must be transcription")
		if not isinstance(finding["note"], str):
			raise ValueError("each transcription finding note must be a string")

	updated_quality_notes = copy.deepcopy(quality_notes)
	quality_pages = updated_quality_notes.get("pages")
	if not isinstance(quality_pages, list):
		raise ValueError("quality notes pages must be a list")

	transcription_page = None
	for quality_page in quality_pages:
		if not isinstance(quality_page, dict):
			raise ValueError("each quality-notes page must be an object")
		if (
			quality_page.get("page_number") == page_number
			and quality_page.get("stage") == "transcription"
		):
			transcription_page = quality_page
			break

	if not findings:
		return updated_quality_notes

	if transcription_page is None:
		transcription_page = {
			"page_number": page_number,
			"stage": "transcription",
			"issues": [],
		}
		quality_pages.append(transcription_page)

	page_issues = transcription_page.get("issues")
	if not isinstance(page_issues, list):
		raise ValueError(f"transcription issues for page {page_number} must be a list")
	page_issues.extend(copy.deepcopy(findings))
	return updated_quality_notes


# ---------------------------------------------------------------------------
# Section 9: Provider failure and verification handoff
# ---------------------------------------------------------------------------

def handle_transcription_provider_error(
	error: GeminiProviderError,
) -> dict[str, object]:
	"""Convert content failures to evidence and propagate other provider errors."""
	if not isinstance(error, GeminiProviderError):
		raise TypeError("error must be a GeminiProviderError")
	if error.category != "content_failure":
		raise error

	return {
		"category": "uncertain_or_missing_content",
		"severity": "blocking",
		"confidence": "high",
		"evidence": {"provider_error": str(error)},
		"action": "manual_verify",
		"source_stage": "transcription",
		"note": "Gemini returned unusable transcript content; verify the page before continuing.",
	}


# ---------------------------------------------------------------------------
# Section 10: End-of-stage checkpoint
# ---------------------------------------------------------------------------

def write_transcription_checkpoint(context: dict[str, object]) -> tuple[str, str]:
	"""Persist the updated manifest and quality notes as one stage checkpoint."""
	if not isinstance(context, dict):
		raise ValueError("context must be an object")

	manifest = context.get("manifest")
	quality_notes = context.get("quality_notes")
	manifest_path = context.get("manifest_path")
	quality_notes_path = context.get("quality_notes_path")
	if not isinstance(manifest, dict):
		raise ValueError("context manifest must be an object")
	if not isinstance(quality_notes, dict):
		raise ValueError("context quality_notes must be an object")
	if not isinstance(manifest_path, str) or not manifest_path:
		raise ValueError("context manifest_path must be a non-empty string")
	if not isinstance(quality_notes_path, str) or not quality_notes_path:
		raise ValueError("context quality_notes_path must be a non-empty string")

	manifest_directory = Path(manifest_path).parent
	quality_notes_directory = Path(quality_notes_path).parent
	if manifest_directory.resolve() != quality_notes_directory.resolve():
		raise ValueError("manifest and quality notes must share a submission directory")

	return write_submission_checkpoint(str(manifest_directory), manifest, quality_notes)
