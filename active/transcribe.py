"""Helpers for preparing, validating, and recording OCR transcription work.

The module keeps workflow metadata and quality evidence separate from transcript
text. Gemini HTTP and request handling live in the provider adapter.
"""

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from active.gemini_api import (
	GeminiClient,
	GeminiContentError,
	GeminiProviderError,
	transcribe_gemini_page,
)
from active.submission import write_submission_checkpoint


# ---------------------------------------------------------------------------
# Section 1: Load the submission context
# ---------------------------------------------------------------------------

def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, object]:
	"""Load a submission artifact and reject malformed JSON or non-object roots."""
	try:
		with path.open("r", encoding="utf-8") as artifact_file:
			artifact = json.load(artifact_file)
	except json.JSONDecodeError as error:
		raise ValueError(f"{artifact_name} contains invalid JSON: {path}") from error

	if not isinstance(artifact, dict):
		raise ValueError(f"{artifact_name} must contain a JSON object: {path}")
	return artifact


def _validate_manifest_pages(pages: object) -> list[dict[str, object]]:
	"""Validate page records and preserve their original, increasing page order."""
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
		if not isinstance(page_status, str) or page_status not in valid_statuses:
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
	"""Load matching manifest and quality sidecar from a submission directory.

	Validate submission identity and manifest page order. Return quality evidence
	separately so it is not mistaken for operational page state.
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
	"""Return accepted and needs-review pages, preserving order and source records."""
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
	*,
	include_transcription: bool,
) -> list[dict[str, object]]:
	"""Collect submission issues and, for revisits, prior transcription issues.

	Translation and final-review findings are not transcription context and are
	excluded. The returned issue dictionaries are not modified.
	"""
	quality_pages = quality_notes.get("pages")
	if not isinstance(quality_pages, list):
		raise ValueError("quality notes pages must be a list")

	issues = []
	context_stages = {"submission"}
	if include_transcription:
		context_stages.add("transcription")
	valid_stages = context_stages | {"transcription", "translation", "final_review"}
	for quality_page in quality_pages:
		if not isinstance(quality_page, dict):
			raise ValueError("each quality-notes page must be an object")
		if quality_page.get("page_number") != page_number:
			continue
		quality_stage = quality_page.get("stage")
		if not isinstance(quality_stage, str) or quality_stage not in valid_stages:
			raise ValueError(f"invalid quality-notes stage: {quality_stage!r}")
		if quality_stage not in context_stages:
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

	Only active page states can be prepared. Resolve relative image paths against
	the submission directory and attach relevant evidence as structured metadata;
	the provider adapter keeps that evidence out of the model request.
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
	if not isinstance(page_status, str) or page_status not in {
		"accepted",
		"needs_review",
	}:
		raise ValueError(
			f"page_status for page {page_number} must be accepted or needs_review"
		)

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
		"scan_page_number": page_number,
		"image_path": str(image_file),
		"page_status": page_status,
		"source_stage": "transcription",
	}
	page_job["quality_context"] = _get_page_quality_context(
		quality_notes,
		page_number,
		include_transcription=revisit,
	)
	return page_job


# ---------------------------------------------------------------------------
# Section 4: Prompt construction
# ---------------------------------------------------------------------------

def build_transcription_prompt(page_job: dict[str, object]) -> str:
	"""Request faithful transcription as ordered text regions within one scan.

	The scan number identifies the image only. Printed folios belong on region
	records and must not be confused with the scan number. Quality evidence and
	other workflow metadata are deliberately excluded from the prompt.
	"""
	if not isinstance(page_job, dict):
		raise ValueError("page_job must be an object")

	page_number = page_job.get("page_number")
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_job[page_number] must be a positive integer scan index")
	scan_page_number = page_job.get("scan_page_number", page_number)
	if (
		not isinstance(scan_page_number, int)
		or isinstance(scan_page_number, bool)
		or scan_page_number != page_number
	):
		raise ValueError("scan_page_number must match the manifest page_number")

	return (
		f"Transcribe all visible text in scan image {scan_page_number} faithfully.\n\n"
		"Treat this as one scanned image, which may show one printed page, a "
		"two-page spread, multiple text blocks, or an unusual layout. Identify "
		"each distinct region containing text separately and list regions in "
		"their natural reading order. Do not assume a fixed number of regions, "
		"that left and right regions are different languages, or that separate "
		"regions contain the same story.\n\n"
		"For every region, return position (a short visual location such as "
		"right, left, upper, lower, or center), printed_page_number (the printed "
		"folio exactly as visible, or null if absent or uncertain), language "
		"(the language if identifiable, otherwise null), and transcription "
		"(the text in its original language and writing system). Preserve "
		"headings, paragraph breaks, spelling, and punctuation. Include captions "
		"and other visible text; do not describe pictures. Mark unreadable text "
		"as [unclear]. Do not translate, summarize, modernize, correct, infer, "
		"or invent.\n\n"
		"Return only valid JSON with this shape: "
		'{"regions":[{"position":"right","printed_page_number":"11",'
		'"language":"Hebrew","transcription":"..."}]}. '
		"Use an empty regions array only when no visible text can be transcribed. "
		"Do not include markdown fences, workflow status, quality notes, or "
		"the scan number in the JSON."
	)


# ---------------------------------------------------------------------------
# Section 5: Gemini API invocation
# ---------------------------------------------------------------------------

def parse_transcription_result(
	response_text: str,
	scan_page_number: int,
) -> dict[str, object]:
	"""Validate Gemini's JSON regions and attach the authoritative scan index.

	The scan index comes from the manifest, not model output. Each text region
	may represent part of a page, a printed page in a spread, or another layout.
	"""
	if not isinstance(scan_page_number, int) or isinstance(scan_page_number, bool) or scan_page_number < 1:
		raise ValueError("scan_page_number must be a positive integer")
	if not isinstance(response_text, str) or not response_text.strip():
		raise ValueError("transcription response must be a non-empty JSON string")

	try:
		response_data = json.loads(response_text)
	except json.JSONDecodeError as error:
		raise ValueError("transcription response must contain valid JSON") from error
	if not isinstance(response_data, dict) or set(response_data) != {"regions"}:
		raise ValueError("transcription JSON must contain only a regions array")

	regions = response_data["regions"]
	if not isinstance(regions, list):
		raise ValueError("transcription regions must be a list")

	required_region_fields = {
		"position",
		"printed_page_number",
		"language",
		"transcription",
	}
	validated_regions = []
	for index, region in enumerate(regions, start=1):
		if not isinstance(region, dict) or set(region) != required_region_fields:
			raise ValueError(
				f"transcription region {index} must contain exactly "
				"position, printed_page_number, language, and transcription"
			)
		position = region["position"]
		if not isinstance(position, str) or not position.strip():
			raise ValueError(f"transcription region {index} position must be non-empty")

		printed_page_number = region["printed_page_number"]
		if printed_page_number is not None and (
			not isinstance(printed_page_number, (str, int))
			or isinstance(printed_page_number, bool)
			or (isinstance(printed_page_number, str) and not printed_page_number.strip())
		):
			raise ValueError(
				f"transcription region {index} printed_page_number must be "
				"a non-empty string, integer, or null"
			)

		language = region["language"]
		if language is not None and (
			not isinstance(language, str) or not language.strip()
		):
			raise ValueError(
				f"transcription region {index} language must be non-empty or null"
			)

		transcription = region["transcription"]
		if not isinstance(transcription, str) or not transcription.strip():
			raise ValueError(
				f"transcription region {index} transcription must be non-empty"
			)

		validated_regions.append({
			"position": position,
			"printed_page_number": printed_page_number,
			"language": language,
			"transcription": transcription,
		})

	return {
		"scan_page_number": scan_page_number,
		"regions": validated_regions,
	}


def transcribe_page_with_gemini(
	page_job: dict[str, object],
	prompt: str,
	*,
	client: GeminiClient | None = None,
	revisit: bool = False,
) -> dict[str, object]:
	"""Transcribe a scan and return validated, ordered regions with its scan index.

	The provider returns model text; this stage parses the structured region
	result and supplies the scan index from the page job. Provider errors are
	propagated unchanged for workflow-level handling.
	"""
	if not isinstance(page_job, dict):
		raise ValueError("page_job must be an object")
	page_number = page_job.get("page_number")
	scan_page_number = page_job.get("scan_page_number", page_number)
	if (
		not isinstance(page_number, int)
		or isinstance(page_number, bool)
		or not isinstance(scan_page_number, int)
		or isinstance(scan_page_number, bool)
		or page_number != scan_page_number
		or page_number < 1
	):
		raise ValueError("page_job scan_page_number must match a positive page_number")

	response_text = transcribe_gemini_page(
		page_job,
		prompt,
		client=client,
		revisit=revisit,
	)
	try:
		return parse_transcription_result(response_text, scan_page_number)
	except ValueError as error:
		raise GeminiContentError(
			f"Gemini returned an invalid structured transcription: {error}"
		) from error


# ---------------------------------------------------------------------------
# Section 6: Validate transcript output
# ---------------------------------------------------------------------------

def inspect_transcription_output(
	transcription_result: Mapping[str, object],
) -> list[dict[str, object]]:
	"""Flag empty results, unclear markers, and adjacent duplicate lines per region.

	This lightweight check reports evidence only; it does not set page status,
	assess image-text accuracy, or compare the narratives across regions.
	"""
	if not isinstance(transcription_result, Mapping):
		raise TypeError("transcription_result must be a mapping")
	scan_page_number = transcription_result.get("scan_page_number")
	if (
		not isinstance(scan_page_number, int)
		or isinstance(scan_page_number, bool)
		or scan_page_number < 1
	):
		raise ValueError("transcription_result scan_page_number must be a positive integer")
	regions = transcription_result.get("regions")
	if not isinstance(regions, list):
		raise ValueError("transcription_result regions must be a list")

	findings = []
	if not regions:
		findings.append({
			"category": "uncertain_or_missing_content",
			"severity": "blocking",
			"confidence": "high",
			"evidence": {"scan_page_number": scan_page_number, "region_count": 0},
			"action": "manual_verify",
			"source_stage": "transcription",
			"note": "No text regions were returned; verify the scan before continuing.",
		})

	unclear_regions = []
	duplicate_regions = []
	for region_index, region in enumerate(regions, start=1):
		if not isinstance(region, Mapping):
			raise ValueError(f"transcription region {region_index} must be an object")
		region_text = region.get("transcription")
		position = region.get("position")
		if not isinstance(region_text, str) or not region_text.strip():
			raise ValueError(
				f"transcription region {region_index} must have non-empty transcription"
			)
		if not isinstance(position, str) or not position.strip():
			raise ValueError(f"transcription region {region_index} must have a position")

		unclear_marker_count = region_text.casefold().count("[unclear]")
		if unclear_marker_count:
			unclear_regions.append({
				"region_index": region_index,
				"position": position,
				"unclear_marker_count": unclear_marker_count,
			})

		duplicate_line_numbers = []
		previous_line = None
		for line_number, line in enumerate(region_text.splitlines(), start=1):
			normalized_line = " ".join(line.split()).casefold()
			if normalized_line and normalized_line == previous_line:
				duplicate_line_numbers.append(line_number)
			previous_line = normalized_line or None

		if duplicate_line_numbers:
			duplicate_regions.append({
				"region_index": region_index,
				"position": position,
				"duplicate_line_numbers": duplicate_line_numbers,
			})

	if unclear_regions:
		findings.append({
			"category": "uncertain_or_missing_content",
			"severity": "warning",
			"confidence": "high",
			"evidence": {"regions": unclear_regions},
			"action": "manual_verify",
			"source_stage": "transcription",
			"note": "One or more regions mark passages as unclear.",
		})

	if duplicate_regions:
		findings.append({
			"category": "repeated_text",
			"severity": "warning",
			"confidence": "medium",
			"evidence": {"regions": duplicate_regions},
			"action": "manual_verify",
			"source_stage": "transcription",
			"note": "Adjacent identical lines within a region should be checked against the scan.",
		})

	return findings


# ---------------------------------------------------------------------------
# Section 7: Save transcription results
# ---------------------------------------------------------------------------

def save_transcription_result(
	submission_directory: str | Path,
	page_number: int,
	transcription_result: Mapping[str, object],
) -> str:
	"""Save a validated region result as UTF-8 JSON without replacing prior attempts.

	The manifest page number is the PDF scan index and must match the result.
	The initial result uses ``page_NNN.json``; revisits are saved as numbered
	attempts so earlier transcriptions remain available.
	"""
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_number must be a positive integer")
	if not isinstance(transcription_result, Mapping):
		raise TypeError("transcription_result must be a mapping")
	if transcription_result.get("scan_page_number") != page_number:
		raise ValueError("transcription result scan_page_number must match page_number")
	regions = transcription_result.get("regions")
	if not isinstance(regions, list):
		raise ValueError("transcription result regions must be a list")
	normalized_result = parse_transcription_result(
		json.dumps({"regions": regions}, ensure_ascii=False),
		page_number,
	)

	transcript_directory = Path(submission_directory) / "transcripts"
	transcript_directory.mkdir(parents=True, exist_ok=True)
	page_stem = f"page_{page_number:03}"
	transcript_path = transcript_directory / f"{page_stem}.json"
	attempt_directory = transcript_directory / page_stem
	attempt_number = 2

	while True:
		try:
			with transcript_path.open("x", encoding="utf-8", newline="\n") as transcript_file:
				json.dump(normalized_result, transcript_file, ensure_ascii=False, indent=2)
				transcript_file.write("\n")
			return str(transcript_path)
		except FileExistsError:
			attempt_directory.mkdir(parents=True, exist_ok=True)
			transcript_path = attempt_directory / f"attempt_{attempt_number:03}.json"
			attempt_number += 1

# ---------------------------------------------------------------------------
# Section 8: Record transcription evidence
# ---------------------------------------------------------------------------

def append_transcription_findings(
	quality_notes: dict[str, object],
	page_number: int,
	findings: list[dict[str, object]],
) -> dict[str, object]:
	"""Return a deep-copied quality sidecar with validated findings appended.

	Existing source and transcription evidence is preserved; an empty finding
	list does not create a new stage record.
	"""
	if not isinstance(quality_notes, dict):
		raise ValueError("quality_notes must be an object")
	if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
		raise ValueError("page_number must be a positive integer")
	if not isinstance(findings, list):
		raise ValueError("findings must be a list")

	valid_severities = {"informational", "warning", "blocking"}
	valid_confidences = {"low", "medium", "high"}
	valid_actions = {"accepted", "review", "retry", "exclude", "manual_verify"}
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
	"""Turn unusable model content into a blocking finding; re-raise other errors.

	Configuration and request failures need caller-level recovery rather than
	being recorded as transcript-quality evidence.
	"""
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
	"""Validate context paths and persist its manifest and quality sidecar together.

	The shared checkpoint writer verifies submission identity and handles the
	paired artifact writes.
	"""
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
