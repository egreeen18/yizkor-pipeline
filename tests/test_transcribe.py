import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from active.gemini_api import (
    GeminiConfigurationError,
    GeminiContentError,
    GeminiRequestError,
)
from active.transcribe import (
    build_transcription_prompt,
    append_transcription_findings,
    inspect_transcription_output,
    handle_transcription_provider_error,
    load_submission_context,
    prepare_page_job,
    save_transcription_result,
    select_active_pages,
    transcribe_page_with_gemini,
    write_transcription_checkpoint,
)


class LoadSubmissionContextTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)
        self.manifest = {
            "submission_id": "submission-1",
            "source_pdf": "book.pdf",
            "created_at": "2026-10-13T12:00:00Z",
            "status": "in_progress",
            "pages": [
                {
                    "page_number": 4,
                    "image_path": "page_004.png",
                    "page_status": "accepted",
                },
                {
                    "page_number": 5,
                    "image_path": "page_005.png",
                    "page_status": "skipped",
                    "skip_reason": "blank_page",
                },
            ],
        }
        self.quality_notes = {
            "submission_id": "submission-1",
            "generated_at": "2026-10-13T12:00:00Z",
            "pages": [{
                "page_number": 4,
                "stage": "submission",
                "issues": [{"category": "low_contrast", "note": "Dim scan."}],
            }],
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def write_artifacts(self, *, manifest=None, quality_notes=None):
        manifest_path = self.directory / "submission_manifest.json"
        quality_notes_path = self.directory / "quality_notes.json"
        manifest_path.write_text(
            json.dumps(self.manifest if manifest is None else manifest),
            encoding="utf-8",
        )
        quality_notes_path.write_text(
            json.dumps(self.quality_notes if quality_notes is None else quality_notes),
            encoding="utf-8",
        )
        return manifest_path, quality_notes_path

    def test_loads_matching_artifacts_and_preserves_page_order_and_sidecar(self):
        manifest_path, quality_notes_path = self.write_artifacts()

        context = load_submission_context(self.directory)

        self.assertEqual(context["manifest_path"], str(manifest_path))
        self.assertEqual(context["quality_notes_path"], str(quality_notes_path))
        manifest = context["manifest"]
        quality_notes = context["quality_notes"]
        self.assertEqual(
            [page["page_number"] for page in manifest["pages"]],
            [4, 5],
        )
        self.assertEqual(manifest["pages"][1]["page_status"], "skipped")
        self.assertNotIn("issues", manifest["pages"][0])
        self.assertEqual(
            quality_notes["pages"][0]["issues"][0]["category"],
            "low_contrast",
        )

    def test_rejects_mismatched_submission_ids(self):
        quality_notes = {**self.quality_notes, "submission_id": "another-submission"}
        self.write_artifacts(quality_notes=quality_notes)

        with self.assertRaisesRegex(ValueError, "same submission_id"):
            load_submission_context(self.directory)

    def test_rejects_out_of_order_manifest_pages(self):
        manifest = {
            **self.manifest,
            "pages": list(reversed(self.manifest["pages"])),
        }
        self.write_artifacts(manifest=manifest)

        with self.assertRaisesRegex(ValueError, "increasing positive page numbers"):
            load_submission_context(self.directory)

    def test_rejects_invalid_manifest_page_state(self):
        manifest = {
            **self.manifest,
            "pages": [{
                "page_number": 1,
                "image_path": "page_001.png",
                "page_status": "transcribed",
            }],
        }
        self.write_artifacts(manifest=manifest)

        with self.assertRaisesRegex(ValueError, "invalid page_status"):
            load_submission_context(self.directory)

    def test_rejects_non_object_json_artifact(self):
        (self.directory / "submission_manifest.json").write_text("[]", encoding="utf-8")
        (self.directory / "quality_notes.json").write_text(
            json.dumps(self.quality_notes),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_submission_context(self.directory)

    def test_rejects_malformed_json_artifact(self):
        (self.directory / "submission_manifest.json").write_text("{", encoding="utf-8")
        (self.directory / "quality_notes.json").write_text(
            json.dumps(self.quality_notes),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_submission_context(self.directory)


class SelectActivePagesTests(unittest.TestCase):
    def make_page(self, page_number, page_status):
        return {
            "page_number": page_number,
            "image_path": f"page_{page_number:03}.png",
            "page_status": page_status,
        }

    def test_selects_accepted_and_needs_review_in_original_order(self):
        pages = [
            self.make_page(4, "accepted"),
            self.make_page(5, "skipped"),
            self.make_page(6, "needs_review"),
            self.make_page(7, "rejected"),
        ]

        active_pages = select_active_pages(pages)

        self.assertEqual(
            [page["page_number"] for page in active_pages],
            [4, 6],
        )
        self.assertIs(active_pages[0], pages[0])
        self.assertIs(active_pages[1], pages[2])
        self.assertEqual([page["page_number"] for page in pages], [4, 5, 6, 7])

    def test_returns_empty_list_when_no_pages_are_active(self):
        pages = [
            self.make_page(1, "skipped"),
            self.make_page(2, "rejected"),
        ]

        self.assertEqual(select_active_pages(pages), [])

    def test_rejects_non_list_or_malformed_page_records(self):
        with self.assertRaisesRegex(ValueError, "pages must be a list"):
            select_active_pages(None)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "page must be an object"):
            select_active_pages([None])  # type: ignore[list-item]

        with self.assertRaisesRegex(ValueError, "invalid page_status"):
            select_active_pages([self.make_page(1, "transcribed")])


class PreparePageJobTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)
        self.image_directory = self.directory / "page_images"
        self.image_directory.mkdir()
        self.image_path = self.image_directory / "page_011.png"
        Image.new("RGB", (16, 16), "white").save(self.image_path)
        self.page = {
            "page_number": 11,
            "image_path": "page_images/page_011.png",
            "page_status": "needs_review",
        }
        self.quality_notes = {
            "submission_id": "submission-1",
            "pages": [
                {
                    "page_number": 11,
                    "stage": "submission",
                    "issues": [{"category": "low_contrast", "note": "Dim scan."}],
                },
                {
                    "page_number": 11,
                    "stage": "transcription",
                    "issues": [{"category": "repeated_text", "note": "Repeated line."}],
                },
                {
                    "page_number": 12,
                    "stage": "submission",
                    "issues": [{"category": "blur", "note": "Other page."}],
                },
                {
                    "page_number": 11,
                    "stage": "translation",
                    "issues": [{"category": "meaning_drift", "note": "Later-stage issue."}],
                },
            ],
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_builds_page_job_and_matching_revisit_quality_context(self):
        page_job = prepare_page_job(
            self.page,
            self.quality_notes,
            self.directory,
            revisit=True,
        )

        self.assertEqual(page_job["page_number"], 11)
        self.assertEqual(page_job["image_path"], str(self.image_path.resolve()))
        self.assertEqual(page_job["page_status"], "needs_review")
        self.assertEqual(page_job["source_stage"], "transcription")
        self.assertEqual(
            page_job["quality_context"],
            [
                {"category": "low_contrast", "note": "Dim scan."},
                {"category": "repeated_text", "note": "Repeated line."},
            ],
        )

    def test_omits_quality_context_for_non_revisit_job(self):
        page_job = prepare_page_job(self.page, self.quality_notes, self.directory)

        self.assertNotIn("quality_context", page_job)

    def test_rejects_missing_or_unreadable_page_images(self):
        missing_page = {**self.page, "image_path": "page_images/missing.png"}
        with self.assertRaisesRegex(ValueError, "does not exist"):
            prepare_page_job(missing_page, self.quality_notes, self.directory)

        unreadable_path = self.image_directory / "unreadable.png"
        unreadable_path.write_text("not an image", encoding="utf-8")
        unreadable_page = {**self.page, "image_path": str(unreadable_path)}
        with self.assertRaisesRegex(ValueError, "not readable"):
            prepare_page_job(unreadable_page, self.quality_notes, self.directory)


class BuildTranscriptionPromptTests(unittest.TestCase):
    def test_prompt_requests_faithful_transcription_and_unclear_marker(self):
        prompt = build_transcription_prompt({
            "page_number": 11,
            "page_status": "needs_review",
            "quality_context": [{"category": "low_contrast", "note": "Dim scan."}],
        })

        self.assertIn("page 11", prompt)
        self.assertIn("original language and writing system", prompt)
        self.assertIn("reading order", prompt)
        self.assertIn("Do not translate", prompt)
        self.assertIn("Do not", prompt)
        self.assertIn("[unclear]", prompt)
        self.assertIn("Return only the transcription", prompt)
        self.assertNotIn("low_contrast", prompt)
        self.assertNotIn("Dim scan", prompt)

    def test_prompt_rejects_invalid_page_job(self):
        with self.assertRaisesRegex(ValueError, "page_job must be an object"):
            build_transcription_prompt(None)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_transcription_prompt({"page_number": 0})


class TranscribePageWithGeminiTests(unittest.TestCase):
    def test_forwards_page_job_prompt_client_and_revisit_to_provider(self):
        page_job = {
            "page_number": 11,
            "image_path": "page_011.png",
            "page_status": "needs_review",
            "source_stage": "transcription",
        }
        prompt = "Transcribe page 11 faithfully."
        client = object()

        with patch(
            "active.transcribe.transcribe_gemini_page",
            return_value="Transcribed text.",
        ) as provider_call:
            result = transcribe_page_with_gemini(
                page_job,
                prompt,
                client=client,
                revisit=True,
            )

        self.assertEqual(result, "Transcribed text.")
        provider_call.assert_called_once_with(
            page_job,
            prompt,
            client=client,
            revisit=True,
        )

    def test_propagates_categorized_provider_errors_unchanged(self):
        provider_error = GeminiRequestError("provider unavailable")
        with patch(
            "active.transcribe.transcribe_gemini_page",
            side_effect=provider_error,
        ):
            with self.assertRaises(GeminiRequestError) as raised:
                transcribe_page_with_gemini({"page_number": 11}, "prompt")

        self.assertIs(raised.exception, provider_error)


class InspectTranscriptionOutputTests(unittest.TestCase):
    def test_clean_transcript_has_no_quality_findings(self):
        self.assertEqual(
            inspect_transcription_output("First line.\nSecond line."),
            [],
        )

    def test_unclear_markers_create_quality_note_compatible_finding(self):
        findings = inspect_transcription_output("Readable [unclear] passage; [UNCLEAR].")

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "uncertain_or_missing_content")
        self.assertEqual(findings[0]["severity"], "warning")
        self.assertEqual(findings[0]["confidence"], "high")
        self.assertEqual(findings[0]["evidence"], {"unclear_marker_count": 2})
        self.assertEqual(findings[0]["action"], "manual_verify")
        self.assertEqual(findings[0]["source_stage"], "transcription")

    def test_adjacent_duplicate_lines_are_flagged_with_line_numbers(self):
        findings = inspect_transcription_output("Opening line.\nRepeated line.\nRepeated line.")

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "repeated_text")
        self.assertEqual(findings[0]["evidence"], {"duplicate_line_numbers": [3]})
        self.assertEqual(findings[0]["action"], "manual_verify")

    def test_nonadjacent_duplicate_lines_are_not_flagged(self):
        self.assertEqual(
            inspect_transcription_output("Repeated line.\nOther line.\nRepeated line."),
            [],
        )

    def test_rejects_empty_or_non_string_transcript(self):
        with self.assertRaisesRegex(ValueError, "transcript must not be empty"):
            inspect_transcription_output(" \n ")

        with self.assertRaisesRegex(TypeError, "transcript must be a string"):
            inspect_transcription_output(None)  # type: ignore[arg-type]


class SaveTranscriptionResultTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_saves_first_page_transcript_as_utf8_without_modifying_text(self):
        transcript = "שלום\nA faithful transcription.\n"

        saved_path = save_transcription_result(self.directory, 7, transcript)

        expected_path = self.directory / "transcripts" / "page_007.txt"
        self.assertEqual(Path(saved_path), expected_path)
        self.assertEqual(expected_path.read_text(encoding="utf-8"), transcript)

    def test_saves_revisits_as_numbered_attempts_without_overwriting(self):
        initial_path = Path(save_transcription_result(self.directory, 7, "Initial text."))
        second_path = Path(save_transcription_result(self.directory, 7, "Revisit text."))
        third_path = Path(save_transcription_result(self.directory, 7, "Another revisit."))

        attempts_directory = self.directory / "transcripts" / "page_007"
        self.assertEqual(initial_path.read_text(encoding="utf-8"), "Initial text.")
        self.assertEqual(second_path, attempts_directory / "attempt_002.txt")
        self.assertEqual(third_path, attempts_directory / "attempt_003.txt")
        self.assertEqual(second_path.read_text(encoding="utf-8"), "Revisit text.")
        self.assertEqual(third_path.read_text(encoding="utf-8"), "Another revisit.")

    def test_rejects_invalid_page_number_and_blank_transcript(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            save_transcription_result(self.directory, 0, "Text.")

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            save_transcription_result(self.directory, 1, " \n ")


class AppendTranscriptionFindingsTests(unittest.TestCase):
    def setUp(self):
        self.source_issue = {
            "category": "low_contrast",
            "severity": "warning",
            "confidence": "medium",
            "evidence": {"contrast_score": 0.3},
            "action": "review",
            "source_stage": "submission",
            "note": "The scan has low contrast.",
        }
        self.quality_notes = {
            "submission_id": "submission-1",
            "generated_at": "2026-10-13T12:00:00Z",
            "pages": [{
                "page_number": 7,
                "stage": "submission",
                "issues": [self.source_issue],
            }],
        }

    def test_creates_transcription_stage_entry_without_mutating_input(self):
        findings = inspect_transcription_output("Text with [unclear].")
        original_quality_notes = json.loads(json.dumps(self.quality_notes))

        updated = append_transcription_findings(self.quality_notes, 7, findings)

        self.assertEqual(self.quality_notes, original_quality_notes)
        self.assertEqual(updated["pages"][0]["stage"], "submission")
        self.assertEqual(updated["pages"][0]["issues"], [self.source_issue])
        self.assertEqual(updated["pages"][1]["page_number"], 7)
        self.assertEqual(updated["pages"][1]["stage"], "transcription")
        self.assertEqual(updated["pages"][1]["issues"], findings)

    def test_appends_to_existing_transcription_group_and_keeps_prior_findings(self):
        existing_finding = {
            "category": "repeated_text",
            "severity": "warning",
            "confidence": "medium",
            "evidence": {"duplicate_line_numbers": [3]},
            "action": "manual_verify",
            "source_stage": "transcription",
            "note": "A line repeats.",
        }
        quality_notes = {
            **self.quality_notes,
            "pages": [
                *self.quality_notes["pages"],
                {
                    "page_number": 7,
                    "stage": "transcription",
                    "issues": [existing_finding],
                },
            ],
        }
        new_finding = inspect_transcription_output("Text with [unclear].")[0]

        updated = append_transcription_findings(quality_notes, 7, [new_finding])

        self.assertEqual(len(updated["pages"]), 2)
        self.assertEqual(
            updated["pages"][1]["issues"],
            [existing_finding, new_finding],
        )
        self.assertEqual(quality_notes["pages"][1]["issues"], [existing_finding])

    def test_empty_findings_do_not_add_empty_stage_entries(self):
        updated = append_transcription_findings(self.quality_notes, 7, [])

        self.assertEqual(updated, self.quality_notes)
        self.assertIsNot(updated, self.quality_notes)

    def test_rejects_findings_with_missing_fields_or_wrong_stage(self):
        with self.assertRaisesRegex(ValueError, "all required quality fields"):
            append_transcription_findings(self.quality_notes, 7, [{"category": "repeated_text"}])

        invalid_finding = {
            **inspect_transcription_output("Text with [unclear].")[0],
            "source_stage": "submission",
        }
        with self.assertRaisesRegex(ValueError, "source_stage must be transcription"):
            append_transcription_findings(self.quality_notes, 7, [invalid_finding])


class HandleTranscriptionProviderErrorTests(unittest.TestCase):
    def test_content_failure_becomes_quality_evidence_for_verification(self):
        error = GeminiContentError("Gemini response contains no usable transcript text")

        finding = handle_transcription_provider_error(error)

        self.assertEqual(finding["category"], "uncertain_or_missing_content")
        self.assertEqual(finding["severity"], "blocking")
        self.assertEqual(finding["confidence"], "high")
        self.assertEqual(finding["evidence"], {"provider_error": str(error)})
        self.assertEqual(finding["action"], "manual_verify")
        self.assertEqual(finding["source_stage"], "transcription")

        updated = append_transcription_findings(
            {"submission_id": "submission-1", "pages": []},
            7,
            [finding],
        )
        self.assertEqual(updated["pages"][0]["issues"], [finding])

    def test_configuration_and_request_failures_propagate_unchanged(self):
        errors = [
            GeminiConfigurationError("API key is missing"),
            GeminiRequestError("request retries were exhausted"),
        ]

        for error in errors:
            with self.subTest(category=error.category):
                with self.assertRaises(type(error)) as raised:
                    handle_transcription_provider_error(error)
                self.assertIs(raised.exception, error)

    def test_rejects_non_provider_exception(self):
        with self.assertRaisesRegex(TypeError, "GeminiProviderError"):
            handle_transcription_provider_error(ValueError("not a provider error"))


class WriteTranscriptionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name) / "submission"
        self.manifest = {
            "submission_id": "submission-1",
            "status": "in_progress",
            "pages": [
                {"page_number": 1, "image_path": "page_001.png", "page_status": "accepted"},
                {
                    "page_number": 2,
                    "image_path": "page_002.png",
                    "page_status": "skipped",
                    "skip_reason": "blank_page",
                },
            ],
        }
        self.quality_notes = {
            "submission_id": "submission-1",
            "generated_at": "2026-10-13T12:00:00Z",
            "pages": [
                {"page_number": 1, "stage": "submission", "issues": []},
                {
                    "page_number": 1,
                    "stage": "transcription",
                    "issues": [{"category": "repeated_text", "note": "Verify repetition."}],
                },
            ],
        }

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_context(self, *, quality_submission_id=None):
        return {
            "manifest": self.manifest,
            "quality_notes": {
                **self.quality_notes,
                "submission_id": quality_submission_id or self.quality_notes["submission_id"],
            },
            "manifest_path": str(self.directory / "submission_manifest.json"),
            "quality_notes_path": str(self.directory / "quality_notes.json"),
        }

    def test_persists_manifest_and_quality_notes_to_the_same_checkpoint(self):
        context = self.make_context()

        manifest_path, quality_notes_path = write_transcription_checkpoint(context)

        self.assertEqual(Path(manifest_path), self.directory / "submission_manifest.json")
        self.assertEqual(Path(quality_notes_path), self.directory / "quality_notes.json")
        self.assertEqual(
            json.loads(Path(manifest_path).read_text(encoding="utf-8")),
            self.manifest,
        )
        self.assertEqual(
            json.loads(Path(quality_notes_path).read_text(encoding="utf-8")),
            context["quality_notes"],
        )

    def test_rejects_mismatched_submission_ids_without_writing(self):
        context = self.make_context(quality_submission_id="another-submission")

        with self.assertRaisesRegex(ValueError, "same submission_id"):
            write_transcription_checkpoint(context)

        self.assertFalse(self.directory.exists())