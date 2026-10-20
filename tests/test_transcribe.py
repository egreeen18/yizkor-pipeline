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
    parse_transcription_result,
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

    def test_rejects_non_string_manifest_page_state(self):
        manifest = {
            **self.manifest,
            "pages": [{
                "page_number": 1,
                "image_path": "page_001.png",
                "page_status": [],
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

    def test_builds_page_job_with_relevant_quality_context_for_revisit(self):
        page_job = prepare_page_job(
            self.page,
            self.quality_notes,
            self.directory,
            revisit=True,
        )

        self.assertEqual(page_job["page_number"], 11)
        self.assertEqual(page_job["scan_page_number"], 11)
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

    def test_includes_source_issues_but_not_prior_transcription_issues_for_normal_job(self):
        page_job = prepare_page_job(self.page, self.quality_notes, self.directory)

        self.assertEqual(
            page_job["quality_context"],
            [{"category": "low_contrast", "note": "Dim scan."}],
        )

    def test_rejects_pages_that_must_not_be_sent_to_the_model(self):
        for status in ("rejected", "skipped"):
            with self.subTest(status=status):
                page = {**self.page, "page_status": status}
                with self.assertRaisesRegex(ValueError, "accepted or needs_review"):
                    prepare_page_job(page, self.quality_notes, self.directory)

    def test_rejects_non_string_page_status(self):
        page = {**self.page, "page_status": []}

        with self.assertRaisesRegex(ValueError, "accepted or needs_review"):
            prepare_page_job(page, self.quality_notes, self.directory)

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

        self.assertIn("scan image 11", prompt)
        self.assertIn("original language and writing system", prompt)
        self.assertIn("reading order", prompt)
        self.assertIn("Do not translate", prompt)
        for prohibited_action in ("summarize", "modernize", "correct", "infer", "invent"):
            with self.subTest(prohibited_action=prohibited_action):
                self.assertIn(prohibited_action, prompt)
        self.assertIn("[unclear]", prompt)
        self.assertIn("Return only valid JSON", prompt)
        self.assertIn("two-page spread", prompt)
        self.assertIn("printed_page_number", prompt)
        self.assertIn("Do not assume a fixed number of regions", prompt)
        self.assertNotIn("low_contrast", prompt)
        self.assertNotIn("Dim scan", prompt)

    def test_prompt_rejects_invalid_page_job(self):
        with self.assertRaisesRegex(ValueError, "page_job must be an object"):
            build_transcription_prompt(None)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_transcription_prompt({"page_number": 0})

    def test_prompt_rejects_conflicting_scan_page_number(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            build_transcription_prompt({
                "page_number": 11,
                "scan_page_number": 12,
            })


class ParseTranscriptionResultTests(unittest.TestCase):
    def test_adds_authoritative_scan_index_and_preserves_region_order(self):
        response = json.dumps({
            "regions": [
                {
                    "position": "right",
                    "printed_page_number": "11",
                    "language": "Hebrew",
                    "transcription": "Right-side text.",
                },
                {
                    "position": "left",
                    "printed_page_number": "12",
                    "language": "Hebrew",
                    "transcription": "Left-side text.",
                },
            ],
        })

        result = parse_transcription_result(response, 6)

        self.assertEqual(result["scan_page_number"], 6)
        self.assertEqual(result["regions"][0]["printed_page_number"], "11")
        self.assertEqual(result["regions"][1]["printed_page_number"], "12")
        self.assertEqual(result["regions"][0]["position"], "right")
        self.assertEqual(result["regions"][1]["position"], "left")

    def test_accepts_single_region_and_uncertain_optional_metadata(self):
        response = json.dumps({
            "regions": [{
                "position": "center",
                "printed_page_number": None,
                "language": None,
                "transcription": "Visible text.",
            }],
        })

        result = parse_transcription_result(response, 25)

        self.assertEqual(len(result["regions"]), 1)
        self.assertIsNone(result["regions"][0]["printed_page_number"])
        self.assertIsNone(result["regions"][0]["language"])

    def test_accepts_no_text_regions_as_a_structural_result(self):
        self.assertEqual(
            parse_transcription_result('{"regions": []}', 26),
            {"scan_page_number": 26, "regions": []},
        )

    def test_rejects_invalid_json_and_malformed_regions(self):
        invalid_responses = [
            ("not JSON", "valid JSON"),
            ('{"transcription":"text"}', "regions array"),
            ('{"regions":{}}', "regions must be a list"),
            (
                '{"regions":[{"position":"right","printed_page_number":11,'
                '"language":"Hebrew"}]}',
                "must contain exactly",
            ),
            (
                '{"regions":[{"position":"","printed_page_number":null,'
                '"language":null,"transcription":"text"}]}',
                "position must be non-empty",
            ),
            (
                '{"regions":[{"position":"right","printed_page_number":true,'
                '"language":null,"transcription":"text"}]}',
                "printed_page_number must be",
            ),
            (
                '{"regions":[{"position":"right","printed_page_number":null,'
                '"language":" ","transcription":"text"}]}',
                "language must be",
            ),
            (
                '{"regions":[{"position":"right","printed_page_number":null,'
                '"language":null,"transcription":" "}]}',
                "transcription must be non-empty",
            ),
        ]
        for response, message in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, message):
                    parse_transcription_result(response, 1)

    def test_rejects_invalid_scan_page_number(self):
        for scan_page_number in (0, True, "6"):
            with self.subTest(scan_page_number=scan_page_number):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    parse_transcription_result('{"regions":[]}', scan_page_number)


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
            return_value=(
                '{"regions":[{"position":"single","printed_page_number":null,'
                '"language":"Hebrew","transcription":"Transcribed text."}]}'
            ),
        ) as provider_call:
            result = transcribe_page_with_gemini(
                page_job,
                prompt,
                client=client,
                revisit=True,
            )

        self.assertEqual(result["scan_page_number"], 11)
        self.assertEqual(
            result["regions"][0]["transcription"],
            "Transcribed text.",
        )
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

    def test_rejects_malformed_provider_json(self):
        with patch(
            "active.transcribe.transcribe_gemini_page",
            return_value="not JSON",
        ):
            with self.assertRaisesRegex(GeminiContentError, "valid JSON"):
                transcribe_page_with_gemini(
                    {"page_number": 11},
                    "prompt",
                )

    def test_rejects_conflicting_manifest_and_scan_page_indices(self):
        with patch(
            "active.transcribe.transcribe_gemini_page",
            return_value='{"regions":[]}',
        ) as provider_call:
            with self.assertRaisesRegex(ValueError, "must match"):
                transcribe_page_with_gemini(
                    {"page_number": 11, "scan_page_number": 12},
                    "prompt",
                )
        provider_call.assert_not_called()


class InspectTranscriptionOutputTests(unittest.TestCase):
    def setUp(self):
        self.single_region_result = {
            "scan_page_number": 7,
            "regions": [{
                "position": "single",
                "printed_page_number": "4",
                "language": "Yiddish",
                "transcription": "First line.\nSecond line.",
            }],
        }

    def test_clean_transcript_has_no_quality_findings(self):
        self.assertEqual(
            inspect_transcription_output(self.single_region_result),
            [],
        )

    def test_unclear_markers_create_quality_note_compatible_finding(self):
        result = {
            **self.single_region_result,
            "regions": [{
                **self.single_region_result["regions"][0],
                "transcription": "Readable [unclear] passage; [UNCLEAR].",
            }],
        }
        findings = inspect_transcription_output(result)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "uncertain_or_missing_content")
        self.assertEqual(findings[0]["severity"], "warning")
        self.assertEqual(findings[0]["confidence"], "high")
        self.assertEqual(
            findings[0]["evidence"],
            {
                "regions": [{
                    "region_index": 1,
                    "position": "single",
                    "unclear_marker_count": 2,
                }],
            },
        )
        self.assertEqual(findings[0]["action"], "manual_verify")
        self.assertEqual(findings[0]["source_stage"], "transcription")

    def test_adjacent_duplicate_lines_are_flagged_with_line_numbers(self):
        result = {
            **self.single_region_result,
            "regions": [{
                **self.single_region_result["regions"][0],
                "transcription": "Opening line.\nRepeated line.\nRepeated line.",
            }],
        }
        findings = inspect_transcription_output(result)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "repeated_text")
        self.assertEqual(
            findings[0]["evidence"],
            {
                "regions": [{
                    "region_index": 1,
                    "position": "single",
                    "duplicate_line_numbers": [3],
                }],
            },
        )
        self.assertEqual(findings[0]["action"], "manual_verify")

    def test_nonadjacent_duplicate_lines_are_not_flagged(self):
        self.assertEqual(
            inspect_transcription_output({
                **self.single_region_result,
                "regions": [{
                    **self.single_region_result["regions"][0],
                    "transcription": "Repeated line.\nOther line.\nRepeated line.",
                }],
            }),
            [],
        )

    def test_flags_empty_region_list_for_human_review(self):
        findings = inspect_transcription_output({
            "scan_page_number": 7,
            "regions": [],
        })

        self.assertEqual(findings[0]["severity"], "blocking")
        self.assertEqual(findings[0]["category"], "uncertain_or_missing_content")
        self.assertEqual(
            findings[0]["evidence"],
            {"scan_page_number": 7, "region_count": 0},
        )
        self.assertEqual(findings[0]["action"], "manual_verify")

    def test_duplicate_detection_is_scoped_to_each_region(self):
        result = {
            "scan_page_number": 7,
            "regions": [
                {
                    "position": "right",
                    "printed_page_number": "11",
                    "language": "Hebrew",
                    "transcription": "Shared line.",
                },
                {
                    "position": "left",
                    "printed_page_number": "12",
                    "language": "Hebrew",
                    "transcription": "Shared line.",
                },
            ],
        }

        self.assertEqual(inspect_transcription_output(result), [])

    def test_rejects_malformed_result_or_region(self):
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            inspect_transcription_output(None)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "positive integer"):
            inspect_transcription_output({"scan_page_number": 0, "regions": []})

        with self.assertRaisesRegex(ValueError, "must be a list"):
            inspect_transcription_output({
                "scan_page_number": 1,
                "regions": {},
            })

        with self.assertRaisesRegex(ValueError, "must be an object"):
            inspect_transcription_output({
                "scan_page_number": 1,
                "regions": [None],
            })


class SaveTranscriptionResultTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_result(self, text="שלום\nA faithful transcription.\n"):
        return {
            "scan_page_number": 7,
            "regions": [{
                "position": "single",
                "printed_page_number": "3",
                "language": "Hebrew",
                "transcription": text,
            }],
        }

    def test_saves_structured_result_as_utf8_json(self):
        result = self.make_result()
        saved_path = save_transcription_result(self.directory, 7, result)

        expected_path = self.directory / "transcripts" / "page_007.json"
        self.assertEqual(Path(saved_path), expected_path)
        self.assertEqual(
            json.loads(expected_path.read_text(encoding="utf-8")),
            result,
        )

    def test_saves_revisits_as_numbered_attempts_without_overwriting(self):
        initial_path = Path(save_transcription_result(
            self.directory, 7, self.make_result("Initial text.")
        ))
        second_path = Path(save_transcription_result(
            self.directory, 7, self.make_result("Revisit text.")
        ))
        third_path = Path(save_transcription_result(
            self.directory, 7, self.make_result("Another revisit.")
        ))

        attempts_directory = self.directory / "transcripts" / "page_007"
        self.assertEqual(
            json.loads(initial_path.read_text(encoding="utf-8"))["regions"][0]["transcription"],
            "Initial text.",
        )
        self.assertEqual(second_path, attempts_directory / "attempt_002.json")
        self.assertEqual(third_path, attempts_directory / "attempt_003.json")
        self.assertEqual(
            json.loads(second_path.read_text(encoding="utf-8"))["regions"][0]["transcription"],
            "Revisit text.",
        )
        self.assertEqual(
            json.loads(third_path.read_text(encoding="utf-8"))["regions"][0]["transcription"],
            "Another revisit.",
        )

    def test_rejects_invalid_page_number_and_mismatched_scan_identity(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            save_transcription_result(self.directory, 0, self.make_result())

        with self.assertRaisesRegex(ValueError, "must match"):
            save_transcription_result(self.directory, 8, self.make_result())

    def test_rejects_malformed_region_result(self):
        result = self.make_result()
        result["regions"] = [{
            "position": "single",
            "printed_page_number": "3",
            "language": "Hebrew",
            "transcription": " ",
        }]

        with self.assertRaisesRegex(ValueError, "transcription must be non-empty"):
            save_transcription_result(self.directory, 7, result)


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

    def inspect_sample(self):
        return inspect_transcription_output({
            "scan_page_number": 7,
            "regions": [{
                "position": "single",
                "printed_page_number": None,
                "language": None,
                "transcription": "Text with [unclear].",
            }],
        })

    def test_creates_transcription_stage_entry_without_mutating_input(self):
        findings = self.inspect_sample()
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
        new_finding = self.inspect_sample()[0]

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
            **self.inspect_sample()[0],
            "source_stage": "submission",
        }
        with self.assertRaisesRegex(ValueError, "source_stage must be transcription"):
            append_transcription_findings(self.quality_notes, 7, [invalid_finding])

    def test_rejects_actions_outside_the_quality_specification(self):
        invalid_finding = {
            **self.inspect_sample()[0],
            "action": "skipped",
        }

        with self.assertRaisesRegex(ValueError, "invalid issue action"):
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