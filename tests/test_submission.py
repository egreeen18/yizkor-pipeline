import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf
from PIL import Image, ImageDraw

from active.submission import (
    create_quality_notes,
    create_submission_manifest,
    create_submission,
    decide_initial_page_state,
    evaluate_page_quality,
    evaluate_submission_pages,
    normalize_submission_path,
    render_pdf_pages,
    validate_submission_pdf,
    write_submission_checkpoint,
)


class NormalizeSubmissionPathTests(unittest.TestCase):
    def test_strips_whitespace_and_surrounding_quotes(self):
        self.assertEqual(
            normalize_submission_path('  "  book.pdf  "  '),
            "book.pdf",
        )
        self.assertEqual(normalize_submission_path(" 'book.pdf' "), "book.pdf")

    def test_rejects_non_string_input(self):
        with self.assertRaises(TypeError):
            normalize_submission_path(None)


class ValidateSubmissionPdfTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_pdf(self, filename="book.pdf", content=b"%PDF-1.7\n"):
        pdf_path = self.directory / filename
        pdf_path.write_bytes(content)
        return pdf_path

    def test_accepts_pdf_and_returns_normalized_path(self):
        pdf_path = self.make_pdf()

        self.assertEqual(
            validate_submission_pdf(f' "{pdf_path}" '),
            str(pdf_path),
        )

    def test_accepts_uppercase_pdf_extension(self):
        pdf_path = self.make_pdf("book.PDF")

        self.assertEqual(validate_submission_pdf(str(pdf_path)), str(pdf_path))

    def test_rejects_missing_path(self):
        with self.assertRaises(ValueError):
            validate_submission_pdf(str(self.directory / "missing.pdf"))

    def test_rejects_directory_even_when_name_has_pdf_extension(self):
        pdf_directory = self.directory / "folder.pdf"
        pdf_directory.mkdir()

        with self.assertRaises(ValueError):
            validate_submission_pdf(str(pdf_directory))

    def test_rejects_non_pdf_extension(self):
        text_path = self.make_pdf("book.txt")

        with self.assertRaises(ValueError):
            validate_submission_pdf(str(text_path))

    def test_rejects_file_without_pdf_header(self):
        invalid_pdf = self.make_pdf(content=b"not a PDF")

        with self.assertRaises(ValueError):
            validate_submission_pdf(str(invalid_pdf))


class RenderPdfPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_renderable_pdf(self, page_count):
        pdf_path = self.directory / "renderable.pdf"
        document = pymupdf.open()
        for page_number in range(page_count):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {page_number + 1}")
        document.save(pdf_path)
        document.close()
        return pdf_path

    def test_renders_pages_in_original_order(self):
        pdf_path = self.make_renderable_pdf(2)
        image_directory = self.directory / "rendered"

        pages = render_pdf_pages(str(pdf_path), str(image_directory))

        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertEqual(
            [Path(page["image_path"]).name for page in pages],
            ["page_001.png", "page_002.png"],
        )
        for page in pages:
            self.assertTrue(Path(page["image_path"]).is_file())
            self.assertTrue(Path(page["image_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_rejects_pdf_with_no_pages(self):
        pdf_path = self.make_renderable_pdf(1)
        empty_document = Mock(page_count=0)

        with patch("pymupdf.open", return_value=empty_document):
            with self.assertRaisesRegex(ValueError, "no pages"):
                render_pdf_pages(str(pdf_path), str(self.directory / "rendered"))

        empty_document.close.assert_called_once_with()

    def test_reports_invalid_pdf_structure(self):
        pdf_path = self.directory / "invalid.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nnot a complete PDF")

        with self.assertRaisesRegex(OSError, "Could not open PDF"):
            render_pdf_pages(str(pdf_path), str(self.directory / "rendered"))

    def test_preserves_page_record_when_one_page_cannot_be_rendered(self):
        pdf_path = self.make_renderable_pdf(4)
        first_page = Mock()
        first_page.get_pixmap.return_value = Mock()
        failed_page = Mock()
        failed_page.get_pixmap.side_effect = RuntimeError("damaged page")
        failed_save_page = Mock()
        failed_save_pixmap = Mock()
        failed_save_pixmap.save.side_effect = OSError("image write failed")
        failed_save_page.get_pixmap.return_value = failed_save_pixmap
        last_page = Mock()
        last_page.get_pixmap.return_value = Mock()
        document = Mock(page_count=4)
        document.load_page.side_effect = [
            first_page,
            failed_page,
            failed_save_page,
            last_page,
        ]

        with patch("pymupdf.open", return_value=document):
            pages = render_pdf_pages(str(pdf_path), str(self.directory / "rendered"))

        self.assertEqual([page["page_number"] for page in pages], [1, 2, 3, 4])
        self.assertIsNone(pages[1]["image_path"])
        self.assertEqual(pages[1]["render_error"], "damaged page")
        self.assertIsNone(pages[2]["image_path"])
        self.assertEqual(pages[2]["render_error"], "image write failed")
        self.assertTrue(pages[0]["image_path"].endswith("page_001.png"))
        self.assertTrue(pages[3]["image_path"].endswith("page_004.png"))
        document.close.assert_called_once_with()


class PageQualityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def save_image(self, filename, size, color):
        image_path = self.directory / filename
        Image.new("RGB", size, color).save(image_path)
        return image_path

    def test_flags_blank_page_without_modifying_source_image(self):
        image_path = self.save_image("blank.png", (800, 800), "white")
        original_bytes = image_path.read_bytes()

        issues = evaluate_page_quality(1, str(image_path))

        self.assertEqual([issue["category"] for issue in issues], ["blank_page"])
        self.assertEqual(issues[0]["severity"], "blocking")
        self.assertEqual(issues[0]["action"], "skipped")
        self.assertEqual(issues[0]["source_stage"], "submission")
        self.assertEqual(image_path.read_bytes(), original_bytes)

    def test_flags_low_resolution_page(self):
        image_path = self.save_image("small.png", (300, 600), "black")

        issues = evaluate_page_quality(2, str(image_path))

        self.assertEqual([issue["category"] for issue in issues], ["low_resolution"])
        self.assertEqual(issues[0]["severity"], "warning")
        self.assertEqual(issues[0]["evidence"]["width"], 300)

    def test_clean_page_has_no_issues(self):
        image_path = self.save_image("readable.png", (800, 800), "black")

        self.assertEqual(evaluate_page_quality(3, str(image_path)), [])

    def test_sparse_printed_pdf_page_is_not_classified_as_blank(self):
        pdf_path = self.directory / "sparse.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "A short but visible line of text")
        document.save(pdf_path)
        document.close()
        rendered_page = render_pdf_pages(
            str(pdf_path),
            str(self.directory / "sparse-pages"),
        )[0]

        issues = evaluate_page_quality(rendered_page["page_number"], rendered_page["image_path"])

        self.assertNotIn("blank_page", [issue["category"] for issue in issues])

    def test_quality_evaluation_turns_render_failure_into_blocking_finding(self):
        rendered_pages = [{
            "page_number": 8,
            "image_path": None,
            "render_error": "damaged page",
        }]

        manifest_pages, quality_pages = evaluate_submission_pages(rendered_pages)

        self.assertEqual(manifest_pages[0]["page_number"], 8)
        self.assertIsNone(manifest_pages[0]["image_path"])
        self.assertEqual(manifest_pages[0]["page_status"], "rejected")
        self.assertEqual(quality_pages[0]["issues"][0]["category"], "rendering_failure")
        self.assertEqual(quality_pages[0]["issues"][0]["evidence"]["error"], "damaged page")

    def test_unreadable_page_becomes_a_blocking_finding(self):
        image_path = self.directory / "unreadable.png"
        image_path.write_text("not an image", encoding="utf-8")

        issues = evaluate_page_quality(4, str(image_path))

        self.assertEqual([issue["category"] for issue in issues], ["unreadable_image"])
        self.assertEqual(issues[0]["severity"], "blocking")
        self.assertEqual(issues[0]["action"], "exclude")


class PageStateDecisionTests(unittest.TestCase):
    def test_blank_page_is_skipped_even_with_other_issues(self):
        issues = [
            {"category": "low_resolution", "severity": "warning", "action": "review"},
            {"category": "blank_page", "severity": "blocking", "action": "skipped"},
        ]

        self.assertEqual(decide_initial_page_state(issues), ("skipped", "blank_page"))

    def test_blocking_issue_rejects_page(self):
        issues = [{"category": "unreadable_image", "severity": "blocking", "action": "exclude"}]

        self.assertEqual(decide_initial_page_state(issues), ("rejected", None))

    def test_warning_requires_review_and_no_issues_are_accepted(self):
        warning = [{"category": "low_resolution", "severity": "warning", "action": "review"}]

        self.assertEqual(decide_initial_page_state(warning), ("needs_review", None))
        self.assertEqual(decide_initial_page_state([]), ("accepted", None))


class EvaluateSubmissionPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_preserves_page_order_and_keeps_skipped_pages_in_both_records(self):
        blank_path = self.directory / "page_004.png"
        Image.new("RGB", (800, 800), "white").save(blank_path)
        readable_path = self.directory / "page_005.png"
        image = Image.new("RGB", (800, 800), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 500, 700), fill="black")
        image.save(readable_path)
        rendered_pages = [
            {"page_number": 4, "image_path": str(blank_path)},
            {"page_number": 5, "image_path": str(readable_path)},
        ]

        manifest_pages, quality_pages = evaluate_submission_pages(rendered_pages)

        self.assertEqual([page["page_number"] for page in manifest_pages], [4, 5])
        self.assertEqual(manifest_pages[0]["page_status"], "skipped")
        self.assertEqual(manifest_pages[0]["skip_reason"], "blank_page")
        self.assertEqual(manifest_pages[1]["page_status"], "accepted")
        self.assertEqual([page["page_number"] for page in quality_pages], [4, 5])
        self.assertEqual(quality_pages[0]["issues"][0]["category"], "blank_page")
        self.assertEqual(quality_pages[1]["issues"], [])


class SubmissionArtifactCreationTests(unittest.TestCase):
    def setUp(self):
        self.pages = [
            {"page_number": 4, "image_path": "page_004.png", "page_status": "accepted"},
            {
                "page_number": 5,
                "image_path": "page_005.png",
                "page_status": "skipped",
                "skip_reason": "blank_page",
            },
        ]
        self.issue = {
            "category": "blank_page",
            "severity": "blocking",
            "confidence": "high",
            "evidence": {"blank_ratio": 1.0},
            "action": "skipped",
            "source_stage": "submission",
            "note": "The page appears blank.",
        }
        self.page_findings = [
            {"page_number": 4, "stage": "submission", "issues": []},
            {"page_number": 5, "stage": "submission", "issues": [self.issue]},
        ]

    def test_manifest_contains_ordered_workflow_state_without_evidence(self):
        manifest = create_submission_manifest(
            "submission-1",
            "book.pdf",
            "2026-09-29T12:00:00Z",
            self.pages,
        )

        self.assertEqual(manifest["status"], "in_progress")
        self.assertEqual([page["page_number"] for page in manifest["pages"]], [4, 5])
        self.assertEqual(manifest["pages"][1]["page_status"], "skipped")
        self.assertEqual(manifest["pages"][1]["skip_reason"], "blank_page")
        self.assertNotIn("issues", manifest["pages"][1])
        self.assertNotIn("flags", manifest["pages"][1])

    def test_manifest_rejects_invalid_or_unordered_page_records(self):
        with self.assertRaises(ValueError):
            create_submission_manifest(
                "submission-1",
                "book.pdf",
                "2026-09-29T12:00:00Z",
                [
                    {"page_number": 2, "image_path": "page_002.png", "page_status": "accepted"},
                    {"page_number": 1, "image_path": "page_001.png", "page_status": "accepted"},
                ],
            )

        with self.assertRaises(ValueError):
            create_submission_manifest(
                "submission-1",
                "book.pdf",
                "2026-09-29T12:00:00Z",
                [{"page_number": 1, "image_path": "page_001.png", "page_status": "skipped"}],
            )

    def test_quality_notes_preserve_issue_evidence(self):
        quality_notes = create_quality_notes(
            "submission-1",
            "2026-09-29T12:00:30Z",
            self.page_findings,
        )

        self.assertEqual(quality_notes["submission_id"], "submission-1")
        self.assertEqual(quality_notes["pages"][1]["issues"][0], self.issue)
        self.assertEqual(quality_notes["pages"][0]["issues"], [])
        self.assertIsNot(quality_notes["pages"][1]["issues"], self.page_findings[1]["issues"])

    def test_quality_notes_reject_incomplete_issue_records(self):
        with self.assertRaises(ValueError):
            create_quality_notes(
                "submission-1",
                "2026-09-29T12:00:30Z",
                [{
                    "page_number": 1,
                    "stage": "submission",
                    "issues": [{"category": "blank_page"}],
                }],
            )

    def test_quality_notes_reject_mismatched_issue_provenance(self):
        mismatched_findings = [{
            "page_number": 1,
            "stage": "submission",
            "issues": [{**self.issue, "source_stage": "translation"}],
        }]

        with self.assertRaisesRegex(ValueError, "source_stage must match"):
            create_quality_notes(
                "submission-1",
                "2026-09-29T12:00:30Z",
                mismatched_findings,
            )

    def test_checkpoint_writes_both_json_files(self):
        manifest = create_submission_manifest(
            "submission-1",
            "book.pdf",
            "2026-09-29T12:00:00Z",
            self.pages,
        )
        quality_notes = create_quality_notes(
            "submission-1",
            "2026-09-29T12:00:30Z",
            self.page_findings,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "checkpoint"

            manifest_path, quality_notes_path = write_submission_checkpoint(
                str(output_directory),
                manifest,
                quality_notes,
            )

            self.assertEqual(Path(manifest_path).name, "submission_manifest.json")
            self.assertEqual(Path(quality_notes_path).name, "quality_notes.json")
            self.assertEqual(json.loads(Path(manifest_path).read_text(encoding="utf-8")), manifest)
            self.assertEqual(json.loads(Path(quality_notes_path).read_text(encoding="utf-8")), quality_notes)

    def test_checkpoint_rejects_mismatched_submission_ids_before_writing(self):
        manifest = {"submission_id": "submission-1"}
        quality_notes = {"submission_id": "submission-2"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                write_submission_checkpoint(
                    temporary_directory,
                    manifest,
                    quality_notes,
                )

            self.assertEqual(list(Path(temporary_directory).iterdir()), [])


class CreateSubmissionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_pdf(self, filename, with_text):
        pdf_path = self.directory / filename
        document = pymupdf.open()
        page = document.new_page()
        if with_text:
            for line_number in range(25):
                page.insert_text(
                    (72, 60 + line_number * 24),
                    "A readable page with enough printed text for evaluation.",
                )
        document.save(pdf_path)
        document.close()
        return pdf_path

    def test_creates_rendered_pages_and_initial_checkpoint(self):
        pdf_path = self.make_pdf("readable.pdf", with_text=True)
        output_directory = self.directory / "submission"

        result = create_submission(
            str(pdf_path),
            "submission-1",
            str(output_directory),
            "2026-09-29T12:00:00Z",
        )

        self.assertEqual(result["manifest"]["status"], "in_progress")
        self.assertEqual(result["manifest"]["pages"][0]["page_status"], "accepted")
        self.assertTrue(Path(result["manifest"]["pages"][0]["image_path"]).is_file())
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertTrue(Path(result["quality_notes_path"]).is_file())
        self.assertEqual(result["quality_notes"]["generated_at"], "2026-09-29T12:00:00Z")

    def test_persists_rejected_checkpoint_when_no_pages_are_usable(self):
        pdf_path = self.make_pdf("blank.pdf", with_text=False)
        output_directory = self.directory / "rejected-submission"

        result = create_submission(
            str(pdf_path),
            "submission-blank",
            str(output_directory),
            "2026-09-29T12:00:00Z",
        )

        self.assertEqual(result["manifest"]["status"], "rejected")
        self.assertEqual(result["manifest"]["pages"][0]["page_status"], "skipped")
        self.assertEqual(result["manifest"]["pages"][0]["skip_reason"], "blank_page")
        self.assertEqual(
            result["quality_notes"]["pages"][0]["issues"][0]["category"],
            "blank_page",
        )
        self.assertTrue(Path(result["manifest_path"]).is_file())
        self.assertTrue(Path(result["quality_notes_path"]).is_file())


if __name__ == "__main__":
    unittest.main()