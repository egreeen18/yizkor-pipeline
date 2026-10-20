import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib import error as urllib_error
from unittest.mock import patch

from active.gemini_api import (
    GeminiConfigurationError,
    GeminiContentError,
    GeminiProviderError,
    GeminiRequestError,
    build_gemini_client,
    build_gemini_ocr_request,
    get_gemini_api_key,
    get_gemini_config,
    normalize_gemini_error,
    parse_gemini_response,
    send_gemini_request,
    transcribe_gemini_image,
    transcribe_gemini_page,
)


class GeminiConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_get_gemini_api_key_reads_environment_value(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "  test-key-123  "}, clear=True),
            patch("active.gemini_api.get_stored_gemini_api_key") as stored_key,
        ):
            self.assertEqual(get_gemini_api_key(), "test-key-123")
        stored_key.assert_not_called()

    def test_get_gemini_api_key_rejects_missing_value(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("active.gemini_api.get_stored_gemini_api_key", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "store_gemini_api_key"):
                get_gemini_api_key()

    def test_get_gemini_api_key_uses_stored_value_when_environment_is_absent(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "active.gemini_api.get_stored_gemini_api_key",
                return_value="  saved-key  ",
            ),
        ):
            self.assertEqual(get_gemini_api_key(), "saved-key")

    def test_get_gemini_api_key_reports_unavailable_credential_store(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "active.gemini_api.get_stored_gemini_api_key",
                side_effect=RuntimeError("Credential Manager unavailable"),
            ),
        ):
            with self.assertRaisesRegex(
                GeminiConfigurationError,
                "Credential Manager unavailable",
            ):
                get_gemini_api_key()

    def test_get_gemini_config_uses_defaults_and_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "env-key",
                "GEMINI_MODEL": "gemini-test-model",
                "GEMINI_TIMEOUT_SECONDS": "120",
                "GEMINI_MAX_RETRIES": "7",
            },
            clear=True,
        ):
            config = get_gemini_config()

        self.assertEqual(config["api_key"], "env-key")
        self.assertEqual(config["model"], "gemini-test-model")
        self.assertEqual(config["timeout_seconds"], 120)
        self.assertEqual(config["max_retries"], 7)

    def test_get_gemini_config_uses_project_defaults_when_env_is_absent(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret"}, clear=True):
            config = get_gemini_config()

        self.assertEqual(config["api_key"], "secret")
        self.assertEqual(config["model"], "gemini-2.5-flash")
        self.assertEqual(config["timeout_seconds"], 60)
        self.assertEqual(config["max_retries"], 3)

    def test_get_gemini_config_selects_test_and_revisit_models(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "secret",
                "GEMINI_MODEL": "operator-normal-model",
                "GEMINI_TEST_MODE": "yes",
                "GEMINI_REVISIT_MODEL": "operator-revisit-model",
            },
            clear=True,
        ):
            test_config = get_gemini_config()
            revisit_config = get_gemini_config(revisit=True)

        self.assertEqual(test_config["model"], "gemini-2.5-flash-lite")
        self.assertEqual(revisit_config["model"], "operator-revisit-model")

    def test_get_gemini_config_rejects_invalid_test_mode(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "secret", "GEMINI_TEST_MODE": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(GeminiConfigurationError, "GEMINI_TEST_MODE"):
                get_gemini_config()

    def test_build_gemini_client_uses_configured_values(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "  secret-key  ",
                "GEMINI_MODEL": "gemini-test-model",
                "GEMINI_TIMEOUT_SECONDS": "45",
                "GEMINI_MAX_RETRIES": "5",
            },
            clear=True,
        ):
            client = build_gemini_client()

        self.assertEqual(client.api_key, "secret-key")
        self.assertEqual(client.model, "gemini-test-model")
        self.assertEqual(client.timeout_seconds, 45)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.headers["x-goog-api-key"], "secret-key")
        self.assertTrue(client.generate_content_url.endswith("gemini-test-model:generateContent"))

    def test_build_gemini_client_rejects_missing_required_values(self):
        with self.assertRaisesRegex(ValueError, "api_key"):
            build_gemini_client({"api_key": "", "model": "m", "timeout_seconds": 10, "max_retries": 1})

        with self.assertRaisesRegex(ValueError, "model"):
            build_gemini_client({"api_key": "key", "model": "", "timeout_seconds": 10, "max_retries": 1})

    def test_build_gemini_ocr_request_includes_prompt_and_inline_image(self):
        image_path = self.directory / "page.png"
        image_path.write_bytes(b"fake-image-bytes")

        payload = build_gemini_ocr_request(
            str(image_path),
            "Transcribe this page faithfully.",
        )

        self.assertIn("contents", payload)
        self.assertEqual(len(payload["contents"]), 1)
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "Transcribe this page faithfully.")
        self.assertIn("inlineData", parts[1])
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(parts[1]["inlineData"]["data"], "ZmFrZS1pbWFnZS1ieXRlcw==")

    def test_build_gemini_ocr_request_validates_image_path_and_prompt(self):
        with self.assertRaisesRegex(ValueError, "image_path"):
            build_gemini_ocr_request("", "prompt")

        with self.assertRaisesRegex(ValueError, "image not found"):
            build_gemini_ocr_request("missing.png", "prompt")

        image_path = self.directory / "demo.png"
        image_path.write_bytes(b"demo")
        with self.assertRaisesRegex(ValueError, "prompt"):
            build_gemini_ocr_request(str(image_path), None)  # type: ignore[arg-type]

    def test_send_gemini_request_returns_parsed_json(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 1,
        })
        payload = {"contents": [{"parts": [{"text": "hello"}]}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}'

        with patch("active.gemini_api.urllib_request.urlopen", return_value=FakeResponse()) as mocked_urlopen:
            result = send_gemini_request(client, payload)

        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "hi")
        mocked_urlopen.assert_called_once()

    def test_send_gemini_request_retries_on_transient_http_error(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 2,
        })
        payload = {"contents": [{"parts": [{"text": "hello"}]}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"candidates": [{"content": {"parts": [{"text": "retry-success"}]}}]}'

        attempts = {"count": 0}

        def fake_urlopen(request, timeout):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib_error.HTTPError(
                    url=request.full_url,
                    code=429,
                    msg="rate limited",
                    hdrs={},
                    fp=io.BytesIO(b'{"error": {"message": "rate limited"}}'),
                )
            return FakeResponse()

        with patch("active.gemini_api.urllib_request.urlopen", side_effect=fake_urlopen):
            with patch("active.gemini_api.time.sleep"):
                result = send_gemini_request(client, payload)

        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "retry-success")
        self.assertEqual(attempts["count"], 2)

    def test_send_gemini_request_rejects_malformed_payload(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 0,
        })

        with self.assertRaisesRegex(ValueError, "payload"):
            send_gemini_request(client, {})

        with self.assertRaisesRegex(ValueError, "Malformed Gemini request"):
            class FakeHTTPErrorResponse:
                def __init__(self):
                    self.body = b'{"error": {"message": "bad request"}}'

            def fake_urlopen(request, timeout):
                raise urllib_error.HTTPError(
                    url=request.full_url,
                    code=400,
                    msg="bad request",
                    hdrs={},
                    fp=io.BytesIO(b'{"error": {"message": "bad request"}}'),
                )

            with patch("active.gemini_api.urllib_request.urlopen", side_effect=fake_urlopen):
                send_gemini_request(client, {"contents": [{"parts": [{"text": "hello"}]}]})


class GeminiResponseParsingTests(unittest.TestCase):
    def test_parse_gemini_response_extracts_transcript_text(self):
        response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "  First line\n"}, {"text": "Second line  "}],
                },
            }],
        }

        self.assertEqual(parse_gemini_response(response), "First line\nSecond line")

    def test_parse_gemini_response_rejects_missing_candidates(self):
        with self.assertRaisesRegex(ValueError, "no candidates"):
            parse_gemini_response({"candidates": []})

    def test_parse_gemini_response_rejects_missing_or_blank_text(self):
        responses = [
            {"candidates": [{"content": {"parts": []}}]},
            {"candidates": [{"content": {"parts": [{"text": "  "}]}}]},
            {"candidates": [{"content": {"parts": [None]}}]},
        ]

        for response in responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "transcript text"):
                    parse_gemini_response(response)


class GeminiErrorNormalizationTests(unittest.TestCase):
    def test_normalize_gemini_error_sets_category_and_retryability(self):
        cases = [
            ("configuration_failure", GeminiConfigurationError, ValueError, False),
            ("transient_request_failure", GeminiRequestError, RuntimeError, True),
            ("content_failure", GeminiContentError, ValueError, False),
        ]

        for category, expected_type, builtin_type, retryable in cases:
            with self.subTest(category=category):
                error = normalize_gemini_error(ValueError("provider detail"), category=category)

                self.assertIsInstance(error, GeminiProviderError)
                self.assertIsInstance(error, expected_type)
                self.assertIsInstance(error, builtin_type)
                self.assertEqual(error.category, category)
                self.assertEqual(error.retryable, retryable)
                self.assertEqual(str(error), "provider detail")

    def test_missing_api_key_is_a_non_retryable_configuration_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GeminiConfigurationError) as raised:
                get_gemini_api_key()

        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(raised.exception.category, "configuration_failure")
        self.assertFalse(raised.exception.retryable)

    def test_empty_transcript_is_a_content_failure(self):
        with self.assertRaises(GeminiContentError) as raised:
            parse_gemini_response({"candidates": [{"content": {"parts": []}}]})

        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(raised.exception.category, "content_failure")
        self.assertFalse(raised.exception.retryable)

    def test_exhausted_transport_failure_is_retryable(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 0,
        })

        with patch(
            "active.gemini_api.urllib_request.urlopen",
            side_effect=urllib_error.URLError("connection unavailable"),
        ):
            with self.assertRaises(GeminiRequestError) as raised:
                send_gemini_request(client, {"contents": [{"parts": [{"text": "hello"}]}]})

        self.assertIsInstance(raised.exception, RuntimeError)
        self.assertEqual(raised.exception.category, "transient_request_failure")
        self.assertTrue(raised.exception.retryable)


    def test_http_408_retries_with_malformed_error_body(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 1,
        })
        attempts = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}'

        def fake_urlopen(request, timeout):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib_error.HTTPError(
                    url=request.full_url,
                    code=408,
                    msg="request timeout",
                    hdrs={},
                    fp=io.BytesIO(b'{"error": null}'),
                )
            return FakeResponse()

        with patch("active.gemini_api.urllib_request.urlopen", side_effect=fake_urlopen):
            with patch("active.gemini_api.time.sleep"):
                response = send_gemini_request(
                    client,
                    {"contents": [{"parts": [{"text": "hello"}]}]},
                )

        self.assertEqual(response["candidates"][0]["content"]["parts"][0]["text"], "ok")
        self.assertEqual(attempts["count"], 2)


class GeminiProviderBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_directory.name) / "page.png"
        self.image_path.write_bytes(b"page-image")

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_transcribe_gemini_image_returns_transcript_text(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 0,
        })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"candidates": [{"content": '
                    b'{"parts": [{"text": "The transcript."}]}}]}'
                )

        with patch(
            "active.gemini_api.urllib_request.urlopen",
            return_value=FakeResponse(),
        ) as mocked_urlopen:
            transcript = transcribe_gemini_image(
                str(self.image_path),
                "Transcribe this page.",
                client=client,
            )

        self.assertEqual(transcript, "The transcript.")
        mocked_urlopen.assert_called_once()

    def test_transcribe_gemini_image_classifies_invalid_image_input(self):
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 0,
        })

        with self.assertRaises(GeminiConfigurationError) as raised:
            transcribe_gemini_image(
                str(Path(self.temp_directory.name) / "missing.png"),
                "Transcribe this page.",
                client=client,
            )

        self.assertEqual(raised.exception.category, "configuration_failure")
        self.assertFalse(raised.exception.retryable)

    def test_transcribe_gemini_page_uses_job_and_revisit_model_without_sending_quality(self):
        page_job = {
            "page_number": 17,
            "image_path": str(self.image_path),
            "page_status": "needs_review",
            "source_stage": "transcription",
            "quality_context": [{"category": "low_contrast", "note": "Dim scan"}],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"candidates": [{"content": '
                    b'{"parts": [{"text": "Page text."}]}}]}'
                )

        captured_requests = []

        def fake_urlopen(request, timeout):
            captured_requests.append(request)
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "secret",
                "GEMINI_MODEL": "normal-model",
                "GEMINI_REVISIT_MODEL": "revisit-model",
            },
            clear=True,
        ):
            with patch(
                "active.gemini_api.urllib_request.urlopen",
                side_effect=fake_urlopen,
            ):
                transcript = transcribe_gemini_page(
                    page_job,
                    "Transcribe the page.",
                    revisit=True,
                )

        self.assertEqual(transcript, "Page text.")
        self.assertTrue(captured_requests[0].full_url.endswith("revisit-model:generateContent"))
        sent_body = json.loads(captured_requests[0].data)
        self.assertNotIn("low_contrast", json.dumps(sent_body))

    def test_transcribe_gemini_page_adds_page_number_to_provider_errors(self):
        page_job = {
            "page_number": 17,
            "image_path": str(Path(self.temp_directory.name) / "missing.png"),
            "page_status": "accepted",
            "source_stage": "submission",
        }
        client = build_gemini_client({
            "api_key": "key",
            "model": "gemini-test-model",
            "timeout_seconds": 10,
            "max_retries": 0,
        })

        with self.assertRaisesRegex(GeminiConfigurationError, "Page 17"):
            transcribe_gemini_page(page_job, "Transcribe the page.", client=client)
