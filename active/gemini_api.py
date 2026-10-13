"""Gemini OCR provider adapter.

This file is intentionally limited to the provider boundary:
- configuration and validation
- client construction
- request payload creation
- request execution and error classification

The transcription workflow remains in the transcription stage file.
"""

import base64
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL_ENV_VAR = "GEMINI_MODEL"
DEFAULT_GEMINI_TIMEOUT_ENV_VAR = "GEMINI_TIMEOUT_SECONDS"
DEFAULT_GEMINI_MAX_RETRIES_ENV_VAR = "GEMINI_MAX_RETRIES"
DEFAULT_GEMINI_TEST_MODE_ENV_VAR = "GEMINI_TEST_MODE"
DEFAULT_GEMINI_REVISIT_MODEL_ENV_VAR = "GEMINI_REVISIT_MODEL"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TEST_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_REVISIT_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 60
DEFAULT_GEMINI_MAX_RETRIES = 3
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# Section 1: Configuration and credential handling
# ---------------------------------------------------------------------------


def _get_env_value(name: str, *, default: str | None = None) -> str | None:
    """Return a trimmed environment value or the default when set."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _get_bool_env_value(name: str, *, default: bool = False) -> bool:
    """Read a common boolean environment value or reject an unsupported value."""
    raw_value = _get_env_value(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.casefold()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise GeminiConfigurationError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off"
    )


def _get_int_env_value(name: str, *, default: int, minimum: int = 1) -> int:
    """Return a validated integer environment value."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value.strip())
    except (TypeError, ValueError) as error:
        raise GeminiConfigurationError(f"{name} must be an integer") from error

    if value < minimum:
        raise GeminiConfigurationError(
            f"{name} must be greater than or equal to {minimum}"
        )

    return value


def get_gemini_api_key() -> str:
    """Return the configured Gemini API key.

    The project docs do not specify a custom environment variable name, so the
    conventional and provider-neutral default is GEMINI_API_KEY. This keeps the
    API key out of the workflow logic and makes it easy to override without
    changing the transcription stage contract.
    """
    api_key = _get_env_value(DEFAULT_GEMINI_API_KEY_ENV_VAR)
    if not api_key:
        raise GeminiConfigurationError(
            "Gemini API key is not configured. Set the GEMINI_API_KEY environment variable."
        )
    return api_key


def get_gemini_config(*, revisit: bool = False) -> dict[str, object]:
    """Return the provider configuration used by the Gemini API wrapper.

    This is intentionally limited to configuration and validation only. It does
    not perform network calls or make page-state decisions. Revisit routing is
    selected explicitly by the workflow caller; test mode selects the low-cost
    test model. GEMINI_MODEL overrides the normal-run model only.
    """
    if not isinstance(revisit, bool):
        raise GeminiConfigurationError("revisit must be a boolean")

    if revisit:
        model = _get_env_value(
            DEFAULT_GEMINI_REVISIT_MODEL_ENV_VAR,
            default=DEFAULT_GEMINI_REVISIT_MODEL,
        ) or DEFAULT_GEMINI_REVISIT_MODEL
    elif _get_bool_env_value(DEFAULT_GEMINI_TEST_MODE_ENV_VAR):
        model = DEFAULT_GEMINI_TEST_MODEL
    else:
        model = _get_env_value(
            DEFAULT_GEMINI_MODEL_ENV_VAR,
            default=DEFAULT_GEMINI_MODEL,
        ) or DEFAULT_GEMINI_MODEL

    return {
        "api_key": get_gemini_api_key(),
        "model": model,
        "timeout_seconds": _get_int_env_value(
            DEFAULT_GEMINI_TIMEOUT_ENV_VAR,
            default=DEFAULT_GEMINI_TIMEOUT_SECONDS,
        ),
        "max_retries": _get_int_env_value(
            DEFAULT_GEMINI_MAX_RETRIES_ENV_VAR,
            default=DEFAULT_GEMINI_MAX_RETRIES,
        ),
    }


# ---------------------------------------------------------------------------
# Section 2: Client construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeminiClient:
    """Minimal provider client container for Gemini OCR requests."""

    api_key: str
    model: str
    timeout_seconds: int
    max_retries: int
    base_url: str = DEFAULT_GEMINI_BASE_URL

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @property
    def generate_content_url(self) -> str:
        return f"{self.base_url}/{self.model}:generateContent"


def build_gemini_client(config: dict[str, object] | None = None) -> GeminiClient:
    """Construct a Gemini client configuration object for later API calls.

    This is the minimal provider-boundary assumption for section 2: a tiny client
    object that holds the endpoint, auth headers, and default request settings.
    It does not read the manifest or choose page states.
    """
    resolved_config = get_gemini_config() if config is None else config
    api_key = resolved_config.get("api_key")
    model = resolved_config.get("model")
    timeout_seconds = resolved_config.get("timeout_seconds")
    max_retries = resolved_config.get("max_retries")

    if not isinstance(api_key, str) or not api_key.strip():
        raise GeminiConfigurationError("config[api_key] must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise GeminiConfigurationError("config[model] must be a non-empty string")
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise GeminiConfigurationError("config[timeout_seconds] must be an integer >= 1")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise GeminiConfigurationError("config[max_retries] must be an integer >= 0")

    return GeminiClient(
        api_key=api_key.strip(),
        model=model.strip(),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Section 3: Request payload construction
# ---------------------------------------------------------------------------

def _infer_image_mime_type(image_path: str) -> str:
    """Infer a Gemini-compatible MIME type from a page image extension."""
    extension = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mime_types.get(extension, "image/png")


def build_gemini_ocr_request(
    image_path: str,
    prompt: str,
    *,
    mime_type: str | None = None,
) -> dict[str, object]:
    """Build the payload for a Gemini OCR request.

    This is intentionally limited to the provider-boundary contract for section 3:
    an image is included as inline base64 data and the transcription instruction is
    attached as a text part. The workflow layer remains responsible for choosing
    the page and deciding whether the result is acceptable.
    """
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("image_path must be a non-empty string")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not os.path.isfile(image_path):
        raise ValueError(f"image not found: {image_path}")

    resolved_mime_type = mime_type or _infer_image_mime_type(image_path)
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": resolved_mime_type,
                        "data": encoded_image,
                    }
                },
            ]
        }]
    }


# ---------------------------------------------------------------------------
# Section 4: API call execution
# ---------------------------------------------------------------------------

def send_gemini_request(
    client: GeminiClient,
    payload: dict[str, object],
) -> dict[str, object]:
    """Send a Gemini request and return the parsed JSON response.

    This is the provider-layer execution step for section 4. It handles only the
    HTTP boundary: request sending, timeout/retry handling, and error
    classification. It does not inspect page state or decide workflow actions.
    """
    if not isinstance(client, GeminiClient):
        raise GeminiConfigurationError("client must be a GeminiClient instance")
    if not isinstance(payload, dict) or not payload:
        raise GeminiConfigurationError("payload must be a non-empty dictionary")

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        client.generate_content_url,
        data=request_body,
        headers=client.headers,
        method="POST",
    )

    retryable_status_codes = {408, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(client.max_retries + 1):
        try:
            with urllib_request.urlopen(request, timeout=client.timeout_seconds) as response:
                response_body = response.read()
                if not response_body:
                    raise ValueError("Gemini returned an empty response body")
                parsed = json.loads(response_body.decode("utf-8"))
                return parsed
        except urllib_error.HTTPError as error:
            last_error = error
            try:
                response_body = error.read().decode("utf-8", errors="replace")
            except (AttributeError, ValueError):
                response_body = ""
                fp = getattr(error, "fp", None)
                if fp is not None and hasattr(fp, "read"):
                    try:
                        if hasattr(fp, "seek"):
                            fp.seek(0)
                        response_body = fp.read().decode("utf-8", errors="replace")
                    except Exception:
                        response_body = ""
            try:
                parsed = json.loads(response_body) if response_body else {}
            except json.JSONDecodeError:
                parsed = {}
            error_message = _get_gemini_http_error_message(parsed, response_body)

            if error.code in {400, 422}:
                raise GeminiConfigurationError(
                    f"Malformed Gemini request: {error_message}"
                ) from error
            if error.code in {401, 403}:
                raise GeminiConfigurationError(
                    f"Gemini API access denied: {error_message}"
                ) from error
            if error.code in retryable_status_codes and attempt < client.max_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            error_type = (
                GeminiRequestError
                if error.code in retryable_status_codes
                else GeminiConfigurationError
            )
            raise error_type(
                f"Gemini request failed: {error_message}"
            ) from error
        except urllib_error.URLError as error:
            last_error = error
            if attempt < client.max_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise GeminiRequestError(
                f"Gemini request failed due to transport error: {error}"
            ) from error
        except json.JSONDecodeError as error:
            last_error = error
            if attempt < client.max_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise GeminiRequestError("Gemini returned malformed JSON") from error
        except ValueError as error:
            last_error = error
            if attempt < client.max_retries:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise GeminiRequestError(str(error)) from error

    if last_error is not None:
        raise GeminiRequestError(f"Gemini request did not complete successfully: {last_error}")
    raise GeminiRequestError("Gemini request did not complete successfully")


def _get_gemini_http_error_message(parsed: object, response_body: str) -> str:
    """Safely extract a provider message from an HTTP error response."""
    if isinstance(parsed, dict):
        error_details = parsed.get("error")
        if isinstance(error_details, dict):
            message = error_details.get("message")
            if isinstance(message, str) and message:
                return message
    return response_body or "No error details were provided"


# ---------------------------------------------------------------------------
# Section 5: Response parsing
# ---------------------------------------------------------------------------

def parse_gemini_response(response: dict[str, object]) -> str:
    """Extract and validate transcript text from a Gemini response."""
    if not isinstance(response, dict):
        raise TypeError("response must be a dictionary")

    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiContentError("Gemini response contains no candidates or transcript text")

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise GeminiContentError("Gemini response candidate is malformed")

    content = candidate.get("content")
    if not isinstance(content, dict):
        raise GeminiContentError("Gemini response candidate contains no transcript text")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise GeminiContentError("Gemini response candidate contains no transcript text")

    text_parts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    transcript = "".join(text_parts).strip()
    if not transcript:
        raise GeminiContentError("Gemini response contains no usable transcript text")

    return transcript


# ---------------------------------------------------------------------------
# Section 6: Error normalization
# ---------------------------------------------------------------------------

class GeminiProviderError(Exception):
    """Base error carrying provider category and retry guidance."""

    category: str
    retryable: bool

    def __init__(self, message: str, *, category: str, retryable: bool) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


class GeminiConfigurationError(GeminiProviderError, ValueError):
    """Fatal provider setup, authentication, or request-configuration failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="configuration_failure", retryable=False)


class GeminiRequestError(GeminiProviderError, RuntimeError):
    """Transient transport or provider request failure that can be retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="transient_request_failure", retryable=True)


class GeminiContentError(GeminiProviderError, ValueError):
    """Missing or unusable model content for workflow-level review handling."""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="content_failure", retryable=False)


def normalize_gemini_error(
    error: Exception,
    *,
    category: str,
) -> GeminiProviderError:
    """Convert an exception into a categorized provider error."""
    error_types = {
        "configuration_failure": GeminiConfigurationError,
        "transient_request_failure": GeminiRequestError,
        "content_failure": GeminiContentError,
    }
    error_type = error_types.get(category)
    if error_type is None:
        raise ValueError(f"Unsupported Gemini error category: {category}")
    if isinstance(error, GeminiProviderError) and error.category == category:
        return error
    return error_type(str(error))


# ---------------------------------------------------------------------------
# Section 7: Provider boundary contract
# ---------------------------------------------------------------------------

def transcribe_gemini_image(
    image_path: str,
    prompt: str,
    *,
    client: GeminiClient | None = None,
) -> str:
    """Return transcript text for one image without managing workflow state."""
    resolved_client = build_gemini_client() if client is None else client
    try:
        payload = build_gemini_ocr_request(image_path, prompt)
    except (OSError, TypeError, ValueError) as error:
        raise normalize_gemini_error(
            error,
            category="configuration_failure",
        ) from error

    response = send_gemini_request(resolved_client, payload)
    return parse_gemini_response(response)


def transcribe_gemini_page(
    page_job: Mapping[str, object],
    prompt: str,
    *,
    client: GeminiClient | None = None,
    revisit: bool = False,
) -> str:
    """Transcribe a manifest page record without managing its workflow state.

    Page metadata and quality context remain available as structured input, but
    only the image and caller-provided prompt are sent to Gemini. A supplied
    client fixes the model; otherwise the explicit revisit flag selects routing.
    """
    if not isinstance(page_job, Mapping):
        raise GeminiConfigurationError("page_job must be a mapping")

    page_number = page_job.get("page_number")
    image_path = page_job.get("image_path")
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise GeminiConfigurationError("page_job[page_number] must be a positive integer")
    if not isinstance(image_path, str) or not image_path.strip():
        raise GeminiConfigurationError("page_job[image_path] must be a non-empty string")
    if not isinstance(page_job.get("page_status"), str):
        raise GeminiConfigurationError("page_job[page_status] must be a string")
    if not isinstance(page_job.get("source_stage"), str):
        raise GeminiConfigurationError("page_job[source_stage] must be a string")
    quality_context = page_job.get("quality_context")
    if quality_context is not None and (
        not isinstance(quality_context, list)
        or any(not isinstance(issue, Mapping) for issue in quality_context)
    ):
        raise GeminiConfigurationError(
            "page_job[quality_context] must be a list of mappings"
        )
    if not isinstance(revisit, bool):
        raise GeminiConfigurationError("revisit must be a boolean")

    resolved_client = client
    if resolved_client is None:
        resolved_client = build_gemini_client(get_gemini_config(revisit=revisit))

    try:
        return transcribe_gemini_image(image_path, prompt, client=resolved_client)
    except GeminiProviderError as error:
        page_error = normalize_gemini_error(
            RuntimeError(f"Page {page_number}: {error}"),
            category=error.category,
        )
        raise page_error from error
