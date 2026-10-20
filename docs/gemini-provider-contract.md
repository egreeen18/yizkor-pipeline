# Gemini provider contract

## Purpose

This document records the project’s default Gemini operating policy for OCR transcription. It exists to keep model choice, retry behavior, and revisit rules stable across runs.

It is a provider-layer contract only. It does not replace the workflow rules in the main project docs.

## Source of truth

The provider contract follows the project’s existing rules:

- the manifest is the operational record
- quality notes are the evidence record
- each page has one final workflow state
- skipped pages remain in the manifest
- review and retry happen after each stage where quality matters

## Default model policy

- Test mode: Gemini 2.5 Flash Lite
- Normal run: Gemini 2.5 Flash
- Revisit / escalation: Gemini 2.5 Pro

This is the project’s default routing policy.

The provider applies this routing when it builds a client for a page job. If the workflow supplies an already-built client, that client’s model is used as an explicit override:

- an explicit revisit request uses GEMINI_REVISIT_MODEL, defaulting to Gemini 2.5 Pro
- otherwise, a truthy GEMINI_TEST_MODE uses Gemini 2.5 Flash Lite
- otherwise, GEMINI_MODEL selects the normal model, defaulting to Gemini 2.5 Flash

GEMINI_TEST_MODE accepts true/false, 1/0, yes/no, and on/off, case-insensitively. An explicit revisit request takes precedence over test mode. GEMINI_MODEL is a normal-run override and does not override test or revisit routing.

## Revisit policy

A page is eligible for a revisit when it is in needs_review or when its transcript is weak, incomplete, empty, or otherwise not trustworthy enough to continue.

The revisit procedure is:

1. read the page state from the manifest
2. confirm the page is eligible for a second-pass run
3. preserve existing evidence in quality_notes.json
4. rerun OCR only for that page
5. compare the new result to the earlier output and source image
6. record new findings and confidence
7. either accept, keep in review, or reject the result

A revisit must not silently overwrite earlier evidence.

## Configuration

Required environment variables:

- GEMINI_MODEL
- GEMINI_TIMEOUT_SECONDS
- GEMINI_MAX_RETRIES

Credential:

- On Windows, the Gemini API key can be stored in Windows Credential Manager
  with `python -m active.store_gemini_api_key`.
- At runtime, a non-empty `GEMINI_API_KEY` environment variable takes
  precedence; otherwise, the provider reads the saved Windows credential.
- The key is not stored in project files. Install `requirements.txt` first to
  enable Credential Manager integration.

Optional:

- GEMINI_TEST_MODE
- GEMINI_REVISIT_MODEL

Defaults:

- default model: gemini-2.5-flash
- test model: gemini-2.5-flash-lite
- revisit model: gemini-2.5-pro
- timeout: 60 seconds
- max retries: 3

## Provider boundary

The provider module is responsible only for the Gemini boundary.

It must:

- load and validate credentials
- build the API request
- send the request with retries and timeouts
- normalize the response
- raise project-readable errors

It must not:

- decide page state
- rewrite the manifest
- decide whether a page is skipped or accepted
- manage the overall pipeline

Those decisions belong to the transcription and verification stages.

## Failure handling

Provider failures fall into three buckets:

- configuration failure: missing key, invalid value, unsupported config
- transient request failure: timeout, rate limit, temporary network problem
- content failure: empty output, unusable output, output inconsistent with the page

Only configuration failures are necessarily fatal. Request failures may be retried. Content failures become quality findings and are handled through the project’s review logic.

## Data contract

The transcription stage passes a minimal page-job mapping to `transcribe_gemini_page`:

- page_number (the manifest's PDF scan index)
- scan_page_number (the same authoritative scan index)
- image_path
- page_status
- source_stage
- optional quality_context, as a list of structured issue records

The workflow stage explicitly indicates a revisit when building the provider client or calling the page entry point. The provider does not infer revisit eligibility from page state or quality context. Page metadata stays available at the adapter boundary for traceability; only the image and transcription prompt are sent to Gemini. Quality evidence remains in quality_notes.json and is not copied into the model prompt.

The provider returns the model's raw response text. The workflow asks Gemini
for JSON containing an ordered `regions` array. Each region records a visual
position, an optional printed folio, an optional language label, and the
verbatim transcription. The workflow validates that response and attaches the
scan index from the manifest; the model does not supply scan identity.
Malformed JSON or an invalid region schema is classified as a content failure
so the existing quality-review handoff can record it.

The normalized workflow result has this shape:

```json
{
  "scan_page_number": 12,
  "regions": [
    {
      "position": "right",
      "printed_page_number": "11",
      "language": "Hebrew",
      "transcription": "..."
    }
  ]
}
```

A scan may contain one printed page, a spread, several text blocks, or an
unusual layout. Region count, language, and narrative relationship must not be
assumed from position. An empty region list is retained as a result but blocks
progress for manual verification. Transcription artifacts are UTF-8 JSON files
named by scan index; revisits use numbered JSON attempts and do not overwrite
earlier results.

The provider does not write the final transcript artifact or update the manifest.

## Final rule

For this project, the default strategy is:

- use Flash Lite for smoke tests and cheap broad passes
- use Flash for normal work
- use Pro only on revisits or difficult pages that need a second shot

This keeps the project usable within the rate limits while still allowing selective high-quality review when needed.
