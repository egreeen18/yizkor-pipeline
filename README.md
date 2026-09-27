# yizkor-pipeline

Yizkor books are local memorials and histories for Jewish communities destroyed in the Holocaust. Many of them contain important historical information, but they are often hard to access, incomplete, or untranslated.

This project is meant to help process a submitted Yizkor book through a simple, page-by-page pipeline: evaluate the source, transcribe it, verify the result, translate it, verify the translation, and assemble the final output.

## Core operating principles

1. Keep one submission record at the top level.
2. Make decisions at the page level.
3. Keep workflow state separate from evidence.
4. Save quality findings in a sidecar JSON file, not in the prompt or in the final output.
5. Review at the end of each stage, not only at the end of the whole process.
6. A page may have many quality flags, but it should end with one final workflow state.

## Page states

Each page is assigned one of these states:

- accepted: the page is usable and can continue
- needs_review: the page is still in play, but it needs closer attention or a retry
- rejected: the page should not continue until it is fixed or rechecked
- skipped: the page is intentionally excluded from LLM work and remains in the manifest for continuity

A skipped page is still part of the submission record. It is not silently removed or renumbered.

## Quality evidence model

The project keeps two layers of information:

- submission_manifest.json: operational state of the submission and pages
- quality_notes.json: detailed quality evidence, uncertainty, and risk notes

The manifest answers: what happened?
The quality notes answer: why did it happen and what should we do next?

## Review policy

Human review is not reserved only for the final answer. It is part of the pipeline logic after each stage where the output may be uncertain or low quality.

A page can be:

- accepted and continue
- retried and rechecked
- paused for human review
- excluded from the model run and marked as skipped

The decision depends on the stage and the risk level, not just on a single generic quality label.

## File layout

- active/submission.py: intake, page rendering, and initial page evaluation
- unstarted/transcribe.py: transcription stage
- unstarted/verify_transcription.py: transcription verification
- unstarted/translate.py: translation stage
- unstarted/verify_translation.py: translation verification
- unstarted/compile.py: final assembly
- unstarted/llm_client.py: model wrapper
- docs/pipeline-spec.md: detailed process specification
- docs/quality-evaluation-spec.md: quality policy and JSON contract

## High-level flow

1. Validate the PDF and render pages.
2. Record the page list and initial quality findings.
3. Transcribe the active pages.
4. Verify transcription quality and decide whether to continue, retry, or review.
5. Translate the accepted text.
6. Verify the translation and decide whether to continue, retry, or review.
7. Compile the final output.

## Simple decision rule

For each page, ask in order:

1. Is the page usable?
2. Is the page active or skipped?
3. Is the current stage output good enough to continue?
4. If not, should we retry, ask for review, or exclude the page?

This keeps the workflow simple and makes the code easier to reason about later.

The system is designed to be easy to understand, easy to pause and resume, and easy to debug when a stage produces bad output.

---

For the full process details, see [docs/pipeline-spec.md](docs/pipeline-spec.md) and [docs/quality-evaluation-spec.md](docs/quality-evaluation-spec.md).
