# Quality Evaluation Specification

## 1. Purpose

This document defines how the project records quality problems, uncertainty, and page-level risk. The goal is simple: keep quality data structured, easy to read, and separate from the actual text output.

The quality system is not a prompt-editing system. It is a metadata layer that helps the pipeline decide what is usable, what needs review, what should be retried, and what should be excluded from LLM work.

---

## 2. Core design rules

1. Quality data is tracked at the page level.
2. Each page receives one final workflow state.
3. A page may have many issue flags, but only one final action state.
4. Quality findings are saved as JSON metadata, not mixed into the prompt or the final document.
5. Quality notes live in a separate sidecar file.
6. Skipped pages remain in the manifest with their original page numbers.
7. Review happens after meaningful stages, not only at the very end.

These rules are the foundation of the working policy.

---

## 3. Page-state policy

The pipeline makes operational decisions using a small page-state model.

- accepted: the page is usable and can continue
- needs_review: the page is still in play, but needs a closer look or retry
- rejected: the page should not continue without correction or a human decision
- skipped: the page is intentionally excluded from LLM work and remains in the record

This is the main decision model. It is separate from the issue category and severity.

### Important distinction

A page can be flagged as low contrast or potentially mistranslated, but that does not automatically mean it is rejected.

The state answers: what should we do next?
The flag answers: what was the problem?

This distinction keeps the workflow clear and prevents cluttered or vague decision logic.

---

## 4. Quality verification policy

The quality system must behave like a policy, not just a list of notes.

### Required procedural rules

1. Every page must have a page state before the next stage begins.
2. A skipped page must remain in the manifest with the original page number.
3. A page may have multiple issue flags, but it should still end with one final workflow state.
4. Each issue should include evidence, severity, confidence, and a recommended action.
5. A page should not be excluded without a recorded reason in quality_notes.json.
6. If a later stage updates the page status, earlier evidence should remain in the record.
7. Quality checks happen where the problem matters: source quality, transcription quality, or translation quality.
8. A human review gate should occur after a stage when the output is uncertain enough to affect trust.

This turns the quality system into an implementation policy rather than a loose note-taking system.

---

## 5. How the JSON files work together

The project uses two layers of information:

- submission_manifest.json: operational state of the submission and page decisions
- quality_notes.json: detailed evidence, flags, and reasoning

A good rule to remember is:

- manifest = state and workflow
- quality notes = evidence and explanation

### submission_manifest.json

This file should answer:

- what was submitted?
- what pages exist?
- which pages are active?
- which pages were skipped or rejected?
- what is the current status of the submission?

It is meant to be simple and operational.

### quality_notes.json

This file should answer:

- what issue was found?
- how strong is the evidence?
- how serious is the problem?
- what action should happen next?

This is where the detailed reasoning lives.

---

## 6. JSON examples

### Example: submission_manifest.json

```json
{
  "submission_id": "2026-09-27_001",
  "source_pdf": "C:/path/to/book.pdf",
  "created_at": "2026-09-27T12:00:00Z",
  "status": "in_progress",
  "overall_confidence": "medium",
  "pages": [
    {
      "page_number": 12,
      "image_path": "page_012.png",
      "page_status": "accepted",
      "flags": [
        {
          "category": "low_contrast",
          "severity": "warning",
          "confidence": "medium",
          "action": "review",
          "source_stage": "submission"
        }
      ]
    },
    {
      "page_number": 13,
      "image_path": "page_013.png",
      "page_status": "skipped",
      "skip_reason": "blank_page",
      "flags": [
        {
          "category": "blank_page",
          "severity": "blocking",
          "confidence": "high",
          "action": "exclude",
          "source_stage": "submission"
        }
      ]
    }
  ]
}
```

### Example: quality_notes.json

```json
{
  "submission_id": "2026-09-27_001",
  "generated_at": "2026-09-27T12:00:30Z",
  "pages": [
    {
      "page_number": 12,
      "stage": "submission",
      "issues": [
        {
          "category": "low_contrast",
          "severity": "warning",
          "confidence": "medium",
          "evidence": {
            "contrast_score": 0.41,
            "brightness_variance": 0.18
          },
          "action": "review",
          "note": "Low contrast may reduce transcription accuracy."
        }
      ]
    },
    {
      "page_number": 13,
      "stage": "submission",
      "issues": [
        {
          "category": "blank_page",
          "severity": "blocking",
          "confidence": "high",
          "evidence": {
            "blank_ratio": 0.98
          },
          "action": "exclude",
          "note": "Page appears blank and is not useful for LLM transcription."
        }
      ]
    }
  ]
}
```

### Quality flag fields

Each issue should include the following fields:

- category: the type of issue
- severity: informational / warning / blocking
- confidence: low / medium / high
- evidence: measurable or observed support for the flag
- action: accepted / review / exclude / manual_verify / retry
- source_stage: submission / transcription / translation / final_review
- note: short human-readable explanation

---

## 7. Quality category model

The project should classify issues by what they affect in the pipeline.

### State-to-category examples

These examples are meant to make the policy easier to apply in code.

- blank page -> skipped
- unreadable image -> rejected or skipped, depending on whether the page is usable at all
- low contrast but readable -> accepted or needs_review
- a transcript with obvious gaps -> needs_review or rejected
- a translated passage with likely meaning drift -> needs_review or manual_verify
- a page that is intentionally ignored because it is irrelevant -> skipped

This keeps the category and the state separate while still making the decision rules concrete.

### Source quality issues

These affect whether a page is usable at all.

- invalid PDF
- page rendering failure
- blank page
- duplication
- missing page order
- blurry or unreadable image
- low contrast
- skew or rotation
- bleed-through
- clipping at the edges
- heavy marks or noise

### Transcription risk issues

These affect whether the model can transcribe the page reliably.

- text may be unreadable
- structure may confuse the model
- uncertain or missing content
- repeated text or broken sequence
- suspicious OCR-like artifacts
- page is structurally unusual or hard to follow

### Translation risk issues

These affect whether the translated output is trustworthy.

- likely mistranslation
- ambiguous names, places, or terms
- wording drift from the source
- suspiciously smooth output hiding uncertainty
- missing or repeated content impacting interpretation

### Workflow-state issues

These describe the operational decision to continue, retry, skip, or review.

- continue normally
- retry or recheck
- review by human
- exclude from model work
- skip and retain in record

These are not the same as source-quality or content-quality issues. They are the decision layer built on top of the evidence.

---

## 8. Severity model

### Informational

A note worth recording but not materially affecting the workflow.

Examples:
- watermark
- decorative element
- unusual but readable formatting

### Warning

The page can continue, but with lower confidence or more risk.

Examples:
- low contrast
- minor skew
- slightly blurred content

### Blocking

The page or document should not continue automatically.

Examples:
- unreadable text
- major rendering failure
- content too compromised to trust

---

## 9. Confidence model

Confidence describes how certain the system is that the issue exists.

- low: weak or ambiguous evidence
- medium: noticeable issue, but not conclusive
- high: clear evidence and a consistent pattern

A low-confidence warning may still allow the page to proceed. A high-confidence blocking issue should trigger review or exclusion.

---

## 10. Action model

Each flag should suggest a next step. These actions are operational decisions, not just labels.

- accepted: continue normally
- review: continue with a note that the page deserves attention
- retry: re-run the stage or re-evaluate the output
- exclude: do not include this page automatically in downstream work
- manual_verify: require a human decision before continuing
- skipped: keep the page in the manifest but exclude it from model processing

This supports the real workflow without forcing every page into a single binary good/bad decision.

---

## 11. Where quality is checked

Quality checks happen at the stage where the risk matters.

### During submission

- validate the PDF
- render pages
- detect blank or unreadable pages
- set initial page-quality flags

### Before transcription

- review page readability
- estimate whether transcription is likely to be risky
- decide whether to proceed with caution, retry, or review

### After transcription

- check whether the result is complete and plausible
- flag gaps, repetition, or suspicious artifacts
- note where source quality likely affected the result

### After translation

- look for mistranslation or meaning drift
- flag ambiguous names, places, or terms
- record whether the page needs human review

### In the final writeup

- summarize the main risk areas
- identify pages that were difficult or compromised
- note what should be checked by a historian

---

## 12. Review and escalation policy

Human review should occur at the end of each meaningful stage when the output is low-confidence or risky.

This means the user can choose to:

- continue with caution
- retry the stage
- flag the page for manual review
- exclude the page from the run

This is more useful than a single end-of-pipeline review because it lets the project respond at the stage where the problem actually appears.

---

## 13. Final rule

A page may have many issues, but it should still end with one final page state.

The quality system is there to explain the risk, preserve evidence, and support a clear operational decision. It is not meant to replace the actual page-by-page workflow.

That is the key principle behind the whole design.

- low-quality pages may still continue if they are usable
- pages with serious issues may be rejected or sent to manual review
- the final writeup is the place for historian-facing interpretation

This keeps the system practical and easy to follow.

---

## 12. Why this helps later debugging

Because the system stores structured evidence in JSON, you can later answer questions like:

- which pages keep triggering warnings?
- did transcription risk get worse after a certain page number?
- were translation problems concentrated on pages with poor image quality?
- did a scanning issue correlate with final output errors?

This makes the project easier to improve over time without losing the reasoning behind earlier decisions.

---

## 13. Final recommendation

Keep the quality model simple and structured:

- page-level records
- JSON storage
- separate quality notes file
- severity + confidence + action
- final writeup reserved for historian-facing interpretation

This gives the project a useful quality layer without turning the workflow into a complicated prompt-engineering system.
