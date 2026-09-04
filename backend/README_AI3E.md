# AI-3E — Medical Document Verification

AI-3E is the human-in-the-loop verification layer for AI-3D structured medical extraction.

## What it does
- Builds an item-level verification queue from AI-3D output.
- Shows confidence and OCR evidence for each extracted item.
- Requires explicit `verified=true` before an item becomes verified.
- Allows a reviewer to correct an extracted value while retaining the original AI evidence.
- Stores verification status, reviewer ID, notes and timestamp on the medical document.
- Never diagnoses, invents values, or silently changes extracted clinical information.

## Endpoints

### `GET /api/documents/{document_id}/verification`
Returns the verification queue and current verification state.

### `POST /api/documents/{document_id}/verify`
Body:
```json
{
  "verified_items": [
    {"index": 0, "verified": true},
    {"index": 1, "verified": true, "value": "Corrected value"}
  ],
  "notes": "Checked against original document",
  "verified_by": 2
}
```

## Pipeline

AI-3B OCR → AI-3C classification → AI-3D structured extraction → **AI-3E verification** → downstream clinical review.

## Validation

AI-3E + AI-3D + AI-3C + AI-1F regression tests: **18 passed**.
