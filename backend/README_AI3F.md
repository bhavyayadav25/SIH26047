# AI-3F — Longitudinal Clinical Timeline

AI-3F consolidates existing consultations and medical documents into a chronological, traceable timeline for practitioner review.

## Design guarantees
- Uses stored record timestamps only; it does not invent dates.
- Keeps document verification status attached to document-derived information.
- Preserves source type and source ID for traceability.
- Does not diagnose, recommend treatment, or silently convert unverified extraction into verified fact.
- Works on the existing AI-1F → AI-3E unified backend.

## Endpoints

### `GET /api/patients/{patient_id}/timeline`
Returns the patient's consolidated clinical timeline.

### `GET /api/documents/{document_id}/timeline-context`
Returns the full patient timeline with the selected document's patient context.

## Event types
- `consultation`
- `document`

Each event contains `event_type`, `occurred_at`, `title`, `summary`, `source`, `source_id`, `verification_status`, and `details`.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q test_ai1f.py test_ai3c.py test_ai3d.py test_ai3e.py test_ai3f.py
uvicorn main:app --reload
```

The existing standalone AI-3A tests are intentionally not included in this regression command because they target a different isolated router contract than the unified backend.
