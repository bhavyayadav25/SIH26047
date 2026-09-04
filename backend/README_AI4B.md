# AI-4B — Clinical Risk / Red-Flag Intelligence

AI-4B is a conservative clinician-facing safety layer built on the existing
AI-2 transparent rule engine and the verified outputs of AI-3 / AI-4A.

## Endpoint

`GET /api/patients/{patient_id}/risk-assessment`

Requires an authenticated `doctor`, `triage`, or `admin` session.

## Safety contract

- Not a diagnostic engine.
- Does not prescribe or recommend treatment.
- Only verified document extraction can contribute document-derived alerts.
- Historical stored alerts remain traceable context and require clinician review.
- A lack of a rule match is explicitly **not** a claim that the patient is safe.
- Emergency-level prototype matches request immediate human triage.

## Pipeline

AI-3 verified handoff → AI-4A clinical summary → AI-4B risk/red-flag assessment → clinician review.
