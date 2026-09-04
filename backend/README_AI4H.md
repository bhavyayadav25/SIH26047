# AI-4H — Final Clinical Intelligence & Safety Gate

AI-4H is the final orchestration/safety layer over AI-4A through AI-4G. It produces a clinician-facing readiness package and never diagnoses, prescribes, changes treatment, or writes clinical decisions.

## Endpoint

`GET /api/doctor/consultations/{consultation_id}/clinical-gate`

Roles: `doctor`, `triage`, `admin`.

## Dispositions

- `immediate_triage`: an AI-4B emergency-level signal interrupts routine flow.
- `verification_required`: extracted document items remain unverified.
- `clinician_review`: no blocking condition is present; advisory AI outputs still require clinician judgment.

A non-blocking result is **not** a statement that the patient is safe.
