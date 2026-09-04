# AI-3H — Final Clinical Intake Handoff

AI-3H is the final phase of the AI-3 document pipeline. It packages existing, traceable information from AI-3B through AI-3G into a physician handoff and applies a readiness gate.

## Safety rules
- No diagnosis or treatment recommendation is generated.
- Missing facts are not inferred.
- Extracted document values are only presented as confirmed when AI-3E explicitly verified them.
- Unverified extraction or low-confidence document classification blocks a handoff from being marked ready.
- Source evidence and provenance remain available.

## Endpoints
- `GET /api/patients/{patient_id}/clinical-handoff`
- `GET /api/documents/{document_id}/clinical-handoff`

The endpoints are read-only and use the existing database records. No new schema migration is required.
