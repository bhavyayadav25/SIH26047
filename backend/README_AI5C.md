# MediKiosk AI-5C — Doctor Workspace

AI-5C adds a read-only clinician workspace aggregation layer on top of AI-5B.

## Endpoint

`GET /api/doctor/encounters/{encounter_id}/workspace`

Requires an authenticated `doctor`, `triage`, or `admin` session.

The response assembles the existing encounter, patient snapshot, documents,
consultations, timeline, AI-4A through AI-4H outputs, and verification status.
It does not create diagnoses, prescriptions, treatment recommendations, or
clinical decisions.

## Safety

- Unverified document data remains explicitly unverified.
- AI output is presented as decision support, not as a clinician decision.
- The workspace endpoint is read-only with respect to clinical records.
- Viewing the workspace creates an audit event.
