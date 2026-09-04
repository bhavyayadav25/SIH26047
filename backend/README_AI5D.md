# MediKiosk AI-5D — Actual Consultation

AI-5D adds the clinician-owned consultation workflow to the unified SIH26047 backend.

## Endpoints

- `POST /api/doctor/encounters/{encounter_id}/consultation` — start/retrieve the draft for an encounter
- `GET /api/doctor/consultations/{consultation_id}/record` — read the structured consultation record
- `PUT /api/doctor/consultations/{consultation_id}/record` — save clinician-entered draft sections
- `POST /api/doctor/consultations/{consultation_id}/complete` — complete and lock the consultation

## Consultation sections

`history`, `examination`, `assessment`, `diagnosis`, `plan`, `prescription`, `follow_up`

All clinical decision fields are clinician-entered. AI-5D does not generate, infer, validate, or modify diagnoses, treatment plans, doses, prescriptions, or follow-up decisions.

Completed consultations are locked in this prototype. Every write is recorded in the existing audit log.

## Safety boundary

AI-5D is a workflow/data-entry layer. AI-3 and AI-4 outputs remain available through the existing endpoints, but they are not silently copied into the doctor's final clinical decision fields.
