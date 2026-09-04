# MediKiosk AI-4D — Medication Intelligence

AI-4D is a conservative medication-reconciliation layer for doctor/triage/admin review.

## Endpoint
`GET /api/patients/{patient_id}/medication-intelligence`

Requires an authenticated doctor, triage, or admin session.

## What it does
- Consolidates patient-profile medication entries with **verified** medication entries from medical documents.
- Flags possible record duplication and reconciliation differences.
- Clearly separates patient-reported profile entries from verified document entries.
- Preserves source information where available.
- Surfaces missing medication history as an information gap.

## Safety boundaries
AI-4D does **not** prescribe, discontinue, change doses, choose treatments, or resolve discrepancies automatically. It does not perform drug-interaction or allergy matching because that requires a validated and maintained medication knowledge base; claiming such checks without one would be unsafe.

A discrepancy means "confirm/reconcile," not "wrong medication." A medicine absent from one record is not treated as proof that it was stopped.

## Validation performed
- AI-4D tests: 5/5 passed
- Core regression suite (AI-1F, AI-3C through AI-3H, AI-4A through AI-4D): 42/42 passed
- Python compilation: passed
- FastAPI import/startup: passed
- `/api/health`: 200
- `/openapi.json`: 200
- AI-4D route registration: passed
- Unauthorized AI-4D request: 401
- ZIP integrity: verified after packaging

## Prototype limitation
Medication normalization is deliberately conservative and does not map brand names to generic ingredients. Real-world deployment requires a validated medication database, interaction/allergy knowledge base, clinical validation, and appropriate governance.
