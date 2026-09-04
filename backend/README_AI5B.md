# MediKiosk AI-5B — Encounter & Queue Workflow

AI-5B adds the operational encounter/OPD queue layer on top of the cumulative AI-1 through AI-4 backend and AI-5A integration audit.

## What it provides
- Encounter record as the operational unit for a patient visit.
- Daily department-scoped token numbers with a database uniqueness guarantee.
- Explicit queue priority: `normal`, `urgent`, `emergency` (never inferred by AI).
- Queue statuses: `waiting`, `called`, `in_consultation`, `completed`, `cancelled`.
- Controlled status transitions.
- Patient self-service encounter creation for their own account.
- Doctor/triage/admin queue management.
- Audit logging for encounter creation and status changes.
- No clinical diagnosis or autonomous triage decision is performed by this module.

## Endpoints
- `POST /api/encounters`
- `GET /api/encounters/{encounter_id}`
- `GET /api/queue?department=General%20Medicine&visit_date=YYYY-MM-DD`
- `POST /api/encounters/{encounter_id}/status`

## Safety boundary
Priority is supplied explicitly by an authorized workflow actor and is not generated from symptoms. `emergency` therefore means an explicitly marked queue priority requiring human handling; it is not an AI medical diagnosis.

## Verification performed
- Full pytest suite: 73 passed.
- Python compilation: passed.
- FastAPI startup: passed.
- `/api/health`: HTTP 200.
- `/openapi.json`: HTTP 200.
- Integration audit: PASS, no failed checks, AI-5B present.
