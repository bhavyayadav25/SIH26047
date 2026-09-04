# MediKiosk AI-5G — Analytics

AI-5G adds an administrator-only, read-only aggregate analytics endpoint to the cumulative SIH26047 backend.

## Endpoint

`GET /api/admin/analytics`

Optional query parameters:
- `start_date=YYYY-MM-DD`
- `end_date=YYYY-MM-DD`

## Metrics

- Encounter and unique-patient volume
- Assigned-doctor count
- Consultation volume
- Medical-document volume
- Active departments/doctors
- Queue status/priority/department distributions
- Recorded consultation risk-level counts
- Document verification status counts
- Average observed queue wait time (where timestamps exist)
- Average completed-consultation duration (where timestamps exist)
- Daily encounter volume
- Audit event count

The endpoint deliberately reports `ai_correction_rate` as unavailable because the current schema does not contain a normalized correction-event denominator. It does not invent a percentage.

## Safety boundary

AI-5G is operational analytics only. It does not diagnose, prescribe, recommend treatment, score individual patients, or make autonomous clinical decisions. It returns aggregate summaries and does not expose individual patient records through the analytics payload.

## Verification

The integrated backend regression suite passed **94 tests** after AI-5G integration. FastAPI startup, `/api/health`, `/openapi.json`, and AI-5G route registration were also verified.
