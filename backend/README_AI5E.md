# MediKiosk AI-5E — Triage Dashboard

AI-5E operationalizes the existing encounter queue for authorized clinical/administrative staff.

## Endpoints

- `GET /api/triage/dashboard?department=General%20Medicine&visit_date=YYYY-MM-DD`
- `POST /api/triage/encounters/{encounter_id}/action`

Action payload:

```json
{
  "action": "acknowledge | escalate | resolve",
  "notes": "optional human triage note",
  "priority": "optional: normal | urgent | emergency"
}
```

## Safety boundary

- AI-5E does not autonomously assign clinical priority.
- AI-4B risk signals are displayed as decision support.
- A missing AI alert does not mean a patient is clinically safe.
- Triage actions are explicit human actions and are audit logged.
- Closed/cancelled encounters cannot receive triage actions.

## Database migration

The unified backend automatically adds these optional columns to `encounters` when missing:

- `triage_status`
- `triage_notes`
- `triage_updated_by`
- `triage_updated_at`

## Verification

The complete regression suite was run after integration.
