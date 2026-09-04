# AI-5F — Hospital Administration

Integrated into the unified SIH26047 backend.

## Admin endpoints
- `GET/POST/PUT /api/admin/departments`
- `GET/POST/PUT /api/admin/doctors`
- `GET/POST /api/admin/opd-config`
- `GET/POST/PUT /api/admin/availability`
- `GET/POST/PUT /api/admin/routing`
- `GET/PUT /api/admin/hospital`

All AI-5F mutation/read endpoints require an authenticated `admin` role. Administrative changes are audit logged.

## Safety / integrity
- Department names are unique.
- Doctor availability windows reject invalid or reversed times.
- Duplicate availability is rejected.
- Department rename updates dependent OPD, doctor-profile, routing, and default-hospital references.
- AI-5F does not modify clinical decisions, prescriptions, diagnoses, or triage risk.

## Demo admin
- Email: `admin@sih26047.local`
- Password: `admin123`

Change demo credentials before any non-demo deployment.
