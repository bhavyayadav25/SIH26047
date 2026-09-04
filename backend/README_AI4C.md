# MediKiosk AI-4C — Clinical Decision Support

AI-4C is a conservative clinician-facing decision-support layer built on AI-4A Clinical Summary and AI-4B Risk/Red-Flag Intelligence.

## Purpose
- Surface review prompts and questions for the responsible clinician.
- Highlight possible record inconsistencies without resolving them automatically.
- Keep verified document evidence traceable.
- Never diagnose, prescribe, select treatment, or make an autonomous clinical decision.

## Endpoint
`GET /api/patients/{patient_id}/decision-support`

Access: authenticated `doctor`, `triage`, or `admin` roles.

## Safety boundary
AI-4C does not infer missing facts. Unverified document values are not used as decision evidence. Emergency/urgent risk from AI-4B is surfaced as a human-review/triage prompt. All clinical decisions remain with the responsible clinician and local protocol.

## Validation
- AI-4C unit tests: 4/4 passed
- Combined core regression suite (AI-1F, AI-3C through AI-3H, AI-4A through AI-4C): 37/37 passed
- Python compilation: passed
- FastAPI startup: passed
- `/api/health`: HTTP 200
- `/openapi.json`: HTTP 200
- AI-4C route present in OpenAPI: yes

The legacy `test_ai3a_standalone.py` contains three tests for the old `/api/documents/upload` contract. AI-3A is intentionally namespaced in the unified backend at `/api/documents/intake/...` to avoid colliding with the unified upload pipeline. Those three legacy standalone tests are not counted in the core regression total.
