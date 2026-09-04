# MediKiosk Phase AI-2 — Clinical Safety & Red-Flag Intelligence

Backend-only upgrade built on AI-1A through AI-1F.

## Purpose
AI-2 adds a conservative, explainable triage-support layer. It detects a limited set of explicit warning patterns in patient-reported text and returns a human-review action.

**It is not a diagnostic system and must be clinically validated before real-world use.**

## New endpoint
`POST /api/ai/safety-evaluate`

Example body:
```json
{
  "answer": "I have chest pressure and shortness of breath.",
  "structured": {}
}
```

The endpoint returns `risk_level`, `alerts`, `recommended_action`, `interrupt_interview`, evidence, and a safety disclaimer.

## Integrated behavior
`/api/interview/answer` now evaluates the answer through the AI-2 safety layer and includes a `safety` object in the response. Emergency-level patterns take priority over routine adaptive questioning.

## Design rules
- No diagnosis is produced.
- No medication/treatment instruction is produced.
- Patient-reported negatives are guarded with a simple negation window.
- Alerts retain evidence and a deterministic rule rationale.
- Emergency patterns request human triage instead of continuing the normal interview.
- The existing legacy safety function remains for backward compatibility, but new interview handling uses `ai2_safety.evaluate_safety`.

## Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Test
```bash
python -m py_compile main.py ai2_safety.py
```

Before any clinical deployment, have the complete rule set reviewed and approved by qualified clinical/safety stakeholders and test it against a validated dataset.
