# MediKiosk — Phase AI-1F Backend

## Goal
AI-1F connects AI-1A through AI-1E into one end-to-end adaptive interview turn.

## New endpoint
`POST /api/ai/orchestrate-turn`

Example request:
```json
{
  "patient_id": 1,
  "session_id": "demo-session",
  "language": "hi-IN",
  "question_id": "chief_complaint",
  "text": "Mere seene mein 3 din se pressure hai",
  "event": "answer",
  "attempt": 0,
  "input_mode": "voice"
}
```

## Response contract
The endpoint returns one frontend-safe `action`:
- `ask_question`
- `repeat_question`
- `simplify_question`
- `request_correction`
- `voice_retry`
- `touch_fallback`
- `switch_language`
- `triage_interrupt`
- `complete_interview`

Safety has priority over normal question selection. Repair actions do not alter clinical meaning.

## Architecture
1. AI-1E conversation repair
2. AI-1B clinical extraction
3. AI-1A server-owned encounter state
4. Existing red-flag rules
5. AI-1C adaptive next-question selection
6. One normalized orchestration response

## Testing
Run:
```bash
python -m py_compile main.py ai1f_orchestrator.py conversation_repair.py
PYTHONPATH=. python -m pytest -q test_ai1f.py
```

The tests cover safety priority, repair priority, question continuation, and interview completion.

## Safety boundary
AI-1F does not diagnose, prescribe, or assign a doctor. It coordinates history-taking workflow and hands potential urgent cases to the existing triage mechanism.
