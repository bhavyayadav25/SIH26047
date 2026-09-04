# MediKiosk Unified Backend — AI-1 + AI-2 + AI-3A + AI-3C

This package consolidates the supplied AI-1F backend, AI-2 safety layer, AI-3A document-intake module, and AI-3C document-classification layer into **one FastAPI application**.

## Important

There is now **one and only one `main.py`**. Do not copy another phase's `main.py` over it.

### Included
- AI-1A–1F cumulative interview/orchestration functionality from the supplied AI-1F backend
- AI-2 safety/red-flag engine
- AI-3A document intake/storage foundation
- Existing authentication, database, doctor/ABDM/AYUSH routes from the supplied AI-2 backend
- Existing voice endpoints from the supplied cumulative backend

## Architecture

```text
main.py
  ├── AI-1: interview / clinical NLU / repair / adaptive orchestration / voice
  ├── AI-2: safety evaluation
  ├── AI-3A: document intake
  └── AI-3C: explainable document classification
```

Run only one server:

```bash
uvicorn main:app --reload
```

## AI-3A routes

AI-3A is intentionally namespaced to avoid colliding with the legacy document routes already present in the cumulative backend:

```text
POST   /api/documents/intake/upload
GET    /api/documents/intake/{document_id}
GET    /api/documents/intake/encounter/{encounter_id}
DELETE /api/documents/intake/{document_id}
```

The existing `/api/documents/upload` route from the cumulative backend remains available. It was **not deleted** in this integration, so existing frontend work is less likely to break.

## Why AI-3A is namespaced for now

The supplied AI-1/AI-2 backend already contains a `MedicalDocument` model and a legacy `/api/documents/upload` implementation. AI-3A uses a newer modular intake service. Rather than silently replacing the existing document pipeline, this integration keeps both boundaries visible until AI-3B/3C/3D are ready to replace/merge document processing deliberately.

## Setup

Create/activate your backend virtual environment and run:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Verification

```bash
python -m py_compile main.py ai1f_orchestrator.py clinical_nlu.py conversation_repair.py ai2_safety.py ai3a/*.py
pytest -q test_ai1f.py
pytest -q test_ai3a_standalone.py
```

The standalone AI-3A tests exercise the module in isolation; the actual unified server registration is in `main.py`.

## GitHub merge guidance

Use this package as the backend integration point. Do **not** merge phase ZIPs by overwriting files. Commit this unified structure first, then build AI-3B on top of it.

Recommended branch sequence:

```text
backend-unification
        ↓
AI-3B OCR
        ↓
AI-3C classification
        ↓
AI-3D extraction
```

## AI-3C routes

The existing `/api/documents/upload` pipeline now automatically runs AI-3C after text extraction. Additional routes are:

```text
POST /api/documents/classify
POST /api/documents/{document_id}/classify
```

AI-3C is local and explainable; it returns document class, confidence, evidence, per-class scores, and a `needs_review` flag. It does not diagnose the patient. See `README_AI3C.md` for details.

## AI-3E — Medical Document Verification
AI-3E adds human-in-the-loop verification for AI-3D extraction. Use `GET /api/documents/{document_id}/verification` to inspect the review queue and `POST /api/documents/{document_id}/verify` to explicitly confirm/correct extracted items. Verification status and audit metadata are stored on the document.


## AI-4B — Clinical Risk / Red-Flag Intelligence

The unified backend now includes conservative clinician-facing risk assessment via `GET /api/patients/{patient_id}/risk-assessment`. It uses the existing transparent AI-2 safety rules over current intake/profile data and verified document findings only. It does not diagnose, prescribe, or claim that absence of an alert means a patient is safe.
