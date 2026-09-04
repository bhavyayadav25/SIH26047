# MediKiosk AI-3C — Medical Document Classification

AI-3C is integrated into the existing unified FastAPI backend. It sits after text extraction/OCR and before the later extraction/normalisation phase.

## What it does

- Classifies extracted medical-document text into:
  - Prescription
  - Lab Report
  - Discharge Summary
  - Imaging Report
  - Consultation Note
  - Referral Letter
  - Operative Report
  - Vaccination Record
  - Other
- Returns confidence, evidence, per-class scores, method and `needs_review`.
- Uses a deterministic local classifier so the prototype works without an external API or model download.
- Is conservative: weak/ambiguous text is routed to `Other`/review rather than being presented as a confident classification.

## Integrated flow

```text
Upload document
   ↓
Existing text extraction / OCR
   ↓
AI-3C classification
   ↓
Existing clinical finding extraction
   ↓
Stored in MedicalDocument
   ↓
Doctor workspace can see classification metadata
```

## Routes

Existing upload route now automatically classifies:

`POST /api/documents/upload`

New testing/utility routes:

`POST /api/documents/classify`

Body:

```json
{"text":"LABORATORY REPORT ...","filename":"report.pdf","requested_type":"Other"}
```

Reclassify a stored document:

`POST /api/documents/{document_id}/classify`

## Database compatibility

The integration adds AI-3C fields to the existing `medical_documents` SQLite table through a lightweight startup migration. Existing databases are preserved.

## Verification

```bash
python -m py_compile main.py ai3c_document_classifier.py
pytest -q test_ai3c.py
pytest -q test_ai1f.py test_ai3a_standalone.py test_ai3c.py
```

Run the unified backend normally:

```bash
uvicorn main:app --reload
```

No second `main.py` is introduced.
