# MediKiosk — AI-3A Backend
## Medical Document Intake & Processing Foundation

AI-3A is intentionally limited to the document-ingestion boundary.

### Included

- PDF / PNG / JPG / JPEG validation
- 15 MB size limit
- Document ID generation
- Encounter + patient association
- Processing status lifecycle
- Secure server-side storage boundary
- Document metadata endpoint
- Encounter document listing
- Document deletion
- SHA-256 helper for integrity/audit use
- Tests

### Not included yet

- OCR
- Handwriting recognition
- Document classification
- Medical entity extraction
- Diagnosis inference
- Medication interpretation
- Lab interpretation
- Timeline generation

Those belong to AI-3B onward.

## Integration

Copy the `ai3a/` directory into your existing backend.

Install:

```bash
pip install fastapi python-multipart
```

In your existing `main.py`:

```python
from ai3a.integration import register_ai3a

register_ai3a(app, storage_dir="./data/documents")
```

Do NOT replace your existing `main.py`.

## API

### Upload

`POST /api/documents/upload`

Multipart form:

- `patient_id`
- `encounter_id`
- `document`

Supported:
- PDF
- PNG
- JPG/JPEG

### Get

`GET /api/documents/{document_id}`

### List encounter documents

`GET /api/documents/encounter/{encounter_id}`

### Delete

`DELETE /api/documents/{document_id}`

## Status lifecycle

```text
UPLOADED → VALIDATING → QUEUED → PROCESSING → READY_FOR_OCR
                                      └──────→ FAILED
```

In this implementation the stages are recorded logically and the
document reaches `READY_FOR_OCR` immediately after successful storage.
A background queue can be introduced later without changing the API.

## Security note

Health documents are sensitive. The production implementation should put
authentication/authorization around every endpoint, use a protected
storage location, encrypt data at rest/in transit as appropriate, maintain
audit logs, and apply a documented retention/deletion policy.

AI-3A deliberately does not expose raw document bytes through a public
download endpoint.

## Run tests

```bash
pytest -q
```
