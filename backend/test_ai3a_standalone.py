from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai3a.integration import register_ai3a

def make_app(tmp_path: Path):
    app = FastAPI()
    register_ai3a(app, str(tmp_path / "documents"))
    return app

def test_upload_pdf(tmp_path):
    app = make_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/documents/intake/upload",
        data={"patient_id": "p1", "encounter_id": "e1"},
        files={"document": ("report.pdf", b"%PDF-test", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document"]["status"] == "READY_FOR_OCR"
    assert body["next_stage"] == "OCR"

def test_reject_unsupported_type(tmp_path):
    app = make_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/documents/intake/upload",
        data={"patient_id": "p1", "encounter_id": "e1"},
        files={"document": ("script.exe", b"bad", "application/octet-stream")},
    )
    assert response.status_code == 400

def test_list_and_delete(tmp_path):
    app = make_app(tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/api/documents/intake/upload",
        data={"patient_id": "p1", "encounter_id": "e2"},
        files={"document": ("scan.png", b"png-test", "image/png")},
    )
    doc_id = upload.json()["document"]["document_id"]

    listed = client.get("/api/documents/intake/encounter/e2")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["document_id"] == doc_id

    deleted = client.delete(f"/api/documents/intake/{doc_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/documents/intake/{doc_id}").status_code == 404
