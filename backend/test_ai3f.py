from datetime import datetime, timedelta
from types import SimpleNamespace
from ai3f_clinical_timeline import build_timeline


def test_timeline_is_chronological_and_traceable():
    base = datetime(2026, 1, 1)
    consultation = SimpleNamespace(id=1, created_at=base + timedelta(days=2), title="Follow-up", summary="Review", risk_level="none", doctor_review="Verified", red_flags="[]", doctor_notes="", ai_summary=None)
    document = SimpleNamespace(id=4, created_at=base, filename="lab.txt", document_type="Lab Report", classification="Lab Report", extracted_text="Hb 12", verification_status="Verified", verified_data='{"items":[{"label":"Hemoglobin","value":"12","verified":true}]}', structured_extraction=None)
    out = build_timeline([consultation], [document])
    assert out["schema_version"] == "AI-3F.1"
    assert [e["source_id"] for e in out["events"]] == [4, 1]
    assert out["events"][0]["verification_status"] == "Verified"


def test_unverified_extraction_remains_labeled():
    doc = SimpleNamespace(id=7, created_at=datetime(2026, 2, 1), filename="rx.txt", document_type="Prescription", classification="Prescription", extracted_text="Paracetamol 500 mg", verification_status="Pending", verified_data=None, structured_extraction='{"items":[{"label":"Medicine","value":"Paracetamol 500 mg"}]}')
    out = build_timeline([], [doc])
    assert out["events"][0]["verification_status"] == "Pending"
    assert out["events"][0]["details"]["extracted_items"][0]["value"] == "Paracetamol 500 mg"
