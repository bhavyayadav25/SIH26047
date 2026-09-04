import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from types import SimpleNamespace
from ai5c_doctor_workspace import build_doctor_workspace


def _patient():
    return SimpleNamespace(id=1, name="Test Patient", profile=SimpleNamespace(age=42, gender="Male", blood_group="O+", allergies="None recorded", conditions="", medications=""))


def _doc(i, status="Verified"):
    return SimpleNamespace(id=i, filename=f"doc{i}.pdf", document_type="Lab Report", classification="Lab Report", classification_confidence="0.95", classification_needs_review=0, verification_status=status, created_at=None, verified_at=None)


def _enc():
    return SimpleNamespace(id=10, department="General Medicine", visit_date="2026-09-01", token_number=3, priority="normal", status="in_consultation", reason="Cough", doctor_id=2)


def test_workspace_preserves_encounter_and_patient():
    ws = build_doctor_workspace(_patient(), _enc(), [_doc(1)], [], {"x": 1}, {"risk_level": "none"}, {}, {}, {}, {}, {}, [])
    assert ws["patient"]["name"] == "Test Patient"
    assert ws["encounter"]["token_number"] == 3
    assert ws["read_only"] is True


def test_unverified_documents_are_visible_but_flagged():
    ws = build_doctor_workspace(_patient(), _enc(), [_doc(1, "Pending")], [], {}, {"risk_level": "none"}, {}, {}, {}, {}, {}, [])
    assert ws["documents"][0]["verification_status"] == "Pending"
    assert ws["overview"]["pending_document_verifications"] == [1]
    assert ws["safety"]["unverified_document_data_used_as_verified_evidence"] is False


def test_safety_boundary():
    ws = build_doctor_workspace(_patient(), _enc(), [], [], {}, {}, {}, {}, {}, {}, {}, [])
    assert ws["safety"]["diagnosis_generated"] is False
    assert ws["safety"]["prescription_generated"] is False
    assert ws["safety"]["autonomous_clinical_decision"] is False
    assert ws["safety"]["clinician_decision_required"] is True
