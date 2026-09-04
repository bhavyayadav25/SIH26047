import json
from types import SimpleNamespace
from datetime import datetime
from ai3h_clinical_handoff import build_clinical_handoff


def doc(verified=True, classification_confidence=0.95):
    items = [{"category":"medicine","label":"Paracetamol","value":"500 mg","verified":verified,"evidence":"Paracetamol 500 mg"}]
    extraction=json.dumps({"items":items})
    return SimpleNamespace(id=1, filename="rx.txt", document_type="Prescription", classification="Prescription",
        classification_confidence=classification_confidence, verification_status="Verified" if verified else "Pending",
        verified_data=extraction if verified else None, structured_extraction=extraction, created_at=datetime(2026,1,1))


def consultation():
    return SimpleNamespace(id=3, created_at=datetime(2026,1,2), title="Visit", summary="Headache", risk_level="none",
        red_flags="[]", doctor_review="Pending", ai_summary=None)


def test_ready_only_when_extraction_verified():
    out=build_clinical_handoff(10,[consultation()],[doc(True)])
    assert out["schema_version"] == "AI-3H.1"
    assert out["ready_for_physician_review"] is True
    assert out["documents"][0]["confirmed_items"][0]["verified"] is True


def test_unverified_items_block_handoff():
    out=build_clinical_handoff(10,[consultation()],[doc(False)])
    assert out["ready_for_physician_review"] is False
    assert out["blockers"]
    assert out["documents"][0]["pending_items"]


def test_low_classification_confidence_blocks_handoff():
    out=build_clinical_handoff(10,[],[doc(True,0.50)])
    assert out["ready_for_physician_review"] is False
    assert any(x["type"] == "document_classification" for x in out["blockers"])
