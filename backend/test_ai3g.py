import json
from datetime import datetime
from types import SimpleNamespace
from ai3g_explainability import explain_document, explain_timeline


def doc(**kw):
    base = dict(id=7, filename="lab.txt", document_type="Lab Report",
                classification="Lab Report", classification_confidence="0.93",
                classification_method="rule", classification_evidence='["lab"]',
                classification_needs_review=0,
                structured_extraction=json.dumps({"items":[{"category":"lab","label":"Hemoglobin","value":"13.2","confidence":0.91,"evidence":"Hemoglobin 13.2 g/dL","needs_review":False}]}),
                verified_data=None, verification_status="Pending", verified_by=None, verified_at=None, verification_notes=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_document_provenance_and_review_state():
    out = explain_document(doc())
    assert out["schema_version"] == "AI-3G.1"
    assert out["items"][0]["evidence"] == "Hemoglobin 13.2 g/dL"
    assert out["items"][0]["verified"] is False
    assert out["review_required"] is True
    assert out["classification"]["confidence"] == 0.93


def test_verified_document_keeps_human_provenance():
    d = doc(verification_status="Verified", verified_by=4, verified_at=datetime(2026,1,2),
            verified_data=json.dumps({"items":[{"category":"lab","label":"Hemoglobin","value":"13.2","confidence":0.91,"evidence":"Hemoglobin 13.2 g/dL","needs_review":False,"verified":True}]}))
    out = explain_document(d)
    assert out["items"][0]["verified"] is True
    assert out["verification"]["verified_by"] == 4
    assert out["review_required"] is False


def test_timeline_trace_is_non_destructive():
    timeline={"timeline_type":"longitudinal_clinical_record","events":[{"event_type":"document","title":"lab.txt","occurred_at":"2026-01-02T00:00:00","source":"medical_document","source_id":7,"verification_status":"Pending","details":{"evidence_available":True}}]}
    out=explain_timeline(timeline)
    assert out["event_count"] == 1
    assert out["event_trace"][0]["source_id"] == 7
    assert "diagnosis" not in json.dumps(out).lower()
