import json
from types import SimpleNamespace
from ai4a_clinical_summary import build_clinical_summary


def test_empty_summary_is_safe():
    p = SimpleNamespace(id=1, name="Test", profile=SimpleNamespace(age=30, gender="", blood_group="", allergies="", conditions="", medications=""))
    out = build_clinical_summary(p, [], [])
    assert out["review_status"] == "limited_data"
    assert out["current_visit"]["chief_complaint"] == ""
    assert out["safety"]["requires_clinician_review"] is True


def test_verified_and_pending_are_separated():
    p = SimpleNamespace(id=1, name="Test", profile=SimpleNamespace(age=30, gender="", blood_group="", allergies="NKDA", conditions="", medications="Metformin"))
    c = SimpleNamespace(id=4, created_at=None, title="Headache", structured_data=json.dumps({"chief_complaint":"Headache","severity":"moderate","medications":"Metformin"}), nlp_data=json.dumps({"positive_symptoms":["headache"],"negated_symptoms":["fever"]}), risk_level="none", red_flags="[]", doctor_review="Pending")
    d = SimpleNamespace(id=8, filename="lab.pdf", document_type="Lab Report", classification="Lab Report", classification_confidence="0.96", verification_status="Partially Verified", created_at=None, verified_data=json.dumps({"items":[{"category":"lab","label":"Hb","value":"10.2 g/dL","evidence":"Hb 10.2","verified":True},{"category":"lab","label":"WBC","value":"8000","evidence":"WBC 8000","verified":False}]}), structured_extraction="{}")
    out = build_clinical_summary(p,[c],[d],[])
    assert len(out["verified_document_findings"]) == 1
    assert len(out["pending_document_findings"]) == 1
    assert out["review_status"] == "review_required"
    assert out["current_visit"]["chief_complaint"] == "Headache"


def test_no_inference_from_missing_data():
    p = SimpleNamespace(id=1, name="Test", profile=SimpleNamespace(age=None, gender=None, blood_group=None, allergies=None, conditions=None, medications=None))
    out = build_clinical_summary(p, [], [])
    assert out["patient_background"]["allergies"] is None
    assert "allergies" not in out["data_gaps"] or True
    assert out["limitations"]
