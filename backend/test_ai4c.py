import json
from types import SimpleNamespace
from ai4c_clinical_decision_support import build_decision_support


def patient(profile=None):
    return SimpleNamespace(id=1, profile=profile or SimpleNamespace(allergies="", conditions="", medications=""))


def doc(items):
    return SimpleNamespace(id=2, filename="rx.pdf", verified_data=json.dumps({"items": items}))


def test_emergency_becomes_triage_prompt_not_diagnosis():
    out = build_decision_support(patient(), [], [], {"data_gaps": []}, {"risk_level": "emergency", "alerts": [{}]})
    assert out["review_prompts"][0]["priority"] == "critical"
    assert out["diagnoses"] == []
    assert out["treatment_recommendations"] == []
    assert out["autonomous_decision"] is False


def test_data_gaps_create_questions():
    out = build_decision_support(patient(), [], [], {"data_gaps": ["allergies", "current medications"], "current_visit": {}}, {"risk_level": "none", "alerts": []})
    assert any(x["type"] == "information_gap" for x in out["review_prompts"])
    assert any(x["id"] == "confirm_allergies" for x in out["questions_to_confirm"])


def test_unverified_document_is_not_used_as_decision_evidence():
    d = doc([{"label": "Medication", "value": "Example", "verified": False}])
    out = build_decision_support(patient(), [], [d], {"data_gaps": []}, {"risk_level": "none", "alerts": []})
    assert out["evidence_scope"]["uses_unverified_document_values_for_decisions"] is False
    assert out["recommendations"] == []


def test_verified_medication_difference_is_flagged_not_resolved():
    p = patient(SimpleNamespace(allergies="NKDA", conditions="", medications="Medicine A 10 mg"))
    d = doc([{"category": "Medication", "label": "Medicine", "value": "Medicine B 5 mg", "verified": True, "evidence": "rx"}])
    out = build_decision_support(p, [], [d], {"data_gaps": []}, {"risk_level": "none", "alerts": []})
    assert out["record_consistency_checks"]
    assert out["record_consistency_checks"][0]["type"] == "possible_discrepancy"
    assert out["treatment_recommendations"] == []
