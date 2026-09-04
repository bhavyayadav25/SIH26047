import json
from types import SimpleNamespace
from ai4d_medication_intelligence import build_medication_intelligence


def patient(medications="", allergies=""):
    return SimpleNamespace(id=1, profile=SimpleNamespace(medications=medications, allergies=allergies))


def doc(items):
    return SimpleNamespace(id=7, filename="prescription.pdf", verified_data=json.dumps({"items": items}))


def test_verified_medicine_is_included():
    out = build_medication_intelligence(patient(), [doc([{"category":"medication","label":"Medicine","value":"Amoxicillin 500 mg","verified":True,"evidence":"rx"}])], [])
    assert any("Amoxicillin" in x["name"] for x in out["medications"])
    assert out["evidence_scope"]["unverified_document_values_used"] is False


def test_unverified_medicine_is_excluded():
    out = build_medication_intelligence(patient(), [doc([{"category":"medication","label":"Medicine","value":"HiddenDrug 10 mg","verified":False}])], [])
    assert out["medications"] == []
    assert out["reconciliation_alerts"][0]["id"] == "medication_history_missing"


def test_profile_document_difference_is_flagged_not_resolved():
    out = build_medication_intelligence(patient("Medicine A 10 mg"), [doc([{"category":"medication","label":"Medicine","value":"Medicine B 5 mg","verified":True}])], [])
    assert out["discrepancies"]
    assert out["discrepancies"][0]["type"] == "reconciliation_difference"
    assert out["prescribing"] is False
    assert out["dose_changes"] is False


def test_allergy_matching_is_not_falsely_claimed():
    out = build_medication_intelligence(patient("Medicine A", "Medicine B allergy"), [], [])
    assert out["allergy_context"]["recorded"] is True
    assert out["interaction_check"]["performed"] is False


def test_duplicate_same_medicine_is_flagged():
    out = build_medication_intelligence(patient("Medicine A 10 mg"), [doc([{"category":"medication","label":"Medicine","value":"Medicine A 10 mg","verified":True}])], [])
    assert any(x["type"] == "possible_duplicate_record" for x in out["reconciliation_alerts"])
