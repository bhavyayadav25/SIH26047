import json
from datetime import datetime
from types import SimpleNamespace
from ai4g_consultation_copilot import build_consultation_copilot


def patient(meds="Metformin 500 mg", allergies="Penicillin", conditions="Type 2 diabetes"):
    return SimpleNamespace(id=1, profile=SimpleNamespace(medications=meds, allergies=allergies, conditions=conditions))


def consultation(complaint="Cough", history="Two weeks", risk="none", red_flags=None):
    return SimpleNamespace(id=10, created_at=datetime(2026,1,1), chief_complaint=complaint, history=history,
                           summary="Recorded consultation", structured_data="{}", nlp_data="{}",
                           risk_level=risk, red_flags=json.dumps(red_flags or []))


def doc(verified=True):
    items=[{"category":"medication","label":"Medicine","value":"Metformin 500 mg","verified":verified,"evidence":"Metformin 500 mg"},
           {"category":"lab_result","label":"HbA1c","value":"7.2","unit":"%","verified":verified,"evidence":"HbA1c: 7.2 %"}]
    return SimpleNamespace(id=2, filename="report.pdf", verified_data=json.dumps({"items":items}))


def test_copilot_is_clinician_assist_only():
    out=build_consultation_copilot(patient(), consultation(), [doc()])
    assert out["mode"] == "clinician_assist_only"
    assert out["safety"]["diagnosis"] is False
    assert out["safety"]["prescribing"] is False
    assert out["safety"]["writes_clinical_decision"] is False
    assert out["draft"]["assessment"]["ai_diagnoses"] == []


def test_unverified_documents_are_excluded():
    out=build_consultation_copilot(patient(meds="", allergies="", conditions=""), consultation(), [doc(False)])
    assert out["record_scope"]["unverified_document_items_used"] is False
    assert "verified_document_medications" not in out["draft"]["history"]
    assert "verified_document_investigations" not in out["draft"]["history"]


def test_missing_fields_generate_questions():
    out=build_consultation_copilot(patient(meds="", allergies="", conditions=""), consultation("", ""), [])
    questions=" ".join(x["question"] for x in out["suggested_questions"])
    assert "chief complaint" in questions.lower()
    assert "allerg" in questions.lower()


def test_urgent_review_is_not_a_diagnosis():
    out=build_consultation_copilot(patient(), consultation(risk="urgent", red_flags=["recorded red flag"]), [])
    assert out["review_alerts"]
    assert out["review_alerts"][0]["type"] == "red_flag"
    assert "diagnosis" in out["review_alerts"][0]["action"].lower()
