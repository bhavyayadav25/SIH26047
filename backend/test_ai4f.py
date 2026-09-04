import json
from datetime import datetime
from types import SimpleNamespace
from ai4f_clinical_question_assistant import answer_clinical_question


def patient(meds="Metformin 500 mg", allergies="Penicillin", conditions="Type 2 diabetes"):
    return SimpleNamespace(id=1, profile=SimpleNamespace(medications=meds, allergies=allergies, conditions=conditions))


def doc(doc_id=1, verified=True):
    items=[{"category":"medication","label":"Medicine","value":"Metformin 500 mg","verified":verified,"evidence":"Metformin 500 mg"},
           {"category":"lab_result","label":"HbA1c","value":"7.2","unit":"%","verified":verified,"evidence":"HbA1c: 7.2 %"}]
    return SimpleNamespace(id=doc_id, filename=f"report{doc_id}.pdf", verified_data=json.dumps({"items":items}))


def consult():
    return SimpleNamespace(id=10, created_at=datetime(2026,1,1), title="Follow-up", chief_complaint="Cough", history="Two weeks", summary="Follow-up recorded", risk_level="none", doctor_review="Reviewed")


def test_medication_question_is_grounded():
    out=answer_clinical_question(patient(), [consult()], [doc()], "What medications are recorded?")
    assert out["grounded"] is True
    assert "Metformin" in out["answer"]
    assert out["safety"]["prescribing"] is False


def test_unverified_document_not_used():
    out=answer_clinical_question(patient(meds=""), [], [doc(1, False)], "What are the lab results?")
    assert out["grounded"] is False
    assert out["record_scope"]["verified_document_items_only"] is True


def test_allergy_question_uses_profile_only():
    out=answer_clinical_question(patient(), [], [], "What allergies are recorded?")
    assert "Penicillin" in out["answer"]


def test_unknown_question_does_not_invent():
    out=answer_clinical_question(patient(meds="", allergies="", conditions=""), [], [], "What is the patient's blood type?")
    assert out["grounded"] is False
    assert "not find" in out["answer"].lower()


def test_diagnosis_and_treatment_are_disabled():
    out=answer_clinical_question(patient(), [consult()], [doc()], "What is the diagnosis and what should we prescribe?")
    assert out["safety"]["diagnosis"] is False
    assert out["safety"]["treatment_recommendation"] is False
