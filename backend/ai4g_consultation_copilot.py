"""AI-4G Consultation Copilot.

Conservative, clinician-facing assistance for the actual consultation screen.
It organizes existing patient/consultation information into a draft structure.
It does not diagnose, prescribe, or silently write back clinical decisions.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List

SCHEMA_VERSION = "AI-4G.1"
MAX_ITEMS = 8


def _text(v: Any) -> str:
    return "" if v in (None, "") else str(v).strip()


def _json(v: Any, fallback: Any):
    if not v:
        return fallback
    try:
        x = json.loads(v) if isinstance(v, str) else v
        return x
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _source(source_type: str, value: Any, **meta: Any) -> Dict[str, Any]:
    return {"source_type": source_type, "value": _text(value), **meta}


def _profile(patient: Any) -> Dict[str, str]:
    p = getattr(patient, "profile", None)
    return {
        "allergies": _text(getattr(p, "allergies", "")) if p else "",
        "conditions": _text(getattr(p, "conditions", "")) if p else "",
        "medications": _text(getattr(p, "medications", "")) if p else "",
    }


def _verified_document_items(documents: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in documents:
        data = _json(getattr(d, "verified_data", None), {})
        for item in data.get("items", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and item.get("verified") is True:
                out.append({**item, "document_id": getattr(d, "id", None), "filename": getattr(d, "filename", "")})
    return out


def build_consultation_copilot(patient: Any, consultation: Any, documents: List[Any],
                               summary: Dict[str, Any] | None = None,
                               risk: Dict[str, Any] | None = None,
                               medication: Dict[str, Any] | None = None,
                               investigations: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = _profile(patient)
    structured = _json(getattr(consultation, "structured_data", None), {})
    nlp = _json(getattr(consultation, "nlp_data", None), {})
    red_flags = _json(getattr(consultation, "red_flags", None), [])
    verified = _verified_document_items(documents)

    history = _text(getattr(consultation, "history", ""))
    complaint = _text(getattr(consultation, "chief_complaint", ""))
    consult_summary = _text(getattr(consultation, "summary", ""))

    current_context: List[str] = []
    sources: List[Dict[str, Any]] = []
    if complaint:
        current_context.append(complaint); sources.append(_source("consultation", complaint, consultation_id=consultation.id, field="chief_complaint"))
    if history:
        current_context.append(history); sources.append(_source("consultation", history, consultation_id=consultation.id, field="history"))
    if p["conditions"]:
        current_context.append(f"Recorded conditions: {p['conditions']}"); sources.append(_source("patient_profile", p["conditions"], field="conditions"))
    if p["allergies"]:
        current_context.append(f"Recorded allergies: {p['allergies']}"); sources.append(_source("patient_profile", p["allergies"], field="allergies"))

    verified_meds = []
    verified_investigations = []
    for item in verified:
        cat = _text(item.get("category")).lower()
        label = _text(item.get("label"))
        value = _text(item.get("value") or item.get("name"))
        if not value:
            continue
        if "medic" in cat or "prescription" in cat or "medic" in label.lower():
            verified_meds.append({"label": label or "Medicine", "value": value, "document_id": item["document_id"], "filename": item["filename"], "evidence": _text(item.get("evidence"))})
        elif cat in {"lab result", "lab_result", "measurement", "finding", "imaging finding", "impression"}:
            verified_investigations.append({"label": label or "Investigation", "value": value, "unit": _text(item.get("unit")), "document_id": item["document_id"], "filename": item["filename"], "evidence": _text(item.get("evidence"))})

    questions = []
    if not complaint: questions.append({"question": "Confirm the patient's chief complaint and primary reason for today's visit.", "basis": "missing current complaint"})
    if not history: questions.append({"question": "Confirm the relevant history, onset, duration, and progression of the current problem.", "basis": "missing current history"})
    if not p["allergies"]: questions.append({"question": "Confirm allergies and medication intolerances.", "basis": "allergy record is empty"})
    if not p["medications"] and not verified_meds: questions.append({"question": "Confirm current medications, including dose and frequency where known.", "basis": "no medication record available"})
    if risk and risk.get("risk_level") in {"urgent", "emergency"}:
        questions.insert(0, {"question": "Confirm the reported red-flag information and perform appropriate clinical triage.", "basis": "AI-4B recorded urgent/emergency review"})
    questions = questions[:MAX_ITEMS]

    alerts = []
    if red_flags:
        alerts.append({"type": "red_flag", "severity": getattr(consultation, "risk_level", "none") or "none", "items": red_flags[:MAX_ITEMS], "action": "Clinician/triage review required; this is not an AI diagnosis."})
    if risk and risk.get("risk_level") in {"urgent", "emergency"} and not red_flags:
        alerts.append({"type": "risk_review", "severity": risk.get("risk_level"), "action": "Review AI-4B risk evidence before proceeding."})

    draft = {
        "history": {"current_complaint": complaint, "history": history, "recorded_context": current_context[:MAX_ITEMS]},
        "examination": {"status": "Not generated by AI-4G", "instruction": "Enter clinician-observed examination findings here."},
        "assessment": {"status": "Clinician decision required", "ai_diagnoses": []},
        "plan": {"status": "Clinician decision required", "ai_treatment_recommendations": [], "prescription": None},
        "follow_up": {"status": "Clinician decision required", "suggestions": []},
    }

    if consult_summary:
        draft["history"]["existing_consultation_summary"] = consult_summary
    if verified_meds:
        draft["history"]["verified_document_medications"] = verified_meds[:MAX_ITEMS]
    if verified_investigations:
        draft["history"]["verified_document_investigations"] = verified_investigations[:MAX_ITEMS]

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "consultation_id": getattr(consultation, "id", None),
        "mode": "clinician_assist_only",
        "draft": draft,
        "suggested_questions": questions,
        "review_alerts": alerts,
        "sources": sources[:MAX_ITEMS],
        "record_scope": {
            "verified_document_items_only": True,
            "patient_profile_used": True,
            "current_consultation_used": True,
            "unverified_document_items_used": False,
        },
        "safety": {
            "diagnosis": False,
            "prescribing": False,
            "treatment_recommendation": False,
            "autonomous_decision": False,
            "writes_clinical_decision": False,
            "requires_clinician_judgment": True,
        },
        "limitations": [
            "AI-4G organizes recorded information; it does not establish a diagnosis.",
            "Examination, assessment, plan, prescription, and follow-up decisions remain clinician-entered.",
            "Unverified document extraction is excluded from clinical evidence.",
            "The copilot does not independently validate the medical record.",
        ],
    }
