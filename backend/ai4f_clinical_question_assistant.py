"""AI-4F: source-grounded clinical question assistant.

Answers clinician questions from the patient's stored record and the outputs of
AI-3/AI-4. It is deliberately extractive/grounded: it does not diagnose,
prescribe, infer missing facts, or use unverified document extraction as evidence.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "AI-4F.1"
MAX_ANSWER_ITEMS = 8


def _text(v: Any) -> str:
    return "" if v in (None, "") else str(v).strip()


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(v).lower()).strip()


def _profile_sources(patient: Any) -> Dict[str, Any]:
    p = getattr(patient, "profile", None)
    return {
        "conditions": _text(getattr(p, "conditions", "")) if p else "",
        "allergies": _text(getattr(p, "allergies", "")) if p else "",
        "medications": _text(getattr(p, "medications", "")) if p else "",
    }


def _verified_items(documents: List[Any]) -> List[Dict[str, Any]]:
    import json
    out = []
    for d in documents:
        try:
            payload = json.loads(getattr(d, "verified_data", None) or "{}")
        except (TypeError, ValueError):
            payload = {}
        for item in payload.get("items", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict) and item.get("verified") is True:
                out.append({**item, "document_id": getattr(d, "id", None), "filename": getattr(d, "filename", None)})
    return out


def _consultation_sources(consultations: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for c in consultations:
        out.append({
            "consultation_id": getattr(c, "id", None),
            "created_at": getattr(c, "created_at", None).isoformat() if getattr(c, "created_at", None) else None,
            "title": _text(getattr(c, "title", "")),
            "chief_complaint": _text(getattr(c, "chief_complaint", "")),
            "history": _text(getattr(c, "history", "")),
            "summary": _text(getattr(c, "summary", "")),
            "risk_level": _text(getattr(c, "risk_level", "")),
            "doctor_review": _text(getattr(c, "doctor_review", "")),
        })
    return out


def _source(label: str, value: Any, **meta: Any) -> Dict[str, Any]:
    return {"source_type": label, "value": _text(value), **meta}


def _contains_any(q: str, words: List[str]) -> bool:
    return any(w in q for w in words)


def answer_clinical_question(patient: Any, consultations: List[Any], documents: List[Any], question: str) -> Dict[str, Any]:
    q = _norm(question)
    if not q:
        return {"schema_version": SCHEMA_VERSION, "answer": "Please provide a clinical question.", "grounded": False, "sources": [], "safety": _safety()}

    profile = _profile_sources(patient)
    verified = _verified_items(documents)
    consults = _consultation_sources(consultations)
    sources: List[Dict[str, Any]] = []
    facts: List[str] = []
    topic = "record"

    if _contains_any(q, ["medication", "medicines", "medicine", "drug", "prescription", "taking"]):
        topic = "medications"
        if profile["medications"]:
            facts.append(f"Patient profile medications: {profile['medications']}")
            sources.append(_source("patient_profile", profile["medications"], field="medications"))
        for x in verified:
            cat = _norm(x.get("category")); label = _norm(x.get("label"))
            if "medic" in cat or "prescription" in cat or "medic" in label:
                value = _text(x.get("value") or x.get("name"))
                if value:
                    facts.append(f"Verified document medicine: {value}")
                    sources.append(_source("verified_document", value, document_id=x["document_id"], filename=x["filename"], evidence=_text(x.get("evidence"))))
    elif _contains_any(q, ["allerg", "allergic"]):
        topic = "allergies"
        if profile["allergies"]:
            facts.append(f"Recorded allergies: {profile['allergies']}")
            sources.append(_source("patient_profile", profile["allergies"], field="allergies"))
        else:
            facts.append("No allergy information is recorded in the patient profile available to this assistant.")
    elif _contains_any(q, ["condition", "history", "past medical", "disease", "diagnos"]):
        topic = "history"
        if profile["conditions"]:
            facts.append(f"Recorded conditions: {profile['conditions']}")
            sources.append(_source("patient_profile", profile["conditions"], field="conditions"))
        for c in consults[:MAX_ANSWER_ITEMS]:
            text = " — ".join(x for x in [c["title"], c["chief_complaint"], c["history"], c["summary"]] if x)
            if text:
                facts.append(text)
                sources.append(_source("consultation", text, consultation_id=c["consultation_id"], created_at=c["created_at"]))
    elif _contains_any(q, ["lab", "test", "investigation", "report", "result", "blood", "scan", "imaging", "x ray", "xray", "mri", "ct"]):
        topic = "investigations"
        for x in verified[:MAX_ANSWER_ITEMS]:
            cat = _norm(x.get("category")); label = _text(x.get("label")); value = _text(x.get("value"))
            if not value: continue
            if cat in {"lab result", "measurement", "finding", "imaging finding", "impression"} or label:
                line = f"{label or 'Investigation'}: {value}"
                if _text(x.get("unit")): line += f" {_text(x.get('unit'))}"
                facts.append(line)
                sources.append(_source("verified_document", line, document_id=x["document_id"], filename=x["filename"], evidence=_text(x.get("evidence"))))
    elif _contains_any(q, ["risk", "red flag", "urgent", "emergency", "danger"]):
        topic = "risk"
        # Use only persisted consultation risk labels here; no new clinical inference.
        for c in consults[:MAX_ANSWER_ITEMS]:
            if c["risk_level"]:
                facts.append(f"Consultation {c['consultation_id']} recorded risk level: {c['risk_level']}")
                sources.append(_source("consultation", c["risk_level"], consultation_id=c["consultation_id"], created_at=c["created_at"], field="risk_level"))
    elif _contains_any(q, ["when", "date", "latest visit", "last visit", "previous visit", "recent visit"]):
        topic = "encounters"
        for c in consults[:MAX_ANSWER_ITEMS]:
            facts.append(f"Consultation {c['consultation_id']} — {c['created_at'] or 'date not recorded'} — {c['title'] or c['chief_complaint'] or 'encounter'}")
            sources.append(_source("consultation", c["title"] or c["chief_complaint"] or "encounter", consultation_id=c["consultation_id"], created_at=c["created_at"]))
    else:
        topic = "record_overview"
        if profile["conditions"]: facts.append(f"Recorded conditions: {profile['conditions']}"); sources.append(_source("patient_profile", profile["conditions"], field="conditions"))
        if profile["allergies"]: facts.append(f"Recorded allergies: {profile['allergies']}"); sources.append(_source("patient_profile", profile["allergies"], field="allergies"))
        if profile["medications"]: facts.append(f"Recorded medications: {profile['medications']}"); sources.append(_source("patient_profile", profile["medications"], field="medications"))
        for c in consults[:3]:
            text = " — ".join(x for x in [c["title"], c["chief_complaint"]] if x)
            if text: facts.append(text); sources.append(_source("consultation", text, consultation_id=c["consultation_id"], created_at=c["created_at"]))

    if not facts:
        answer = "I could not find a matching verified record entry for that question. Do not treat the absence of a record entry as evidence that the condition, medicine, or result is absent."
        grounded = False
    else:
        answer = "\n".join(f"• {x}" for x in facts[:MAX_ANSWER_ITEMS])
        grounded = True

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "question": _text(question),
        "topic": topic,
        "answer": answer,
        "grounded": grounded,
        "source_count": len(sources),
        "sources": sources[:MAX_ANSWER_ITEMS],
        "record_scope": {
            "verified_document_items_only": True,
            "patient_profile_used": True,
            "consultation_history_used": True,
            "unverified_document_items_used": False,
        },
        "safety": _safety(),
        "limitations": [
            "Answers are limited to information present in the patient's stored record.",
            "Missing information is not inferred or filled in.",
            "A record entry is not independently validated by this assistant.",
            "This assistant does not diagnose, prescribe, change medication, or make treatment decisions.",
            "Clinicians should review the cited source record before acting on an answer.",
        ],
    }


def _safety() -> Dict[str, Any]:
    return {
        "source_grounded": True,
        "uses_unverified_document_data": False,
        "diagnosis": False,
        "prescribing": False,
        "treatment_recommendation": False,
        "autonomous_decision": False,
        "requires_clinician_judgment": True,
    }
