"""AI-4A: clinician-facing longitudinal clinical summary.

This module is deliberately provenance-first. It summarizes information already
stored by MediKiosk; it does not diagnose, prescribe, or fill missing facts.
Verified document items are separated from pending items, and every summary
section retains source identifiers where possible.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List
import json
from datetime import datetime

SCHEMA_VERSION = "AI-4A.1"


def _json(value: Any, default: Any):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _iso(obj: Any):
    return obj.isoformat() if hasattr(obj, "isoformat") else (str(obj) if obj else None)


def _clean(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _document_items(document: Any):
    raw = _json(getattr(document, "verified_data", None), None)
    if not isinstance(raw, dict) or not raw:
        raw = _json(getattr(document, "structured_extraction", None), {})
    return raw.get("items", []) if isinstance(raw, dict) else []


def _latest_consultation(consultations: List[Any]):
    if not consultations:
        return None
    return sorted(consultations, key=lambda c: getattr(c, "created_at", None) or datetime.min, reverse=True)[0]


def _history_from_consultation(c: Any) -> Dict[str, Any]:
    if not c:
        return {"chief_complaint": "", "history": {}, "background": {}, "risk_level": "none", "red_flags": []}
    structured = _json(getattr(c, "structured_data", None), {})
    if not isinstance(structured, dict):
        structured = {}
    keys = [
        ("onset", "onset"), ("duration", "duration"), ("location", "location"),
        ("severity", "severity"), ("character", "character"),
        ("general_change", "aggravating_or_relieving_factors"),
        ("general_impact", "daily_impact"),
    ]
    history = {label: structured.get(key) for key, label in keys if _clean(structured.get(key))}
    background = {}
    for key in ("past_history", "medications", "allergies", "family_history", "personal_history"):
        if _clean(structured.get(key)):
            background[key] = structured.get(key)
    associated = []
    for key in ("associated_symptoms", "chest_breathlessness", "headache_nausea", "abdominal_bowel", "fever_infection", "respiratory_cough", "respiratory_wheeze", "review_of_systems"):
        if _clean(structured.get(key)):
            associated.append({"field": key, "value": structured.get(key)})
    nlp = _json(getattr(c, "nlp_data", None), {})
    return {
        "consultation_id": getattr(c, "id", None),
        "created_at": _iso(getattr(c, "created_at", None)),
        "chief_complaint": structured.get("chief_complaint") or getattr(c, "title", "") or "",
        "history": history,
        "associated_information": associated,
        "background": background,
        "reported_positive_symptoms": list(dict.fromkeys(nlp.get("positive_symptoms") or [])) if isinstance(nlp, dict) else [],
        "reported_negative_symptoms": list(dict.fromkeys(nlp.get("negated_symptoms") or [])) if isinstance(nlp, dict) else [],
        "risk_level": getattr(c, "risk_level", None) or "none",
        "red_flags": _json(getattr(c, "red_flags", None), []),
        "doctor_review": getattr(c, "doctor_review", None) or "Pending",
    }


def build_clinical_summary(patient: Any, consultations: Iterable[Any], documents: Iterable[Any], timeline: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    consultations = list(consultations)
    documents = list(documents)
    latest = _latest_consultation(consultations)
    history = _history_from_consultation(latest)

    profile = getattr(patient, "profile", None)
    demographics = {
        "patient_id": getattr(patient, "id", None),
        "name": getattr(patient, "name", None),
        "age": getattr(profile, "age", None) if profile else None,
        "gender": getattr(profile, "gender", None) if profile else None,
        "blood_group": getattr(profile, "blood_group", None) if profile else None,
    }
    patient_background = {
        k: getattr(profile, k, None) for k in ("allergies", "conditions", "medications")
    } if profile else {"allergies": None, "conditions": None, "medications": None}

    confirmed: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    doc_index: List[Dict[str, Any]] = []
    for d in documents:
        items = _document_items(d)
        verified_count = 0
        pending_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            out = {
                "document_id": getattr(d, "id", None),
                "filename": getattr(d, "filename", None),
                "category": item.get("category"),
                "label": item.get("label"),
                "value": item.get("value"),
                "evidence": item.get("evidence", ""),
                "verified": item.get("verified") is True,
            }
            if out["verified"]:
                confirmed.append(out); verified_count += 1
            else:
                pending.append(out); pending_count += 1
        doc_index.append({
            "document_id": getattr(d, "id", None),
            "filename": getattr(d, "filename", None),
            "document_type": getattr(d, "document_type", None),
            "classification": getattr(d, "classification", None) or getattr(d, "document_type", None),
            "classification_confidence": float(getattr(d, "classification_confidence", 0) or 0),
            "verification_status": getattr(d, "verification_status", None) or "Pending",
            "verified_items": verified_count,
            "pending_items": pending_count,
            "created_at": _iso(getattr(d, "created_at", None)),
        })

    safety = {
        "level": history["risk_level"],
        "alerts": history["red_flags"],
        "requires_clinician_review": True,
    }
    if not latest and not documents:
        readiness = "limited_data"
    elif pending or any(d["verification_status"] != "Verified" for d in doc_index):
        readiness = "review_required"
    else:
        readiness = "review_ready"

    gaps = []
    if not history["chief_complaint"]: gaps.append("chief complaint")
    if profile and not _clean(profile.allergies): gaps.append("allergies")
    if profile and not _clean(profile.medications): gaps.append("current medications")
    if pending: gaps.append(f"{len(pending)} document extraction item(s) awaiting verification")

    return {
        "schema_version": SCHEMA_VERSION,
        "patient": demographics,
        "headline": history["chief_complaint"] or "Clinical summary — information available for review",
        "current_visit": history,
        "patient_background": patient_background,
        "verified_document_findings": confirmed,
        "pending_document_findings": pending,
        "documents": doc_index,
        "recent_encounters": [
            {"consultation_id": getattr(c, "id", None), "created_at": _iso(getattr(c, "created_at", None)),
             "title": getattr(c, "title", None), "risk_level": getattr(c, "risk_level", None) or "none",
             "doctor_review": getattr(c, "doctor_review", None) or "Pending"}
            for c in sorted(consultations, key=lambda x: getattr(x, "created_at", None) or datetime.min, reverse=True)[:10]
        ],
        "timeline": timeline or [],
        "safety": safety,
        "data_gaps": gaps,
        "review_status": readiness,
        "provenance": {
            "sources": ["patient profile", "consultation history", "AI-3 verified document extraction", "AI-3F timeline"],
            "document_values_are_confirmed_only_when_verified": True,
        },
        "limitations": [
            "This summary reorganizes stored information and does not establish a diagnosis.",
            "Unverified document findings remain explicitly marked and should not be treated as confirmed facts.",
            "Missing information is not inferred or filled in.",
            "A clinician must review the source record and make all clinical decisions.",
        ],
    }
