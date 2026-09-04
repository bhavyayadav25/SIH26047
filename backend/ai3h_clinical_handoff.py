"""AI-3H: final clinical-intake handoff and readiness gate.

AI-3H closes the AI-3 document pipeline by packaging the already-stored,
traceable information into a physician handoff. It does not diagnose, prescribe,
or infer missing clinical facts. Unverified document extraction remains clearly
labeled and can block a handoff from being marked ready.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
import json

SCHEMA_VERSION = "AI-3H.1"


def _json(value: Any, default: Any):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _verification_counts(document: Any) -> Dict[str, int]:
    status = getattr(document, "verification_status", None) or "Pending"
    verified = _json(getattr(document, "verified_data", None), None)
    extraction = verified if isinstance(verified, dict) and verified else _json(
        getattr(document, "structured_extraction", None), {}
    )
    items = extraction.get("items", []) if isinstance(extraction, dict) else []
    verified_count = sum(1 for x in items if isinstance(x, dict) and x.get("verified") is True)
    total = sum(1 for x in items if isinstance(x, dict))
    return {"total": total, "verified": verified_count, "unverified": max(total - verified_count, 0), "status": status}


def _document_handoff(document: Any) -> Dict[str, Any]:
    counts = _verification_counts(document)
    classification = getattr(document, "classification", None) or getattr(document, "document_type", None) or "Unknown"
    extraction = _json(getattr(document, "verified_data", None), None)
    if not isinstance(extraction, dict) or not extraction:
        extraction = _json(getattr(document, "structured_extraction", None), {})
    items = extraction.get("items", []) if isinstance(extraction, dict) else []
    # Only expose values as confirmed when AI-3E explicitly verified them.
    confirmed = [
        {"category": x.get("category"), "label": x.get("label"), "value": x.get("value"),
         "evidence": x.get("evidence", ""), "verified": True}
        for x in items if isinstance(x, dict) and x.get("verified") is True
    ]
    pending = [
        {"category": x.get("category"), "label": x.get("label"), "value": x.get("value"),
         "evidence": x.get("evidence", ""), "verified": False}
        for x in items if isinstance(x, dict) and x.get("verified") is not True
    ]
    return {
        "document_id": getattr(document, "id", None),
        "filename": getattr(document, "filename", None),
        "document_type": getattr(document, "document_type", None),
        "classification": classification,
        "classification_confidence": float(getattr(document, "classification_confidence", 0) or 0),
        "verification": counts,
        "confirmed_items": confirmed,
        "pending_items": pending,
        "created_at": getattr(document, "created_at", None).isoformat() if getattr(document, "created_at", None) else None,
    }


def build_clinical_handoff(patient_id: int, consultations: Iterable[Any], documents: Iterable[Any]) -> Dict[str, Any]:
    consultation_list = list(consultations)
    document_list = list(documents)
    docs = [_document_handoff(d) for d in document_list]

    blockers: List[Dict[str, Any]] = []
    for d in docs:
        if d["pending_items"]:
            blockers.append({
                "type": "document_verification",
                "document_id": d["document_id"],
                "message": f"{len(d['pending_items'])} extracted item(s) still require verification.",
            })
        if d["classification_confidence"] < 0.80:
            blockers.append({
                "type": "document_classification",
                "document_id": d["document_id"],
                "message": "Document classification confidence is below the review threshold.",
            })

    consultations_out = []
    for c in sorted(consultation_list, key=lambda x: getattr(x, "created_at", None) or "", reverse=True):
        consultations_out.append({
            "consultation_id": getattr(c, "id", None),
            "created_at": getattr(c, "created_at", None).isoformat() if getattr(c, "created_at", None) else None,
            "title": getattr(c, "title", None),
            "summary": getattr(c, "summary", None),
            "risk_level": getattr(c, "risk_level", None) or "none",
            "red_flags": _json(getattr(c, "red_flags", None), []),
            "doctor_review": getattr(c, "doctor_review", None) or "Pending",
            "ai_summary": _json(getattr(c, "ai_summary", None), None),
        })

    latest_risk = consultations_out[0]["risk_level"] if consultations_out else "none"
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "patient_id": patient_id,
        "handoff_status": "Ready for physician review" if not blockers else "Review required before handoff",
        "ready_for_physician_review": not bool(blockers),
        "blockers": blockers,
        "documents": docs,
        "consultations": consultations_out,
        "current_risk_level": latest_risk,
        "safety": {
            "requires_human_decision": True,
            "ai_diagnosis": False,
            "ai_treatment_recommendation": False,
            "red_flags_are_alerts_not_diagnoses": True,
        },
        "provenance": [
            "AI-3B OCR/text extraction",
            "AI-3C document classification",
            "AI-3D structured extraction",
            "AI-3E human verification",
            "AI-3F longitudinal timeline",
            "AI-3G provenance/explainability",
            "AI-3H final clinical-intake handoff gate",
        ],
        "limitations": [
            "AI-3H only organizes information already stored by earlier phases.",
            "Missing information is not inferred or filled in.",
            "The physician must review source documents and make all clinical decisions.",
        ],
    }
    return handoff
