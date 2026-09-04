"""AI-5C: doctor workspace aggregation.

Read-only orchestration layer for the clinician workstation. It assembles
already persisted patient/encounter information and existing AI outputs. It
never diagnoses, prescribes, edits records, or treats absence of data as a
negative clinical finding.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable

CLINICAL_ROLES = {"doctor", "triage", "admin"}


def _json_load(value, default):
    import json
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _patient_snapshot(patient: Any) -> Dict[str, Any]:
    profile = getattr(patient, "profile", None)
    return {
        "id": patient.id,
        "name": patient.name,
        "age": getattr(profile, "age", None) if profile else None,
        "gender": getattr(profile, "gender", None) if profile else None,
        "blood_group": getattr(profile, "blood_group", None) if profile else None,
        "allergies": getattr(profile, "allergies", None) if profile else None,
        "conditions": getattr(profile, "conditions", None) if profile else None,
        "medications": getattr(profile, "medications", None) if profile else None,
    }


def _document_summary(documents: Iterable[Any]) -> list[Dict[str, Any]]:
    result = []
    for doc in documents:
        result.append({
            "id": doc.id,
            "filename": doc.filename,
            "document_type": doc.document_type,
            "classification": doc.classification or doc.document_type,
            "classification_confidence": float(doc.classification_confidence or 0),
            "classification_needs_review": bool(doc.classification_needs_review),
            "verification_status": doc.verification_status or "Pending",
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "verified_at": doc.verified_at.isoformat() if doc.verified_at else None,
        })
    return result


def build_doctor_workspace(patient: Any, encounter: Any, documents: Iterable[Any],
                           consultations: Iterable[Any], summary: Dict[str, Any],
                           risk: Dict[str, Any], decision_support: Dict[str, Any],
                           medication: Dict[str, Any], investigations: Dict[str, Any],
                           copilot: Dict[str, Any], clinical_gate: Dict[str, Any],
                           timeline: list) -> Dict[str, Any]:
    pending_documents = [d.id for d in documents if (d.verification_status or "Pending") != "Verified"]
    consultations_list = list(consultations)
    latest_consultation = consultations_list[0] if consultations_list else None
    risk_level = (risk or {}).get("risk_level", "none")
    return {
        "workspace_version": "AI-5C.1",
        "read_only": True,
        "patient": _patient_snapshot(patient),
        "encounter": {
            "id": encounter.id,
            "department": encounter.department,
            "visit_date": encounter.visit_date,
            "token_number": encounter.token_number,
            "priority": encounter.priority,
            "status": encounter.status,
            "reason": encounter.reason or "",
            "doctor_id": encounter.doctor_id,
        },
        "overview": {
            "latest_consultation_id": latest_consultation.id if latest_consultation else None,
            "document_count": len(_document_summary(documents)),
            "timeline_event_count": len(timeline),
            "pending_document_verifications": pending_documents,
            "risk_level": risk_level,
        },
        "clinical_summary": summary,
        "risk_assessment": risk,
        "decision_support": decision_support,
        "medication_intelligence": medication,
        "investigation_intelligence": investigations,
        "timeline": timeline,
        "documents": _document_summary(documents),
        "consultations": [
            {
                "id": c.id, "title": c.title, "summary": c.summary,
                "status": c.status, "risk_level": c.risk_level,
                "doctor_review": c.doctor_review,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            } for c in consultations_list
        ],
        "consultation_copilot": copilot,
        "clinical_gate": clinical_gate,
        "safety": {
            "diagnosis_generated": False,
            "prescription_generated": False,
            "autonomous_clinical_decision": False,
            "unverified_document_data_used_as_verified_evidence": False,
            "clinician_decision_required": True,
        },
    }
