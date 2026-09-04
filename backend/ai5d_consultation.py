"""AI-5D: clinician-owned consultation workflow.

AI-5D turns the encounter into an editable consultation record. The AI may
provide context elsewhere in the unified backend, but all examination,
assessment, diagnosis, plan, prescription and follow-up content stored by
these endpoints is explicitly clinician-entered. This module never generates
or infers a diagnosis, treatment, dose, or prescription.
"""
from __future__ import annotations
from typing import Any, Dict
import json

SCHEMA_VERSION = "AI-5D.1"
CONSULTATION_STATUSES = {"draft", "in_progress", "completed"}
CLINICAL_ROLES = {"doctor", "triage", "admin"}


def _clean_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _json(value: Any, default: Any):
    if not value:
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def build_consultation_payload(consultation: Any, encounter: Any | None = None) -> Dict[str, Any]:
    data = _json(getattr(consultation, "structured_data", None), {})
    if not isinstance(data, dict):
        data = {}
    sections = data.get("consultation", {})
    if not isinstance(sections, dict):
        sections = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "consultation_id": consultation.id,
        "encounter_id": getattr(consultation, "encounter_id", None) or getattr(encounter, "id", None),
        "patient_id": consultation.patient_id,
        "status": getattr(consultation, "consultation_status", None) or "draft",
        "title": consultation.title,
        "sections": sections,
        "doctor_review": consultation.doctor_review or "Pending",
        "doctor_notes": consultation.doctor_notes or "",
        "created_at": consultation.created_at.isoformat() if consultation.created_at else None,
        "updated_at": consultation.updated_at.isoformat() if getattr(consultation, "updated_at", None) else None,
        "safety": {
            "diagnosis_ai_generated": False,
            "treatment_ai_generated": False,
            "prescription_ai_generated": False,
            "clinician_owned_clinical_decision": True,
        },
    }


def normalize_sections(payload: Dict[str, Any]) -> Dict[str, Any]:
    sections = payload.get("sections") or {}
    if not isinstance(sections, dict):
        raise ValueError("sections must be an object")
    allowed = {"history", "examination", "assessment", "diagnosis", "plan", "prescription", "follow_up"}
    result: Dict[str, Any] = {}
    for key in allowed:
        value = sections.get(key, "")
        if isinstance(value, dict):
            # Preserve structured clinician-entered content while applying a conservative size cap.
            result[key] = json.loads(json.dumps(value, ensure_ascii=False))
        elif isinstance(value, list):
            result[key] = json.loads(json.dumps(value, ensure_ascii=False))
        else:
            result[key] = _clean_text(value, 5000)
    return result


def consultation_summary(sections: Dict[str, Any]) -> str:
    labels = [("history", "History"), ("examination", "Examination"), ("assessment", "Assessment"),
              ("diagnosis", "Diagnosis"), ("plan", "Plan"), ("prescription", "Prescription"),
              ("follow_up", "Follow-up")]
    parts = []
    for key, label in labels:
        value = sections.get(key)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        value = _clean_text(value, 1200)
        if value:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)[:10000] or "Consultation draft."
