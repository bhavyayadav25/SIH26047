"""AI-4C: conservative clinician decision-support prompts.

AI-4C does not diagnose, prescribe, select treatment, or claim that a patient
has a disease. It converts verified/current record signals into review prompts,
questions, and data-quality checks for a clinician. Every prompt carries its
source and is explicitly marked as requiring clinician judgment.
"""
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "AI-4C.1"


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _verified_items(documents: Iterable[Any]) -> List[Dict[str, Any]]:
    out = []
    for d in documents:
        raw = getattr(d, "verified_data", None) or ""
        try:
            import json
            payload = raw if isinstance(raw, dict) else json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            payload = {}
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items:
            if isinstance(item, dict) and item.get("verified") is True:
                out.append({
                    "document_id": getattr(d, "id", None),
                    "filename": getattr(d, "filename", None),
                    "category": item.get("category"),
                    "label": item.get("label"),
                    "value": item.get("value"),
                    "evidence": item.get("evidence", ""),
                })
    return out


def _profile_values(patient: Any) -> Dict[str, str]:
    profile = getattr(patient, "profile", None)
    if not profile:
        return {}
    return {k: _text(getattr(profile, k, "")) for k in ("allergies", "conditions", "medications")}


def _consultation_data(consultations: Iterable[Any]) -> List[Dict[str, Any]]:
    import json
    result = []
    for c in consultations:
        def load(v):
            if isinstance(v, dict): return v
            try: return json.loads(v or "{}")
            except (TypeError, ValueError): return {}
        structured = load(getattr(c, "structured_data", None))
        nlp = load(getattr(c, "nlp_data", None))
        result.append({
            "id": getattr(c, "id", None),
            "title": getattr(c, "title", ""),
            "created_at": str(getattr(c, "created_at", "") or ""),
            "structured": structured,
            "nlp": nlp,
            "risk_level": getattr(c, "risk_level", "none") or "none",
            "red_flags": load(getattr(c, "red_flags", None)) if getattr(c, "red_flags", None) else [],
        })
    return result


def build_decision_support(patient: Any, consultations: Iterable[Any], documents: Iterable[Any], summary: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    consultations = _consultation_data(consultations)
    verified = _verified_items(documents)
    profile = _profile_values(patient)
    prompts: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    consistency: List[Dict[str, Any]] = []

    risk_level = risk.get("risk_level", "none")
    if risk_level == "emergency":
        prompts.append({
            "id": "urgent_triage_review",
            "priority": "critical",
            "type": "safety_review",
            "title": "Immediate human triage review",
            "text": "Potentially serious warning information is present. Confirm the alert and follow the site's emergency/triage protocol.",
            "source": "AI-4B risk assessment",
            "requires_clinician_judgment": True,
        })
    elif risk_level == "urgent":
        prompts.append({
            "id": "prompt_clinical_review",
            "priority": "high",
            "type": "safety_review",
            "title": "Prompt clinical review",
            "text": "A potential warning pattern was detected. Review the underlying evidence before making clinical decisions.",
            "source": "AI-4B risk assessment",
            "requires_clinician_judgment": True,
        })

    gaps = summary.get("data_gaps", []) or []
    for gap in gaps:
        prompts.append({
            "id": "gap_" + re.sub(r"[^a-z0-9]+", "_", str(gap).lower()).strip("_"),
            "priority": "medium",
            "type": "information_gap",
            "title": "Information gap",
            "text": f"Review or obtain {gap} if clinically relevant to this encounter.",
            "source": "AI-4A clinical summary",
            "requires_clinician_judgment": True,
        })

    current = summary.get("current_visit", {}) or {}
    if not current.get("chief_complaint"):
        questions.append({"id": "confirm_chief_complaint", "priority": "high", "question": "What is the patient's main reason for today's visit?", "source": "AI-4A data gap"})
    if not profile.get("allergies"):
        questions.append({"id": "confirm_allergies", "priority": "high", "question": "Are any medication or other clinically relevant allergies known?", "source": "patient profile"})
    if not profile.get("medications"):
        questions.append({"id": "confirm_medications", "priority": "medium", "question": "What medicines is the patient currently taking, including non-prescription medicines if relevant?", "source": "patient profile"})

    # Conservative consistency checks: surface possible discrepancies, never
    # resolve them automatically.
    profile_meds = _norm(profile.get("medications"))
    doc_meds = [_text(x.get("value")) for x in verified if _norm(x.get("category")) in {"medicine", "medication", "prescription"} or "medic" in _norm(x.get("label"))]
    if profile_meds and doc_meds:
        unmatched = [m for m in doc_meds if m and _norm(m) not in profile_meds]
        if unmatched:
            consistency.append({
                "id": "medication_record_difference",
                "priority": "medium",
                "type": "possible_discrepancy",
                "title": "Medication records may differ",
                "text": "Compare the current medication list with verified document entries before relying on either list.",
                "details": {"document_entries": unmatched[:10]},
                "source": "patient profile + verified documents",
                "requires_clinician_judgment": True,
            })

    if not verified and documents:
        prompts.append({
            "id": "verify_documents",
            "priority": "medium",
            "type": "verification",
            "title": "Verify document information",
            "text": "Document information is not yet available as verified clinical data. Review the source documents before using their contents for decisions.",
            "source": "AI-3E verification state",
            "requires_clinician_judgment": True,
        })

    # Deduplicate prompts by id.
    unique = {p["id"]: p for p in prompts}
    prompts = sorted(unique.values(), key=lambda p: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(p.get("priority"), 9))
    questions = sorted(questions, key=lambda q: {"high": 0, "medium": 1, "low": 2}.get(q.get("priority"), 9))
    consistency = sorted(consistency, key=lambda x: {"critical": 0, "high": 1, "medium": 2}.get(x.get("priority"), 9))

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "headline": "Clinical decision-support prompts for clinician review",
        "risk_context": {"level": risk_level, "alert_count": len(risk.get("alerts", []) or [])},
        "review_prompts": prompts,
        "questions_to_confirm": questions,
        "record_consistency_checks": consistency,
        "recommendations": [],
        "diagnoses": [],
        "treatment_recommendations": [],
        "prescribing": False,
        "autonomous_decision": False,
        "requires_clinician_judgment": True,
        "evidence_scope": {
            "verified_document_items_only": True,
            "uses_ai4a_summary": True,
            "uses_ai4b_risk": True,
            "uses_unverified_document_values_for_decisions": False,
        },
        "limitations": [
            "This is decision-support for clinician review, not a diagnosis or treatment recommendation.",
            "It does not select medicines, doses, investigations, or procedures.",
            "Possible discrepancies are not resolved automatically.",
            "A missing prompt does not mean that no clinical risk exists.",
            "All clinical decisions remain with the responsible clinician and local protocol.",
            "Prototype rules require clinical validation before real-world deployment.",
        ],
    }
