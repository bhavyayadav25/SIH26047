"""AI-4H: final clinical intelligence and safety gate.

This module combines the outputs of AI-4A..AI-4G into a single clinician-facing
readiness package. It is a safety/orchestration layer, not a diagnostic model.
It never creates a diagnosis, prescription, treatment plan, or new clinical fact.
"""
from __future__ import annotations
from typing import Any, Dict, List

SCHEMA_VERSION = "AI-4H.1"


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _level(value: Any) -> str:
    return str(value or "none").lower().strip()


def _review_required_from_documents(risk: Dict[str, Any]) -> bool:
    state = risk.get("document_review") if isinstance(risk, dict) else {}
    return bool(isinstance(state, dict) and state.get("pending_items"))


def build_final_clinical_gate(
    patient: Any,
    consultation: Any,
    summary: Dict[str, Any] | None = None,
    risk: Dict[str, Any] | None = None,
    decision_support: Dict[str, Any] | None = None,
    medication: Dict[str, Any] | None = None,
    investigations: Dict[str, Any] | None = None,
    copilot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    summary = summary or {}
    risk = risk or {}
    decision_support = decision_support or {}
    medication = medication or {}
    investigations = investigations or {}
    copilot = copilot or {}

    risk_level = _level(risk.get("risk_level"))
    emergency = risk_level == "emergency"
    urgent = risk_level == "urgent"
    pending_docs = _review_required_from_documents(risk)

    blockers: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if emergency:
        blockers.append({
            "code": "EMERGENCY_REVIEW_REQUIRED",
            "severity": "emergency",
            "message": "Potential emergency information is present. Immediate human triage/clinical assessment is required before routine consultation flow.",
            "source": "AI-4B",
        })
    elif urgent:
        warnings.append({
            "code": "URGENT_REVIEW_REQUIRED",
            "severity": "urgent",
            "message": "Potential urgent warning information is present. Prompt clinician review is required.",
            "source": "AI-4B",
        })

    if pending_docs:
        blockers.append({
            "code": "DOCUMENT_VERIFICATION_PENDING",
            "severity": "review",
            "message": "One or more extracted document items remain unverified. Verify them before treating them as clinical facts.",
            "source": "AI-3E/AI-4B",
        })

    # AI-4C is intentionally advisory. Surface its review prompts without
    # turning them into autonomous clinical decisions.
    support_prompts = _as_list(decision_support.get("review_prompts"))
    if support_prompts:
        warnings.append({
            "code": "DECISION_SUPPORT_REVIEW",
            "severity": "review",
            "message": "AI decision-support prompts are available for clinician review.",
            "source": "AI-4C",
            "count": len(support_prompts),
        })

    medication_flags = _as_list(medication.get("flags"))
    if medication_flags:
        warnings.append({
            "code": "MEDICATION_RECONCILIATION_REVIEW",
            "severity": "review",
            "message": "Medication reconciliation flags are present and should be reviewed by the clinician.",
            "source": "AI-4D",
            "count": len(medication_flags),
        })

    investigation_flags = _as_list(investigations.get("flags"))
    if investigation_flags:
        warnings.append({
            "code": "INVESTIGATION_REVIEW",
            "severity": "review",
            "message": "Investigation review flags are present. Clinical interpretation remains the clinician's responsibility.",
            "source": "AI-4E",
            "count": len(investigation_flags),
        })

    copilot_alerts = _as_list(copilot.get("review_alerts"))
    if copilot_alerts:
        warnings.append({
            "code": "COPILOT_REVIEW_ALERTS",
            "severity": "review",
            "message": "Consultation-copilot review alerts are available.",
            "source": "AI-4G",
            "count": len(copilot_alerts),
        })

    # A final gate is deliberately stricter than a summary. Routine readiness
    # requires no emergency blocker and no unverified document items. Advisory
    # warnings do not block; they are surfaced for the clinician.
    if emergency:
        disposition = "immediate_triage"
        ready_for_routine_consultation = False
    elif pending_docs:
        disposition = "verification_required"
        ready_for_routine_consultation = False
    else:
        disposition = "clinician_review"
        ready_for_routine_consultation = True

    limitations = [
        "AI-4H is an orchestration and safety gate, not a diagnosis engine.",
        "A non-blocking result does not establish that the patient is clinically safe.",
        "AI suggestions, flags, and summaries require clinician judgment and validation.",
        "The gate does not prescribe, change medications, establish diagnoses, or make treatment decisions.",
        "Clinical deployment requires appropriate medical, privacy, security, and regulatory validation.",
    ]

    provenance = {
        "AI-4A_clinical_summary": bool(summary),
        "AI-4B_risk_assessment": bool(risk),
        "AI-4C_decision_support": bool(decision_support),
        "AI-4D_medication_intelligence": bool(medication),
        "AI-4E_investigation_intelligence": bool(investigations),
        "AI-4F_clinical_question_assistant": False,
        "AI-4G_consultation_copilot": bool(copilot),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "consultation_id": getattr(consultation, "id", None),
        "mode": "clinician_assist_only",
        "disposition": disposition,
        "ready_for_routine_consultation": ready_for_routine_consultation,
        "risk_level": risk_level,
        "blockers": blockers,
        "warnings": warnings,
        "review_summary": {
            "document_verification_pending": pending_docs,
            "decision_support_prompts": len(support_prompts),
            "medication_flags": len(medication_flags),
            "investigation_flags": len(investigation_flags),
            "copilot_alerts": len(copilot_alerts),
        },
        "clinician_action": (
            "Escalate to immediate human triage/clinical assessment."
            if emergency else
            "Complete required document verification before relying on those extracted items."
            if pending_docs else
            "Review the available AI support and complete the clinical assessment using professional judgment."
        ),
        "provenance": provenance,
        "safety": {
            "diagnosis": False,
            "prescribing": False,
            "treatment_recommendation": False,
            "autonomous_decision": False,
            "clinical_record_write": False,
            "requires_clinician_judgment": True,
            "emergency_interrupts_routine_flow": emergency,
        },
        "limitations": limitations,
    }
