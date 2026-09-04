"""AI-5E: clinician-facing triage dashboard and workflow helpers.

Operational triage only. Clinical priority is never inferred autonomously by
this module; the AI-4B risk assessment is surfaced as decision support and the
triage clinician remains responsible for acknowledgement/escalation.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "AI-5E.1"
TRIAGE_ROLES = {"doctor", "triage", "admin"}
TRIAGE_STATUSES = {"unreviewed", "acknowledged", "escalated", "resolved"}
ACTIONS = {"acknowledge", "escalate", "resolve"}


def normalize_triage_status(value: str | None) -> str:
    value = (value or "unreviewed").strip().lower()
    if value not in TRIAGE_STATUSES:
        raise ValueError("Invalid triage status.")
    return value


def normalize_triage_action(value: str) -> str:
    value = (value or "").strip().lower()
    if value not in ACTIONS:
        raise ValueError("Invalid triage action.")
    return value


def triage_action_status(action: str) -> str:
    return {"acknowledge": "acknowledged", "escalate": "escalated", "resolve": "resolved"}[action]


def triage_rank(item: Dict[str, Any]):
    priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
    risk_rank = {"emergency": 0, "urgent": 1, "none": 2}
    return (
        priority_rank.get(str(item.get("priority", "normal")).lower(), 2),
        risk_rank.get(str(item.get("risk_level", "none")).lower(), 2),
        item.get("created_at") or "",
        item.get("encounter_id") or 0,
    )


def serialize_triage_item(encounter: Any, patient: Any = None, doctor: Any = None,
                           risk: Dict[str, Any] | None = None) -> Dict[str, Any]:
    risk = risk or {}
    return {
        "encounter_id": encounter.id,
        "patient_id": encounter.patient_id,
        "patient_name": getattr(patient, "name", None),
        "doctor_id": encounter.doctor_id,
        "doctor_name": getattr(doctor, "name", None),
        "department": encounter.department,
        "visit_date": encounter.visit_date,
        "token_number": encounter.token_number,
        "priority": encounter.priority,
        "queue_status": encounter.status,
        "triage_status": getattr(encounter, "triage_status", None) or "unreviewed",
        "triage_notes": getattr(encounter, "triage_notes", None) or "",
        "reason": encounter.reason or "",
        "risk_level": risk.get("risk_level", "none"),
        "risk_message": risk.get("message", ""),
        "alerts": risk.get("alerts", []),
        "requires_human_review": True,
        "interrupt_routine_flow": bool(risk.get("interrupt_routine_flow", False)),
        "created_at": encounter.created_at.isoformat() if encounter.created_at else None,
        "called_at": encounter.called_at.isoformat() if encounter.called_at else None,
    }


def build_triage_dashboard(items: Iterable[Dict[str, Any]], department: str, visit_date: str) -> Dict[str, Any]:
    rows = list(items)
    rows.sort(key=triage_rank)
    for index, row in enumerate(rows, 1):
        row["triage_position"] = index
    counts = {status: 0 for status in TRIAGE_STATUSES}
    for row in rows:
        status = row.get("triage_status", "unreviewed")
        counts[status] = counts.get(status, 0) + 1
    emergency = sum(1 for row in rows if row.get("risk_level") == "emergency" or row.get("priority") == "emergency")
    urgent = sum(1 for row in rows if row.get("risk_level") == "urgent" or row.get("priority") == "urgent")
    return {
        "schema_version": SCHEMA_VERSION,
        "department": department,
        "visit_date": visit_date,
        "count": len(rows),
        "queue": rows,
        "summary": {
            "emergency_attention": emergency,
            "urgent_attention": urgent,
            "triage_status_counts": counts,
        },
        "safety": {
            "human_triage_required": True,
            "ai_does_not_assign_priority": True,
            "no_alert_does_not_mean_clinical_safety": True,
        },
        "limitations": [
            "AI-4B output is decision support and does not replace clinician triage.",
            "Queue priority is operational and must be confirmed by authorized staff.",
            "A missing AI alert does not establish that a patient is clinically safe.",
        ],
    }
