"""AI-5B: encounter and queue workflow helpers.

This module contains deterministic, non-clinical queue rules. It never assigns a
clinical diagnosis or priority from symptoms; priority is explicitly supplied
by an authorized user or defaults to normal.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Any, Dict

ALLOWED_STATUSES = {"waiting", "called", "in_consultation", "completed", "cancelled"}
ALLOWED_PRIORITIES = {"normal", "urgent", "emergency"}
ROLE_CAN_MANAGE_QUEUE = {"doctor", "triage", "admin"}


def normalize_department(value: str) -> str:
    value = (value or "General Medicine").strip()
    if not value:
        return "General Medicine"
    if len(value) > 80:
        raise ValueError("Department name is too long.")
    return value


def normalize_priority(value: str | None) -> str:
    value = (value or "normal").strip().lower()
    if value not in ALLOWED_PRIORITIES:
        raise ValueError("Invalid queue priority.")
    return value


def normalize_status(value: str) -> str:
    value = (value or "").strip().lower()
    if value not in ALLOWED_STATUSES:
        raise ValueError("Invalid encounter status.")
    return value


def can_transition(current: str, target: str) -> bool:
    allowed = {
        "waiting": {"called", "cancelled"},
        "called": {"in_consultation", "waiting", "cancelled"},
        "in_consultation": {"completed"},
        "completed": set(),
        "cancelled": set(),
    }
    return target in allowed.get(current, set())


def queue_sort_key(encounter: Any):
    priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
    return (
        priority_rank.get((encounter.priority or "normal").lower(), 2),
        encounter.created_at or datetime.min,
        encounter.id or 0,
    )


def serialize_encounter(encounter: Any, patient: Any = None, doctor: Any = None) -> Dict[str, Any]:
    return {
        "id": encounter.id,
        "patient_id": encounter.patient_id,
        "patient_name": getattr(patient, "name", None),
        "doctor_id": encounter.doctor_id,
        "doctor_name": getattr(doctor, "name", None),
        "department": encounter.department,
        "visit_date": encounter.visit_date,
        "token_number": encounter.token_number,
        "priority": encounter.priority,
        "status": encounter.status,
        "reason": encounter.reason or "",
        "created_at": encounter.created_at.isoformat() if encounter.created_at else None,
        "called_at": encounter.called_at.isoformat() if encounter.called_at else None,
        "completed_at": encounter.completed_at.isoformat() if encounter.completed_at else None,
    }
