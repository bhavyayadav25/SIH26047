"""AI-5G: aggregate hospital analytics.

Analytics are operational summaries, not clinical decision-making.  This module
keeps metrics aggregate and source-derived: it never infers diagnoses,
clinical safety, or treatment recommendations from the numbers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional


def _parse_date(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must use YYYY-MM-DD format.")


def _date_bounds(start_date: Optional[str], end_date: Optional[str]):
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start and end and start > end:
        raise ValueError("start_date must be on or before end_date.")
    return start, end


def _in_range(day: str, start, end) -> bool:
    try:
        current = datetime.strptime(day, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    return (start is None or current >= start) and (end is None or current <= end)


def _iso(dt):
    return dt.isoformat() if dt else None


def build_analytics_dashboard(
    db: Any,
    *,
    Encounter: Any,
    Consultation: Any,
    MedicalDocument: Any,
    AuditEvent: Any,
    User: Any,
    Department: Any,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    start, end = _date_bounds(start_date, end_date)

    encounters = db.query(Encounter).all()
    encounters = [e for e in encounters if _in_range(e.visit_date, start, end)]

    encounter_ids = {e.id for e in encounters}
    patient_ids = {e.patient_id for e in encounters}
    doctor_ids = {e.doctor_id for e in encounters if e.doctor_id}

    consultations = db.query(Consultation).all()
    consultations = [c for c in consultations if c.encounter_id in encounter_ids]

    documents = db.query(MedicalDocument).all()
    documents = [d for d in documents if d.patient_id in patient_ids and _in_range((d.created_at.date().isoformat() if d.created_at else ""), start, end)]

    # Operational wait-time metric: only completed call timestamps are used.
    wait_samples = []
    for e in encounters:
        if e.called_at and e.created_at:
            seconds = (e.called_at - e.created_at).total_seconds()
            if seconds >= 0:
                wait_samples.append(seconds / 60.0)

    # Consultation turnaround is intentionally only measured when a consultation
    # is linked to an encounter and has reached a completed state.
    consult_samples = []
    for c in consultations:
        if c.consultation_status == "completed" and c.created_at and c.updated_at:
            seconds = (c.updated_at - c.created_at).total_seconds()
            if seconds >= 0:
                consult_samples.append(seconds / 60.0)

    status_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {}
    department_counts: Dict[str, int] = {}
    for e in encounters:
        status_counts[e.status] = status_counts.get(e.status, 0) + 1
        priority_counts[e.priority] = priority_counts.get(e.priority, 0) + 1
        department_counts[e.department] = department_counts.get(e.department, 0) + 1

    risk_counts: Dict[str, int] = {}
    for c in consultations:
        risk = (c.risk_level or "none").lower()
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    verification_counts: Dict[str, int] = {}
    for d in documents:
        status = d.verification_status or "Pending"
        verification_counts[status] = verification_counts.get(status, 0) + 1

    # Daily volume is deliberately derived from the encounter's visit_date, not
    # from server-local timestamps, so reports remain stable across timezones.
    daily_volume: Dict[str, int] = {}
    for e in encounters:
        daily_volume[e.visit_date] = daily_volume.get(e.visit_date, 0) + 1

    active_departments = db.query(Department).filter(Department.active == 1).count()
    active_doctors = db.query(User).filter(User.role == "doctor").count()

    audit_query = db.query(AuditEvent)
    audit_events = audit_query.count()

    return {
        "schema_version": "AI-5G.1",
        "period": {"start_date": start.isoformat() if start else None, "end_date": end.isoformat() if end else None},
        "scope": "aggregate operational analytics; no individual clinical conclusions",
        "overview": {
            "encounters": len(encounters),
            "unique_patients": len(patient_ids),
            "assigned_doctors": len(doctor_ids),
            "consultations": len(consultations),
            "documents": len(documents),
            "active_departments": active_departments,
            "active_doctors": active_doctors,
        },
        "queue": {
            "status_counts": dict(sorted(status_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "department_counts": dict(sorted(department_counts.items())),
        },
        "clinical_review": {
            "risk_level_counts": dict(sorted(risk_counts.items())),
            "document_verification_counts": dict(sorted(verification_counts.items())),
            "note": "Risk counts summarize recorded AI/consultation risk labels; they do not establish patient safety or diagnosis.",
        },
        "performance": {
            "average_wait_minutes": round(sum(wait_samples) / len(wait_samples), 2) if wait_samples else None,
            "wait_sample_count": len(wait_samples),
            "average_completed_consultation_minutes": round(sum(consult_samples) / len(consult_samples), 2) if consult_samples else None,
            "consultation_duration_sample_count": len(consult_samples),
        },
        "daily_encounter_volume": [
            {"date": day, "encounters": daily_volume[day]}
            for day in sorted(daily_volume)
        ],
        "audit": {"audit_event_count": audit_events},
        "data_quality": {
            "ai_correction_rate": None,
            "ai_correction_rate_reason": "Not measured because the current schema does not record a normalized AI-correction event denominator.",
        },
        "safety_boundary": {
            "diagnosis": False,
            "prescribing": False,
            "autonomous_decision": False,
            "individual_patient_scoring": False,
        },
    }
