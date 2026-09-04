"""AI-3F: longitudinal clinical timeline builder.

AI-3F consolidates existing, timestamped patient-record data into a chronological
view. It does not diagnose, invent dates, or convert unverified AI extraction into
verified clinical facts. Document-derived events are marked with their verification
state and retain source/evidence references.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _event(event_type: str, occurred_at: Any, title: str, summary: str,
           source: str, source_id: Any, verification: str = "Not Applicable",
           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "event_type": event_type,
        "occurred_at": _iso(occurred_at),
        "title": title,
        "summary": summary,
        "source": source,
        "source_id": source_id,
        "verification_status": verification,
        "details": details or {},
    }


def _safe_json(value: Any, default: Any) -> Any:
    import json
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def build_document_events(documents: Iterable[Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for doc in documents:
        status = getattr(doc, "verification_status", None) or "Pending"
        verified = _safe_json(getattr(doc, "verified_data", None), None)
        extraction = verified if isinstance(verified, dict) and verified else _safe_json(
            getattr(doc, "structured_extraction", None), {}
        )
        items = extraction.get("items", []) if isinstance(extraction, dict) else []
        confirmed_items = [
            {"label": x.get("label"), "value": x.get("value"), "category": x.get("category")}
            for x in items if isinstance(x, dict)
        ]
        # The document event itself is always safe to show; its clinical details retain status.
        events.append(_event(
            "document", getattr(doc, "created_at", None),
            getattr(doc, "document_type", None) or "Medical document",
            f"{getattr(doc, 'filename', 'Document')} processed for clinical review.",
            "medical_document", getattr(doc, "id", None), status,
            {
                "filename": getattr(doc, "filename", None),
                "document_type": getattr(doc, "document_type", None),
                "classification": getattr(doc, "classification", None) or getattr(doc, "document_type", None),
                "extracted_items": confirmed_items,
                "source_date": None,
                "evidence_available": bool(getattr(doc, "extracted_text", None)),
            }
        ))
    return events


def build_consultation_events(consultations: Iterable[Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for c in consultations:
        details = {
            "risk_level": getattr(c, "risk_level", None) or "none",
            "doctor_review": getattr(c, "doctor_review", None) or "Pending",
            "red_flags": _safe_json(getattr(c, "red_flags", None), []),
            "doctor_notes": getattr(c, "doctor_notes", None) or "",
            "ai_summary_available": bool(getattr(c, "ai_summary", None)),
        }
        events.append(_event(
            "consultation", getattr(c, "created_at", None),
            getattr(c, "title", None) or "Clinical consultation",
            getattr(c, "summary", None) or "Consultation record.",
            "consultation", getattr(c, "id", None),
            getattr(c, "doctor_review", None) or "Pending", details
        ))
    return events


def build_timeline(consultations: Iterable[Any], documents: Iterable[Any]) -> Dict[str, Any]:
    events = build_consultation_events(consultations) + build_document_events(documents)
    # Unknown timestamps are placed last; no synthetic date is generated.
    events.sort(key=lambda x: (x["occurred_at"] is None, x["occurred_at"] or ""), reverse=False)
    return {
        "schema_version": "AI-3F.1",
        "timeline_type": "longitudinal_clinical_record",
        "event_count": len(events),
        "events": events,
        "limitations": [
            "Only dates already present in the stored record are used.",
            "Unverified document extraction remains labeled and must not be treated as confirmed clinical fact.",
            "AI-3F provides organization and traceability, not diagnosis or treatment recommendations.",
        ],
    }
