"""AI-3G: provenance-first explainability for document-derived clinical data.

AI-3G explains *where* an AI-derived value came from and what verification state it
has. It never upgrades unverified data into clinical truth and never generates a
new diagnosis or treatment recommendation.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List
import json

SCHEMA_VERSION = "AI-3G.1"


def _json(value: Any, default: Any):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _item_explanation(item: Dict[str, Any], index: int, *, verified: bool) -> Dict[str, Any]:
    confidence = item.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "index": index,
        "category": item.get("category"),
        "label": item.get("label"),
        "value": item.get("value"),
        "confidence": confidence,
        "verified": bool(item.get("verified", False) if verified else False),
        "needs_review": bool(item.get("needs_review", True)),
        "evidence": item.get("evidence") or "No source evidence was retained for this item.",
        "method": item.get("method") or "AI-3D structured extraction",
        "provenance": "human-verified extraction" if item.get("verified") else "AI extraction from stored document text",
    }


def explain_document(document: Any) -> Dict[str, Any]:
    extraction = _json(getattr(document, "structured_extraction", None), {})
    verified_data = _json(getattr(document, "verified_data", None), None)
    base_items = extraction.get("items", []) if isinstance(extraction, dict) else []
    active = verified_data if isinstance(verified_data, dict) and verified_data.get("items") else extraction
    items = active.get("items", []) if isinstance(active, dict) else []
    status = getattr(document, "verification_status", None) or "Pending"

    explanations = [_item_explanation(x, i, verified=status in {"Verified", "Partially Verified"})
                    for i, x in enumerate(items) if isinstance(x, dict)]
    review_items = [x["index"] for x in explanations if x["needs_review"] or not x["verified"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": getattr(document, "id", None),
        "filename": getattr(document, "filename", None),
        "document_type": getattr(document, "document_type", None),
        "classification": {
            "document_class": getattr(document, "classification", None) or getattr(document, "document_type", None),
            "confidence": float(getattr(document, "classification_confidence", 0) or 0),
            "method": getattr(document, "classification_method", None),
            "evidence": _json(getattr(document, "classification_evidence", None), []),
            "needs_review": bool(getattr(document, "classification_needs_review", 0)),
        },
        "verification": {
            "status": status,
            "verified_by": getattr(document, "verified_by", None),
            "verified_at": getattr(document, "verified_at", None).isoformat() if getattr(document, "verified_at", None) else None,
            "notes": getattr(document, "verification_notes", None) or "",
        },
        "items": explanations,
        "review_required": bool(review_items),
        "review_item_indexes": review_items,
        "provenance_chain": [
            "stored medical document",
            "AI-3B OCR/text extraction",
            "AI-3C document classification",
            "AI-3D structured extraction",
            "AI-3E human verification (when completed)",
        ],
        "limitations": [
            "Evidence shows source/provenance, not proof of a diagnosis.",
            "Unverified or low-confidence extraction must be confirmed against the original document.",
            "AI-3G does not prescribe, diagnose, or silently modify clinical values.",
        ],
    }


def explain_timeline(timeline: Dict[str, Any]) -> Dict[str, Any]:
    events = timeline.get("events", []) if isinstance(timeline, dict) else []
    trace = []
    for index, event in enumerate(events):
        trace.append({
            "event_index": index,
            "event_type": event.get("event_type"),
            "title": event.get("title"),
            "occurred_at": event.get("occurred_at"),
            "source": event.get("source"),
            "source_id": event.get("source_id"),
            "verification_status": event.get("verification_status"),
            "evidence_available": bool(event.get("details", {}).get("evidence_available", True)),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "timeline_type": timeline.get("timeline_type", "longitudinal_clinical_record"),
        "event_count": len(trace),
        "event_trace": trace,
        "limitations": [
            "Timeline ordering uses only timestamps already stored in the record.",
            "No missing date is inferred and no event is invented.",
            "Clinical interpretation remains the responsibility of the physician.",
        ],
    }
