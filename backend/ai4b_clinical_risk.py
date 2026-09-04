"""AI-4B: conservative clinical risk and red-flag intelligence.

This module aggregates already-stored patient information and applies the
existing transparent AI-2 safety rules. It is intentionally not diagnostic,
does not prescribe treatment, and never upgrades an unverified document item
to a verified clinical fact.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, Iterable, List

from ai2_safety import evaluate_safety, PRIORITY

SCHEMA_VERSION = "AI-4B.1"


def _json(value: Any, default: Any):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _doc_items(document: Any) -> List[Dict[str, Any]]:
    data = _json(getattr(document, "verified_data", None), None)
    if not isinstance(data, dict) or not data:
        data = _json(getattr(document, "structured_extraction", None), {})
    items = data.get("items", []) if isinstance(data, dict) else []
    return [x for x in items if isinstance(x, dict)]


def _consultation_payload(consultations: Iterable[Any]) -> List[Dict[str, Any]]:
    out = []
    for c in consultations:
        structured = _json(getattr(c, "structured_data", None), {})
        nlp = _json(getattr(c, "nlp_data", None), {})
        out.append({
            "consultation_id": getattr(c, "id", None),
            "created_at": str(getattr(c, "created_at", "") or ""),
            "title": getattr(c, "title", "") or "",
            "structured": structured if isinstance(structured, dict) else {},
            "nlp": nlp if isinstance(nlp, dict) else {},
            "risk_level": getattr(c, "risk_level", None) or "none",
            "red_flags": _json(getattr(c, "red_flags", None), []),
        })
    return out


def _source_text(patient: Any, consultations: List[Any], documents: List[Any]) -> str:
    profile = getattr(patient, "profile", None)
    chunks = []
    if profile:
        for key in ("allergies", "conditions", "medications"):
            value = getattr(profile, key, None)
            if value:
                chunks.append(f"{key}: {value}")

    for c in consultations:
        structured = _json(getattr(c, "structured_data", None), {})
        nlp = _json(getattr(c, "nlp_data", None), {})
        if isinstance(structured, dict):
            chunks.extend(f"{k}: {v}" for k, v in structured.items() if v not in (None, "", [], {}))
        if isinstance(nlp, dict):
            chunks.extend(f"{k}: {v}" for k, v in nlp.items() if v not in (None, "", [], {}))

    for d in documents:
        # Only feed verified document values into the risk engine. Unverified
        # extraction is surfaced separately and cannot trigger a clinical alert.
        for item in _doc_items(d):
            if item.get("verified") is True:
                label = item.get("label") or item.get("category") or "finding"
                chunks.append(f"verified document {label}: {item.get('value', '')}")
    return " ".join(chunks)


def _document_review_state(documents: Iterable[Any]) -> Dict[str, Any]:
    pending = []
    reviewed = 0
    total = 0
    for d in documents:
        for item in _doc_items(d):
            total += 1
            if item.get("verified") is True:
                reviewed += 1
            else:
                pending.append({
                    "document_id": getattr(d, "id", None),
                    "filename": getattr(d, "filename", None),
                    "label": item.get("label"),
                    "value": item.get("value"),
                    "evidence": item.get("evidence", ""),
                    "status": "unverified",
                })
    return {"total_extracted_items": total, "verified_items": reviewed, "pending_items": pending}


def build_risk_assessment(patient: Any, consultations: Iterable[Any], documents: Iterable[Any], summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    consultations = list(consultations)
    documents = list(documents)
    summary = summary or {}

    source = _source_text(patient, consultations, documents)
    safety = evaluate_safety(source, {})

    # Preserve stored consultation risk signals as supporting evidence, but
    # never downgrade an explicit higher-priority safety result.
    stored_alerts: List[Dict[str, Any]] = []
    for c in consultations:
        flags = _json(getattr(c, "red_flags", None), [])
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, dict):
                    stored_alerts.append({
                        "source": "consultation",
                        "consultation_id": getattr(c, "id", None),
                        "level": flag.get("level", getattr(c, "risk_level", "none") or "none"),
                        "label": flag.get("label") or flag.get("id") or "Stored safety alert",
                        "evidence": flag.get("evidence", []),
                    })

    all_alerts = []
    for alert in safety.get("alerts", []):
        a = dict(alert)
        a["source"] = "AI-2 safety rules over verified/current intake data"
        all_alerts.append(a)
    all_alerts.extend(stored_alerts)

    # De-duplicate by label/level while retaining the strongest evidence.
    dedup: Dict[str, Dict[str, Any]] = {}
    for alert in all_alerts:
        key = f"{alert.get('level','none')}::{alert.get('label','')}".lower()
        if key not in dedup:
            dedup[key] = alert
        else:
            existing = dedup[key]
            ev = list(existing.get("evidence", []) or []) + list(alert.get("evidence", []) or [])
            existing["evidence"] = list(dict.fromkeys(str(x) for x in ev))[:8]

    alerts = sorted(dedup.values(), key=lambda a: PRIORITY.get(a.get("level", "none"), 0), reverse=True)
    level = alerts[0].get("level", "none") if alerts else "none"

    review = _document_review_state(documents)
    if review["pending_items"]:
        review_status = "review_required"
    else:
        review_status = "review_ready"

    action = "immediate_human_triage" if level == "emergency" else (
        "prompt_clinical_review" if level == "urgent" else "routine_clinical_review"
    )

    # Avoid claiming that absence of a rule match means the patient is safe.
    message = (
        "Potentially serious warning information was detected by the prototype rules. Immediate human triage is required."
        if level == "emergency" else
        "Potential warning information was detected. Prompt clinician review is recommended."
        if level == "urgent" else
        "No configured AI-4B red-flag rule matched the available verified/current information. This does not establish clinical safety."
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "risk_level": level,
        "recommended_action": action,
        "message": message,
        "alerts": alerts,
        "requires_human_review": True,
        "interrupt_routine_flow": level == "emergency",
        "document_review": review,
        "summary_review_status": summary.get("review_status"),
        "source_scope": {
            "patient_profile": True,
            "consultations": len(consultations),
            "documents": len(documents),
            "unverified_document_items_used_for_alerting": False,
        },
        "explainability": {
            "engine": "AI-4B / AI-2 conservative rule engine",
            "rule_based": True,
            "diagnosis": None,
            "treatment_recommendation": None,
            "evidence_is_traceable": True,
        },
        "limitations": [
            "This is a triage-support screen, not a diagnosis or prediction of disease.",
            "A missing alert does not mean the patient is clinically safe.",
            "Only verified document findings are eligible to trigger document-based alerts.",
            "Stored historical alerts are retained as context and require clinician review.",
            "Rules require clinical validation before real-world deployment.",
        ],
    }
