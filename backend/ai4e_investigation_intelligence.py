"""AI-4E: conservative investigation intelligence.

Builds a longitudinal, source-traceable view of explicitly verified investigation
results. It reports numerical trends and record inconsistencies for clinician
review; it does not label results normal/abnormal without a reliable reference
range and does not diagnose or recommend treatment.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "AI-4E.1"


def _text(v: Any) -> str:
    return "" if v in (None, "") else str(v).strip()


def _load(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v or "")
    except (TypeError, ValueError):
        return {}


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(v).lower()).strip()


def _number(v: Any) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", _text(v).replace(",", ""))
    return float(m.group(0)) if m else None


def _iso(v: Any) -> Optional[str]:
    if isinstance(v, datetime):
        return v.isoformat()
    s = _text(v)
    return s or None


def _extract_date(items: List[Dict[str, Any]]) -> Optional[str]:
    for item in items:
        if _norm(item.get("category")) == "date":
            value = _text(item.get("value"))
            if value:
                return value
    return None


def _verified_items(doc: Any) -> List[Dict[str, Any]]:
    payload = _load(getattr(doc, "verified_data", None))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [x for x in items if isinstance(x, dict) and x.get("verified") is True]


def _is_investigation_item(item: Dict[str, Any], doc_type: str) -> bool:
    category = _norm(item.get("category"))
    label = _norm(item.get("label"))
    dtype = _norm(doc_type)
    return (
        category in {"lab result", "measurement"}
        or "lab report" in dtype
        or "investigation" in dtype
        or "radiology" in dtype
        or "imaging" in dtype
    ) and not (category == "medication")


def build_investigation_intelligence(patient: Any, documents: Iterable[Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    by_test: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    imaging_findings: List[Dict[str, Any]] = []

    for doc in documents:
        doc_type = _text(getattr(doc, "classification", None) or getattr(doc, "document_type", None) or "Other")
        items = _verified_items(doc)
        explicit_date = _extract_date(items)
        source_date = _iso(getattr(doc, "created_at", None))
        for item in items:
            if not _is_investigation_item(item, doc_type):
                continue
            category = _norm(item.get("category"))
            label = _text(item.get("label")) or "Investigation"
            value = _text(item.get("value"))
            if not value:
                continue
            unit = _text(item.get("unit"))
            rec = {
                "id": f"doc_{getattr(doc, 'id', 'unknown')}_{len(records)+1}",
                "test": label,
                "value": value,
                "numeric_value": _number(value),
                "unit": unit or None,
                "category": "lab_result" if category == "lab result" else category or "investigation",
                "document_id": getattr(doc, "id", None),
                "filename": getattr(doc, "filename", None),
                "document_type": doc_type,
                "date": explicit_date or source_date,
                "date_basis": "document_explicit_date" if explicit_date else ("document_created_at" if source_date else "undated"),
                "evidence": _text(item.get("evidence"))[:240],
                "verified": True,
            }
            records.append(rec)
            key = _norm(label)
            if key:
                by_test[key].append(rec)

            if category in {"imaging finding", "finding", "impression"} or "imaging" in _norm(doc_type) or "radiology" in _norm(doc_type):
                imaging_findings.append({
                    "id": rec["id"], "label": label, "value": value,
                    "document_id": rec["document_id"], "filename": rec["filename"],
                    "date": rec["date"], "evidence": rec["evidence"],
                })

    records.sort(key=lambda x: (x["date"] is None, x["date"] or "", x["test"].lower()))
    trends: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []

    for key, entries in by_test.items():
        if len(entries) < 2:
            continue
        # Only compare numeric values with the same explicit unit. If units differ,
        # surface the inconsistency instead of converting without a validated unit map.
        units = {_norm(x.get("unit")) for x in entries}
        if len(units - {""}) > 1:
            alerts.append({
                "id": "unit_inconsistency_" + re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:60],
                "priority": "medium", "type": "unit_inconsistency",
                "title": "Investigation has differing recorded units",
                "test": entries[0]["test"],
                "text": "Confirm units before comparing results across records.",
                "sources": [{"document_id": x["document_id"], "unit": x["unit"], "date": x["date"]} for x in entries],
                "requires_clinician_judgment": True,
            })
            continue
        numeric = [x for x in entries if x["numeric_value"] is not None]
        if len(numeric) < 2:
            continue
        # Preserve source ordering by date when parseable; otherwise retain record order.
        dated = sorted(numeric, key=lambda x: x["date"] or "")
        first, last = dated[0], dated[-1]
        delta = round(last["numeric_value"] - first["numeric_value"], 6)
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"
        trends.append({
            "test": first["test"],
            "unit": first["unit"] or None,
            "observations": [{"date": x["date"], "value": x["value"], "document_id": x["document_id"], "filename": x["filename"]} for x in dated],
            "direction": direction,
            "absolute_change": delta,
            "interpretation": "Observed numerical change; clinical significance requires clinician review and an appropriate reference range/context.",
            "requires_clinician_judgment": True,
        })
        if direction != "unchanged":
            alerts.append({
                "id": "trend_" + re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:60],
                "priority": "low", "type": "observed_trend",
                "title": f"Observed change in {first['test']}",
                "text": f"Recorded values {direction}; review the original reports and clinical context before interpreting significance.",
                "test": first["test"], "requires_clinician_judgment": True,
            })

    # A verified result without a reference range must never be called abnormal.
    reference_gap_count = sum(1 for x in records if x["category"] == "lab_result" and not x.get("reference_range"))
    if not records:
        alerts.append({
            "id": "investigation_history_missing", "priority": "medium", "type": "information_gap",
            "title": "No verified investigation results are available",
            "text": "Relevant investigations may need to be confirmed from the patient or source records.",
            "requires_clinician_judgment": True,
        })

    alerts.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("priority"), 9))
    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "headline": "Longitudinal investigation review for clinician use",
        "investigations": records,
        "trends": trends,
        "imaging_findings": imaging_findings,
        "review_alerts": alerts,
        "statistics": {
            "verified_investigation_records": len(records),
            "tests_with_multiple_numeric_observations": sum(1 for v in by_test.values() if len([x for x in v if x["numeric_value"] is not None]) >= 2),
            "observed_trends": len(trends),
        },
        "safety": {
            "uses_verified_document_items_only": True,
            "normal_abnormal_classification_performed": False,
            "diagnosis": False,
            "treatment_recommendation": False,
            "automatic_unit_conversion": False,
            "requires_clinician_judgment": True,
        },
        "limitations": [
            "Only explicitly verified document items are included.",
            "A numerical trend is descriptive and is not a diagnosis or a statement of clinical severity.",
            "Normal/abnormal classification is not performed without a reliable reference range and context.",
            "Different units are not automatically converted.",
            "When an explicit investigation date is unavailable, document creation time is labelled as the date basis rather than assumed to be the test date.",
        ],
    }
