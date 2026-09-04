"""AI-4D: conservative medication reconciliation intelligence.

AI-4D consolidates medication information already present in the record and
surfaces possible reconciliation issues. It does not prescribe, discontinue,
change doses, or claim a drug interaction from incomplete local data.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, Iterable, List, Tuple

SCHEMA_VERSION = "AI-4D.1"


def _text(v: Any) -> str:
    return "" if v in (None, "") else str(v).strip()


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(v).lower()).strip()


def _load(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v or "")
    except (TypeError, ValueError):
        return {}


def _medicine_key(name: str) -> str:
    # Keep the name conservative: remove obvious strength/form suffixes but do
    # not attempt brand-to-generic conversion without a drug dictionary.
    s = _norm(name)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|%)\b", " ", s)
    s = re.sub(r"\b(?:tablet|tablets|tab|capsule|capsules|cap|syrup|injection|inj|cream|ointment|drops?)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_profile_meds(raw: str) -> List[Dict[str, Any]]:
    if not raw.strip():
        return []
    result = []
    # Profile medication text is free-form. Treat each non-empty line or
    # semicolon/comma-delimited entry as a patient-reported item, without
    # pretending the strength/frequency is reliably structured.
    parts = [p.strip() for p in re.split(r"[\n;]+", raw) if p.strip()]
    for p in parts:
        result.append({"name": p, "key": _medicine_key(p), "source": "patient profile", "verified": False})
    return result


def _verified_document_meds(documents: Iterable[Any]) -> List[Dict[str, Any]]:
    result = []
    for d in documents:
        payload = _load(getattr(d, "verified_data", None))
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict) or item.get("verified") is not True:
                continue
            category = _norm(item.get("category"))
            label = _norm(item.get("label"))
            if "medic" not in category and "prescription" not in category and "medic" not in label:
                continue
            name = _text(item.get("value") or item.get("name"))
            if not name:
                continue
            result.append({
                "name": name,
                "key": _medicine_key(name),
                "strength": _text(item.get("strength")),
                "instructions": _text(item.get("instructions")),
                "source": "verified document",
                "document_id": getattr(d, "id", None),
                "filename": getattr(d, "filename", None),
                "evidence": _text(item.get("evidence")),
                "verified": True,
            })
    return result


def _unique(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    out = []
    for item in items:
        key = (item.get("key", ""), _norm(item.get("name")))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_medication_intelligence(patient: Any, documents: Iterable[Any], consultations: Iterable[Any]) -> Dict[str, Any]:
    profile = getattr(patient, "profile", None)
    profile_raw = _text(getattr(profile, "medications", "")) if profile else ""
    profile_items = _parse_profile_meds(profile_raw)
    document_items = _verified_document_meds(documents)

    current = _unique(profile_items + document_items)
    alerts: List[Dict[str, Any]] = []
    discrepancies: List[Dict[str, Any]] = []

    # Possible duplicate entries are surfaced only when the exact normalized
    # medication key repeats. We never infer therapeutic duplication across
    # different drug names.
    key_sources: Dict[str, List[Dict[str, Any]]] = {}
    for item in profile_items + document_items:
        key_sources.setdefault(item["key"], []).append(item)
    for key, entries in key_sources.items():
        if key and len(entries) > 1:
            names = sorted({e["name"] for e in entries})
            if len(names) == 1:
                alerts.append({
                    "id": "duplicate_medication_record_" + re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:60],
                    "priority": "medium",
                    "type": "possible_duplicate_record",
                    "title": "Medication appears in multiple records",
                    "text": "Confirm whether these entries represent the same active medicine before relying on the consolidated list.",
                    "sources": sorted({e["source"] for e in entries}),
                    "medicine": names[0],
                    "requires_clinician_judgment": True,
                })

    # Compare profile and verified-document entries. A missing profile item is
    # not called a discontinuation; it is simply a reconciliation difference.
    profile_keys = {x["key"] for x in profile_items if x["key"]}
    doc_keys = {x["key"] for x in document_items if x["key"]}
    for item in document_items:
        if item["key"] and item["key"] not in profile_keys:
            discrepancies.append({
                "id": "document_not_in_profile_" + re.sub(r"[^a-z0-9]+", "_", item["key"]).strip("_")[:60],
                "priority": "medium",
                "type": "reconciliation_difference",
                "title": "Verified document medicine is not in the profile list",
                "text": "Confirm whether this medicine is current, historical, or intentionally omitted from the profile.",
                "medicine": item["name"],
                "source": item["filename"] or "verified document",
                "requires_clinician_judgment": True,
            })
    for item in profile_items:
        if item["key"] and item["key"] not in doc_keys and document_items:
            discrepancies.append({
                "id": "profile_not_in_document_" + re.sub(r"[^a-z0-9]+", "_", item["key"]).strip("_")[:60],
                "priority": "low",
                "type": "reconciliation_difference",
                "title": "Profile medicine not found in verified documents",
                "text": "Confirm whether this medicine remains current and whether a supporting prescription is available.",
                "medicine": item["name"],
                "source": "patient profile",
                "requires_clinician_judgment": True,
            })

    if not profile_items and not document_items:
        alerts.append({
            "id": "medication_history_missing",
            "priority": "high",
            "type": "information_gap",
            "title": "Current medication list is not available",
            "text": "Confirm current medicines directly with the patient or clinician workflow when relevant to this encounter.",
            "requires_clinician_judgment": True,
        })

    # Allergies are intentionally not cross-matched against drug names here:
    # doing so safely requires a maintained drug/allergen knowledge base and
    # ingredient normalization. This prototype must not create false alerts.
    allergies = _text(getattr(profile, "allergies", "")) if profile else ""

    alerts.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("priority"), 9))
    discrepancies.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("priority"), 9))

    return {
        "schema_version": SCHEMA_VERSION,
        "patient_id": getattr(patient, "id", None),
        "headline": "Medication reconciliation for clinician review",
        "medications": current,
        "reconciliation_alerts": alerts,
        "discrepancies": discrepancies,
        "allergy_context": {"recorded": bool(allergies), "note": "Allergy-to-drug matching is not performed without a maintained ingredient/allergen knowledge base."},
        "interaction_check": {"performed": False, "reason": "No validated medication interaction knowledge base is bundled in this prototype."},
        "prescribing": False,
        "dose_changes": False,
        "discontinuation_decisions": False,
        "autonomous_decision": False,
        "requires_clinician_judgment": True,
        "evidence_scope": {
            "profile_medications": bool(profile_items),
            "verified_document_medications_only": True,
            "unverified_document_values_used": False,
        },
        "limitations": [
            "This module reconciles recorded medication information; it does not prescribe or change treatment.",
            "A discrepancy does not mean a medicine is wrong, stopped, duplicated, or unsafe.",
            "Interaction and allergy matching require a validated, maintained medication knowledge base and are not performed here.",
            "Medication names are not converted between brands and generic ingredients automatically.",
            "All medication decisions require clinician verification and local protocol.",
        ],
    }
