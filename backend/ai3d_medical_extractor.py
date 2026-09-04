"""AI-3D: explainable structured medical-information extraction.

Local, deterministic extraction layer for OCR text. It extracts only information
that is explicitly present in the source text and attaches confidence/evidence.
It does not diagnose, infer missing values, or normalize ambiguous clinical facts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip(" \t:;,-")


def _confidence(kind: str, matched: str, *, unit: bool = False, context: bool = False) -> float:
    score = 0.72
    if unit:
        score += 0.10
    if context:
        score += 0.08
    if len(matched) >= 4:
        score += 0.04
    if kind in {"date", "measurement"} and unit:
        score += 0.03
    return round(min(score, 0.98), 3)


def _item(category: str, label: str, value: str, evidence: str, confidence: float, **extra: Any) -> Dict[str, Any]:
    out = {
        "category": category,
        "label": label,
        "value": _clean(value),
        "confidence": confidence,
        "needs_review": confidence < 0.80,
        "evidence": _clean(evidence)[:240],
        "source": "AI-3D OCR/text extraction",
    }
    out.update(extra)
    return out


MEASUREMENT_PATTERNS = [
    ("Hemoglobin", r"(?:ha?emoglobin|\bhb\b)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(g\s*/?\s*dl)?", "g/dL"),
    ("Blood glucose", r"(?:blood\s+glucose|glucose|blood\s+sugar)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg\s*/?\s*dl|mmol\s*/?\s*l)?", "mg/dL"),
    ("Creatinine", r"creatinine\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg\s*/?\s*dl|µ?mol\s*/?\s*l)?", "mg/dL"),
    ("Blood pressure", r"(?:blood\s+pressure|\bbp\b)\s*[:=\-]?\s*(\d{2,3}\s*/\s*\d{2,3})\s*(mm\s*hg)?", "mmHg"),
    ("Temperature", r"(?:temperature|\btemp\b)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(°?\s*[cf])?", "°C"),
    ("Pulse", r"(?:pulse|heart\s+rate|\bhr\b)\s*[:=\-]?\s*(\d{2,3})\s*(bpm)?", "bpm"),
    ("Respiratory rate", r"(?:respiratory\s+rate|\brr\b)\s*[:=\-]?\s*(\d{1,3})\s*(?:/min|per\s*min|breaths?\s*/?\s*min)?", "/min"),
    ("SpO2", r"(?:spo2|oxygen\s+saturation|o2\s+saturation)\s*[:=\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(%)?", "%"),
    ("Weight", r"(?:weight)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|lb|lbs)?", "kg"),
    ("Height", r"(?:height)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(cm|m|ft|feet|in)?", "cm"),
]

DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})\b", re.I)

MED_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9+.-]{2,}(?:\s+[A-Z][A-Za-z0-9+.-]{2,}){0,2})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|ml|mL|IU)\b",
    re.I,
)


LAB_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9()/% .+_-]{2,60})\s*(?:[:=]|\.{2,}|\s{2,})\s*(\d+(?:\.\d+)?)\s*([A-Za-zµ%/][A-Za-zµ0-9%/.-]{0,15})?\s*$"
)


def extract_structured_medical_data(text: str, document_type: str = "Other") -> Dict[str, Any]:
    raw = text or ""
    if not raw.strip():
        return {"schema_version": "AI-3D.1", "document_type": document_type, "items": [], "needs_review": True, "item_count": 0}

    items: List[Dict[str, Any]] = []
    seen = set()
    lower = raw.lower()

    # Measurements are intentionally explicit: no values are invented or inferred.
    for label, pattern, default_unit in MEASUREMENT_PATTERNS:
        m = re.search(pattern, raw, re.I)
        if not m:
            continue
        value = m.group(1)
        supplied_unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        unit = _clean(supplied_unit) if supplied_unit else None
        evidence = raw[max(0, m.start()-35):min(len(raw), m.end()+35)]
        key = ("measurement", label, value, unit or "")
        if key not in seen:
            seen.add(key)
            items.append(_item("measurement", label, value, evidence, _confidence("measurement", m.group(0), unit=bool(unit), context=label.lower() in lower), unit=unit or default_unit))

    # Prescription-style medication candidates. Keep the complete source line as evidence.
    for line in raw.splitlines():
        line = _clean(line)
        m = MED_RE.search(line)
        if not m or len(line) > 220:
            continue
        medicine = _clean(m.group(1))
        strength = f"{m.group(2)} {m.group(3)}"
        key = ("medication", medicine.lower(), strength.lower())
        if key in seen:
            continue
        seen.add(key)
        items.append(_item("medication", "Medicine", medicine, line, _confidence("medication", line, unit=True, context=document_type.lower() == "prescription"), strength=strength, instructions=line))

    # Common prescription instructions / frequency terms are captured as supporting items.
    if document_type.lower() == "prescription" or any(x in lower for x in ("tablet", "capsule", "prescription", "sig:")):
        for line in raw.splitlines():
            line = _clean(line)
            if not line or len(line) > 180:
                continue
            if re.search(r"\b(?:once|twice|thrice|daily|bd|bid|tid|qid|od|at night|after food|before food|for \d+ days?)\b", line, re.I):
                key = ("instruction", line.lower())
                if key not in seen:
                    seen.add(key)
                    items.append(_item("instruction", "Medication instruction", line, line, 0.82))

    # Dates become timeline candidates, but only when visibly present.
    for m in DATE_RE.finditer(raw):
        value = _clean(m.group(0))
        key = ("date", value.lower())
        if key in seen:
            continue
        seen.add(key)
        context = raw[max(0, m.start()-45):min(len(raw), m.end()+45)]
        label = "Document date"
        if re.search(r"admission", context, re.I): label = "Admission date"
        elif re.search(r"discharge", context, re.I): label = "Discharge date"
        elif re.search(r"follow[- ]?up|review", context, re.I): label = "Follow-up date"
        items.append(_item("date", label, value, context, 0.86))
        if len([x for x in items if x["category"] == "date"]) >= 6:
            break

    # Lab reports often present rows such as Test: value unit. Avoid interpreting every number as a lab.
    if document_type.lower() == "lab report" or re.search(r"\b(?:reference range|laboratory|lab report|test result)\b", lower):
        for line in raw.splitlines():
            line = _clean(line)
            m = LAB_LINE_RE.match(line)
            if not m:
                continue
            label, value, unit = _clean(m.group(1)), m.group(2), _clean(m.group(3) or "")
            if len(label) > 60 or label.lower() in {"result", "date", "patient id"}:
                continue
            key = ("lab", label.lower(), value, unit.lower())
            if key in seen:
                continue
            seen.add(key)
            items.append(_item("lab_result", label, value, line, 0.84 if unit else 0.75, unit=unit or None))

    items.sort(key=lambda x: (x["category"], x["label"]))
    return {
        "schema_version": "AI-3D.1",
        "document_type": document_type,
        "items": items,
        "item_count": len(items),
        "needs_review": any(x["needs_review"] for x in items) or not items,
        "review_reason": "Verify extracted values against the original document before clinical use." if items else "No structured medical items could be extracted from readable text.",
    }
