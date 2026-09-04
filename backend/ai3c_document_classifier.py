"""AI-3C: explainable medical-document classification.

Local, deterministic classifier for the MediKiosk prototype. It classifies
OCR/extracted text into a document category and returns evidence + confidence.
It is deliberately conservative: low-confidence cases are marked for review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

DOCUMENT_CLASSES = [
    "Prescription",
    "Lab Report",
    "Discharge Summary",
    "Imaging Report",
    "Consultation Note",
    "Referral Letter",
    "Operative Report",
    "Vaccination Record",
    "Other",
]

# High-signal phrases get more weight than generic medical words.
PATTERNS: Dict[str, List[tuple[str, float]]] = {
    "Prescription": [
        (r"\bprescription\b", 6), (r"\brx\b", 5), (r"sig(?:nature)?\s*[:.]", 4),
        (r"\btablet\b|\bcapsule\b|\bsyrup\b|\binjection\b", 3),
        (r"\b(?:mg|mcg|ml)\b", 1.5), (r"\bdosage\b|\bdose\b", 3),
        (r"\btake\s+(?:one|two|1|2)\b", 2),
    ],
    "Lab Report": [
        (r"\blaboratory\b|\blab(?:oratory)?\s+report\b", 6),
        (r"\bcbc\b|\bhemoglobin\b|\bcreatinine\b|\bglucose\b", 3),
        (r"\bpatient\s*id\b.*\b(?:reference|range)\b", 2),
        (r"\b(?:reference|normal)\s+range\b", 4),
        (r"\bresult\b\s*[:=]", 2), (r"\bmg/dl\b|\bg/dl\b|\bmmol/l\b", 2),
    ],
    "Discharge Summary": [
        (r"\bdischarge\s+summary\b", 8), (r"\bdischarged\b", 5),
        (r"\badmission\s+date\b|\bdischarge\s+date\b", 4),
        (r"\bhospital\s+course\b", 5), (r"\bdischarge\s+medications?\b", 4),
        (r"\bfollow[- ]?up\b", 1.5),
    ],
    "Imaging Report": [
        (r"\bimpression\s*:", 5), (r"\bfindings\s*:", 3),
        (r"\bx[- ]?ray\b|\bct\s+scan\b|\bcomputed\s+tomography\b", 6),
        (r"\bmri\b|\bmagnetic\s+resonance\b|\bultrasound\b|\bsonography\b", 6),
        (r"\bradiolog(?:y|ical)\b", 5), (r"\btechnique\s*:", 2),
    ],
    "Consultation Note": [
        (r"\bconsultation\s+note\b|\bconsultation\b", 6),
        (r"\bchief\s+complaint\b", 5), (r"\bhistory\s+of\s+present\s+illness\b", 5),
        (r"\bassessment\s*(?:and|&)\s*plan\b", 4), (r"\bphysical\s+examination\b", 4),
        (r"\bclinical\s+notes?\b", 3),
    ],
    "Referral Letter": [
        (r"\breferral\s+(?:letter|note)\b", 7), (r"\breferred\s+to\b", 5),
        (r"\bkindly\s+(?:see|evaluate|review)\b", 3), (r"\breferring\s+(?:doctor|physician)\b", 4),
        (r"\breferral\b", 3),
    ],
    "Operative Report": [
        (r"\boperative\s+report\b", 8), (r"\boperation\s+performed\b", 6),
        (r"\bpre[- ]operative\b|\bpost[- ]operative\b", 5),
        (r"\bprocedure\s+performed\b", 4), (r"\banesthesia\b", 3),
        (r"\bsurgeon\b", 3),
    ],
    "Vaccination Record": [
        (r"\bvaccination\s+record\b|\bvaccine\s+record\b", 8),
        (r"\bimmunization\b|\bimmunisation\b", 6), (r"\bvaccine\b", 3),
        (r"\bdose\s+(?:1|2|3|booster)\b", 3), (r"\bbooster\b", 3),
    ],
}

@dataclass
class Classification:
    document_class: str
    confidence: float
    needs_review: bool
    method: str
    evidence: List[dict]
    scores: Dict[str, float]


def _normalise(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def classify_document(text: str, filename: str = "", requested_type: str = "Other") -> Classification:
    source = _normalise(f"{filename} {text}")
    if not source.strip():
        return Classification("Other", 0.0, True, "no_text", [], {})

    scores = {name: 0.0 for name in PATTERNS}
    evidence_by_class: Dict[str, List[dict]] = {name: [] for name in PATTERNS}
    for cls, patterns in PATTERNS.items():
        for pattern, weight in patterns:
            match = re.search(pattern, source, re.I)
            if match:
                scores[cls] += weight
                evidence_by_class[cls].append({"pattern": pattern, "matched": match.group(0)[:80], "weight": weight})

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_class, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Requested UI type is only a weak prior; OCR evidence wins when strong.
    if requested_type in PATTERNS and best_score < 4 and not evidence_by_class[requested_type]:
        best_class = requested_type
        best_score = 1.0

    total = sum(scores.values())
    margin = (best_score - second_score) / max(best_score, 1.0)
    evidence_count = len(evidence_by_class.get(best_class, []))
    # Calibrated for a prototype: strong evidence + separation -> higher confidence.
    confidence = min(0.98, 0.30 + 0.09 * min(best_score, 7) + 0.25 * max(margin, 0) + 0.03 * min(evidence_count, 4))
    if best_score < 3:
        confidence = min(confidence, 0.48)
    if total and best_score / total < 0.45:
        confidence = min(confidence, 0.60)

    needs_review = confidence < 0.72 or best_score < 4 or margin < 0.20
    if needs_review and confidence < 0.55:
        best_class = "Other"

    method = "AI-3C explainable local classifier"
    return Classification(
        document_class=best_class,
        confidence=round(confidence, 3),
        needs_review=needs_review,
        method=method,
        evidence=evidence_by_class.get(best_class, []),
        scores={k: round(v, 2) for k, v in ranked if v > 0},
    )
