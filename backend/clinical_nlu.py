"""Phase AI-1B: lightweight clinical language understanding.

This module converts patient language into conservative, structured evidence.
It does not diagnose, recommend treatment, or invent missing information.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List

SYMPTOMS = {
    "chest pain": ["chest pain", "chest discomfort", "chest pressure", "chest tightness", "seene mein dard", "seene me dard", "seene mein pressure", "seene me pressure", "seene mein tightness", "सीने में दर्द", "सीने में तकलीफ", "छाती में दर्द", "छाती में तकलीफ़"],
    "headache": ["headache", "head pain", "migraine", "sir dard", "sar dard", "सिरदर्द", "सिर में दर्द"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "chakkar", "चक्कर", "चक्कर आना"],
    "nausea": ["nausea", "nauseous", "vomiting", "vomit", "ji michlana", "मतली", "उल्टी"],
    "abdominal pain": ["stomach pain", "abdominal pain", "belly pain", "pet me dard", "pet mein dard", "पेट में दर्द", "पेट दर्द"],
    "bloating": ["bloating", "bloated", "gas", "pet phoolna", "पेट फूलना", "गैस"],
    "fever": ["fever", "feverish", "high temperature", "bukhar", "बुखार", "ताप"],
    "cough": ["cough", "coughing", "khansi", "खांसी", "खाँसी"],
    "shortness of breath": ["shortness of breath", "breathless", "difficulty breathing", "breathing difficulty", "saans phoolna", "saans phoolti", "saans lene me dikkat", "सांस फूलना", "सांस लेने में दिक्कत", "श्वास लेने में दिक्कत"],
    "wheeze": ["wheeze", "wheezing", "सीटी जैसी सांस"],
    "sore throat": ["sore throat", "throat pain", "gale mein dard", "गले में दर्द"],
    "fatigue": ["fatigue", "tired", "tiredness", "low energy", "weakness", "kamzori", "थकान", "कमजोरी"],
    "skin rash/itching": ["rash", "itching", "itchy", "skin irritation", "khujli", "दाने", "खुजली", "चकत्ते"],
    "urinary symptoms": ["burning urination", "burning while urinating", "frequent urination", "urine burning", "peshab mein jalan", "पेशाब में जलन"],
    "back/joint pain": ["back pain", "joint pain", "knee pain", "shoulder pain", "muscle ache", "kamar dard", "ghutne mein dard", "कमर दर्द", "घुटने में दर्द"],
    "diarrhea": ["diarrhea", "loose motions", "loose stools", "dast", "दस्त", "पतले दस्त"],
    "appetite change": ["loss of appetite", "poor appetite", "increased appetite", "bhook kam", "भूख कम"],
}

NEGATIONS = [
    "no ", "not ", "without ", "never ", "don't ", "do not ", "didn't ", "did not ",
    "nahi ", "nahin ", "nahi hai", "nahin hai", "नहीं", "नही", "नहीं है", "नहीं हैं", "कोई नहीं",
    "nahi hota", "nahin hota", "nahi hoti", "nahin hoti"
]

DURATION_PATTERNS = [
    r"\b(?:for|since)\s+((?:\d+|one|two|three|four|five|six|seven|a|an))\s*(day|days|week|weeks|month|months|year|years)\b",
    r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*(?:ago|se pehle)\b",
    r"\b(since yesterday|since last night|since morning|for a long time)\b",
    r"(\d+)\s*(din|dino|dinon)\s*(se|se hi)?",
    r"(\d+)\s*(hafte|hafton|mahine|mahino|saal)\s*(se|se hi)?",
    r"(\d+)\s*(दिन|हफ्ते|हफ्तों|महीने|महीनों|साल)\s*(से|से ही)?",
]

NUMBER_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"a":1,"an":1}
SEVERITY = {
    "mild": ["mild", "slight", "little", "halka", "हल्का", "कम"],
    "moderate": ["moderate", "medium", "madhyam", "मध्यम"],
    "severe": ["severe", "very painful", "worst", "extreme", "unbearable", "bahut zyada", "बहुत तेज", "बहुत ज्यादा", "असहनीय"],
}

INTENT_MAP = {
    "chest pain": "chest",
    "headache": "headache",
    "abdominal pain": "abdominal",
    "fever": "fever",
    "cough": "respiratory",
    "shortness of breath": "respiratory",
    "skin rash/itching": "skin",
    "urinary symptoms": "urinary",
    "back/joint pain": "pain",
    "diarrhea": "abdominal",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def negated(text: str, start: int) -> bool:
    # Negation is intentionally local. A negation in one clause should not
    # leak across a conjunction into a later symptom.
    window = text[max(0, start - 45):start]
    for boundary in (" but ", " and ", " however ", "."):
        if boundary in window:
            window = window.rsplit(boundary, 1)[-1]
    # Hindi/Hinglish often puts the negation immediately before the concept.
    return any(n in window for n in NEGATIONS)


def extract_duration(text: str) -> List[str]:
    values: List[str] = []
    for pattern in DURATION_PATTERNS:
        values.extend(m.group(0) for m in re.finditer(pattern, text, flags=re.I))
    return list(dict.fromkeys(values))


def extract_severity(text: str) -> str | None:
    for level, terms in SEVERITY.items():
        if any(term in text for term in terms):
            return level
    nums = re.findall(r"\b(?:10|[0-9])\b", text)
    if nums:
        n = int(nums[0])
        if n <= 3:
            return "mild"
        if n <= 6:
            return "moderate"
        return "severe"
    return None


def extract_symptoms(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for canonical, variants in SYMPTOMS.items():
        for term in sorted(variants, key=len, reverse=True):
            pos = text.find(term.lower())
            if pos >= 0:
                is_negated = negated(text, pos)
                out.append({
                    "concept": canonical,
                    "mention": term,
                    "negated": is_negated,
                    "evidence": text[max(0, pos-35):min(len(text), pos+len(term)+35)].strip(),
                })
                break

    # Compositional Hinglish/Hindi patterns: patients frequently insert a
    # duration or other phrase between the body location and symptom quality.
    extra_patterns = [
        ("chest pain", r"\bseene?\s+(?:mein|me)\s+.{0,35}\b(?:pressure|tightness|dard)\b", "seene mein ... symptom"),
        ("shortness of breath", r"\bsaans\s+(?:bhi\s+)?(?:phoolti|phoolna|chadh(?:ti|na))\b", "saans phoolna"),
        ("headache", r"\b(?:sir|sar)\s+(?:mein|me)\s+.{0,25}\b(?:dard|pain)\b", "sir mein dard"),
        ("abdominal pain", r"\bpet\s+(?:mein|me)\s+.{0,25}\b(?:dard|pain)\b", "pet mein dard"),
    ]
    existing = {x["concept"] for x in out}
    for canonical, pattern, mention in extra_patterns:
        if canonical in existing:
            continue
        m = re.search(pattern, text, flags=re.I)
        if m:
            pos = m.start()
            out.append({
                "concept": canonical,
                "mention": mention,
                "negated": negated(text, pos),
                "evidence": m.group(0),
            })
    return out


def infer_intent(symptoms: List[Dict[str, Any]]) -> str:
    positive = [x["concept"] for x in symptoms if not x["negated"]]
    for symptom in positive:
        if symptom in INTENT_MAP:
            return INTENT_MAP[symptom]
    return "general"


def extract_clinical_entities(text: str, language: str = "en-IN") -> Dict[str, Any]:
    raw = text or ""
    normalized = normalize(raw)
    symptoms = extract_symptoms(normalized)
    durations = extract_duration(normalized)
    severity = extract_severity(normalized)
    positive = [x["concept"] for x in symptoms if not x["negated"]]
    negated_terms = [x["concept"] for x in symptoms if x["negated"]]
    return {
        "raw_text": raw.strip(),
        "normalized_text": normalized,
        "language": language,
        "symptoms": symptoms,
        "positive_symptoms": positive,
        "negated_symptoms": negated_terms,
        "duration_mentions": durations,
        "severity": severity,
        "intent": infer_intent(symptoms),
        "evidence": [x["evidence"] for x in symptoms],
        "engine": "AI-1B conservative clinical entity extraction",
        "diagnostic": False,
    }
