"""Phase AI-2: conservative, explainable clinical safety/triage support.

This module is NOT a diagnostic engine. It detects a limited set of explicit
red-flag patterns in patient-reported text and returns a human-review action.
Rules must be clinically reviewed before real-world deployment.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

PRIORITY = {"none": 0, "watch": 1, "urgent": 2, "emergency": 3}

RULES = [
    ("unresponsive_or_unconscious", "emergency", "Unresponsiveness or loss of consciousness", [
        "unconscious", "passed out", "not responding", "unresponsive", "lost consciousness",
        "बेहोश", "होश नहीं", "hosh nahi", "behosh"
    ]),
    ("severe_breathing_difficulty", "emergency", "Severe breathing difficulty", [
        "cannot breathe", "can't breathe", "unable to breathe", "struggling to breathe",
        "severe shortness of breath", "not able to breathe", "सांस नहीं आ रही",
        "सांस लेने में बहुत दिक्कत", "बहुत ज्यादा सांस फूलना", "saans nahi aa rahi",
        "saans lene me bahut dikkat"
    ]),
    ("stroke_like_symptoms", "emergency", "Sudden neurological warning symptoms", [
        "face drooping", "facial droop", "weakness on one side", "one sided weakness",
        "one-sided weakness", "difficulty speaking", "cannot speak", "trouble speaking",
        "sudden numbness on one side", "sudden weakness", "एक तरफ कमजोरी", "बोलने में दिक्कत",
        "एक तरफ सुन्न", "ek taraf kamzori", "bolne me dikkat"
    ]),
    ("severe_chest_pattern", "emergency", "Chest symptoms with concerning associated features", [
        "chest pain", "chest pressure", "chest tightness", "chest discomfort",
        "seene mein dard", "seene me dard", "seene mein pressure", "सीने में दर्द",
        "सीने में दबाव"
    ]),
    ("gi_bleeding", "urgent", "Reported gastrointestinal bleeding", [
        "vomiting blood", "blood in vomit", "black stool", "black stools", "blood in stool",
        "bloody stool", "खून की उल्टी", "उल्टी में खून", "काला मल", "मल में खून",
        "khoon ki ulti", "kaala mal"
    ]),
    ("severe_abdominal_pain", "urgent", "Severe abdominal symptoms", [
        "severe abdominal pain", "severe stomach pain", "rigid abdomen", "very severe stomach pain",
        "बहुत तेज पेट दर्द", "पेट में बहुत तेज दर्द", "bahut tez pet dard"
    ]),
    ("sudden_severe_headache", "urgent", "Sudden or exceptionally severe headache", [
        "worst headache", "worst headache of my life", "thunderclap headache", "sudden severe headache",
        "अचानक बहुत तेज सिरदर्द", "जिंदगी का सबसे तेज सिरदर्द", "achanak bahut tez sir dard"
    ]),
    ("severe_allergic_reaction_pattern", "emergency", "Breathing/swelling symptoms suggesting a severe allergic reaction", [
        "throat swelling", "tongue swelling", "swollen tongue", "difficulty breathing after eating",
        "difficulty breathing after medicine", "गला सूज", "जीभ सूज", "gala sooj", "jeebh sooj"
    ]),
    ("active_heavy_bleeding", "emergency", "Reported heavy or uncontrolled bleeding", [
        "bleeding won't stop", "bleeding will not stop", "heavy bleeding", "uncontrolled bleeding",
        "खून नहीं रुक रहा", "बहुत ज्यादा खून", "khoon nahi ruk raha"
    ]),
]

CHEST_TERMS = ["chest pain", "chest pressure", "chest tightness", "chest discomfort", "seene mein dard", "seene me dard", "seene mein pressure", "सीने में दर्द", "सीने में दबाव"]
BREATH_TERMS = ["shortness of breath", "breathless", "difficulty breathing", "breathlessness", "saans phoolna", "सांस फूलना"]
RADIATION_TERMS = ["left arm", "right arm", "both arms", "jaw", "neck", "back", "shoulder", "बाएं हाथ", "बांह", "जबड़े", "गर्दन", "पीठ", "कंधे"]
SWEAT_TERMS = ["sweating", "cold sweat", "pasina", "पसीना"]
FAINT_TERMS = ["fainting", "fainted", "passed out", "बेहोशी", "behoshi"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 70):start]
    return bool(re.search(r"\b(no|not|without|denies|deny|never|don't|doesn't|didn't)\b|नहीं|नही|कोई नहीं|नहीं है|नहीं हो|नाही|নেই|না|இல்லை|లేదు|નથી|ಇಲ್ಲ", prefix))


def _hit(text: str, term: str) -> bool:
    for m in re.finditer(re.escape(term), text):
        if not _negated(text, m.start()):
            return True
    return False


def _any(text: str, terms: List[str]) -> bool:
    return any(_hit(text, t.lower()) for t in terms)


def _alert(rule_id: str, level: str, label: str, evidence: List[str], action: str, rationale: str) -> Dict[str, Any]:
    return {
        "id": rule_id,
        "level": level,
        "label": label,
        "evidence": evidence[:5],
        "rationale": rationale,
        "recommended_action": action,
        "requires_human_review": True,
        "diagnosis": None,
    }


def evaluate_safety(answer: str, structured: Dict[str, Any] | None = None) -> Dict[str, Any]:
    structured = structured or {}
    combined = _normalize(" ".join([str(answer or "")] + [str(v) for v in structured.values() if v not in (None, "", [], {})]))
    alerts: Dict[str, Dict[str, Any]] = {}

    for rule_id, level, label, terms in RULES:
        hits = [t for t in terms if _hit(combined, t.lower())]
        if not hits:
            continue
        # Chest pain alone is not automatically escalated to emergency. It is
        # escalated when combined with specific concerning features below.
        if rule_id == "severe_chest_pattern":
            continue
        action = "immediate_human_triage" if level == "emergency" else "prompt_clinical_review"
        alerts[rule_id] = _alert(rule_id, level, label, hits, action,
            "A transparent prototype safety rule matched patient-reported information; this is a triage-support alert, not a diagnosis.")

    has_chest = _any(combined, CHEST_TERMS)
    has_breath = _any(combined, BREATH_TERMS)
    has_radiation = _any(combined, RADIATION_TERMS)
    has_sweat = _any(combined, SWEAT_TERMS)
    has_faint = _any(combined, FAINT_TERMS)
    if has_chest and (has_breath or has_radiation or has_sweat or has_faint):
        evidence = [t for t, ok in [("chest symptoms", has_chest), ("breathing difficulty", has_breath), ("spread to another area", has_radiation), ("sweating", has_sweat), ("fainting", has_faint)] if ok]
        alerts["chest_concerning_combination"] = _alert(
            "chest_concerning_combination", "emergency", "Chest symptoms with a concerning associated feature",
            evidence, "immediate_human_triage",
            "Chest symptoms plus a concerning associated feature should interrupt routine intake and receive prompt human assessment."
        )

    # Escalate multiple distinct urgent alerts to a human triage review, but do
    # not invent an emergency diagnosis.
    urgent_ids = [a for a in alerts.values() if a["level"] == "urgent"]
    if len(urgent_ids) >= 2 and "chest_concerning_combination" not in alerts:
        alerts["multiple_urgent_signals"] = _alert(
            "multiple_urgent_signals", "urgent", "Multiple urgent warning signals",
            [a["label"] for a in urgent_ids], "prompt_clinical_review",
            "More than one independent prototype warning pattern was detected."
        )

    ordered = sorted(alerts.values(), key=lambda x: PRIORITY[x["level"]], reverse=True)
    level = ordered[0]["level"] if ordered else "none"
    action = "immediate_human_triage" if level == "emergency" else ("prompt_clinical_review" if level == "urgent" else "continue_routine_intake")
    return {
        "version": "AI-2.1",
        "risk_level": level,
        "alerts": ordered,
        "recommended_action": action,
        "requires_human_review": bool(ordered),
        "interrupt_interview": level == "emergency",
        "message": (
            "Potentially urgent information was detected. Routine intake should pause for human clinical review."
            if level == "emergency" else
            "Potential warning information was detected. Clinical staff should review it promptly."
            if level == "urgent" else
            "No prototype red-flag pattern was identified in the information provided so far."
        ),
        "disclaimer": "Prototype triage-support only; not a diagnosis, treatment recommendation, or substitute for clinical judgment. Rules require clinical validation before deployment."
    }
