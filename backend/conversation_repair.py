"""Phase AI-1E: conversation repair and voice robustness.

Conservative, explainable handling for common kiosk conversation failures:
- silence / empty speech
- unclear or very short recognition
- repeat requests
- misunderstanding
- correction / rejection of AI interpretation
- explicit language switching
- voice-to-touch fallback after repeated failures

This module does not diagnose or change clinical meaning by itself.
"""
from __future__ import annotations
import re
from typing import Any, Dict

LANGUAGE_ALIASES = {
    "english": "en-IN", "eng": "en-IN", "en": "en-IN",
    "hindi": "hi-IN", "हिंदी": "hi-IN", "हिन्दी": "hi-IN", "hi": "hi-IN",
    "bengali": "bn-IN", "বাংলা": "bn-IN", "bn": "bn-IN",
    "tamil": "ta-IN", "தமிழ்": "ta-IN", "ta": "ta-IN",
    "telugu": "te-IN", "తెలుగు": "te-IN", "te": "te-IN",
    "marathi": "mr-IN", "मराठी": "mr-IN", "mr": "mr-IN",
    "gujarati": "gu-IN", "ગુજરાતી": "gu-IN", "gu": "gu-IN",
    "kannada": "kn-IN", "ಕನ್ನಡ": "kn-IN", "kn": "kn-IN",
}

REPEAT_PATTERNS = [
    r"\brepeat\b", r"say that again", r"again please", r"once again",
    r"can you repeat", r"what did you say", r"pardon", r"sorry\?",
    r"फिर से", r"दोबारा", r"दोबारा बताओ", r"फिर बताइए", r"फिर से बोलिए",
    r"samajh nahi", r"samajh nahin", r"phir se", r"dobara",
]
UNDERSTAND_PATTERNS = [
    r"i don't understand", r"i do not understand", r"don't understand", r"not understand",
    r"i didn't understand", r"did not understand", r"what do you mean",
    r"समझ नहीं", r"समझ नहीं आया", r"समझ नहीं आ रहा", r"समझ में नहीं",
    r"samajh nahi aaya", r"samajh nahin aaya", r"samajh nahi aa raha",
]
WRONG_PATTERNS = [
    r"that's wrong", r"that is wrong", r"wrong answer", r"not correct", r"incorrect",
    r"no that's not", r"that isn't what i said", r"not what i said",
    r"गलत", r"ये गलत है", r"यह गलत है", r"मैंने ऐसा नहीं कहा", r"वह नहीं",
    r"galat", r"maine aisa nahi kaha", r"woh nahi",
]
LANGUAGE_PATTERNS = [
    ("hi-IN", [r"speak in hindi", r"hindi mein", r"hindi me", r"हिंदी में", r"हिन्दी में", r"hindi bolo", r"hindi boliye"]),
    ("en-IN", [r"speak in english", r"english mein", r"english me", r"अंग्रेज़ी में", r"angrezi mein"]),
    ("bn-IN", [r"bengali mein", r"bangla mein", r"বাংলায়", r"বাংলাতে"]),
    ("ta-IN", [r"tamil mein", r"தமிழில்"]),
    ("te-IN", [r"telugu mein", r"తెలుగులో"]),
    ("mr-IN", [r"marathi mein", r"मराठीत"]),
    ("gu-IN", [r"gujarati mein", r"ગુજરાતીમાં"]),
    ("kn-IN", [r"kannada mein", r"ಕನ್ನಡದಲ್ಲಿ"]),
]

RESPONSES = {
    "en-IN": {
        "repeat": "Of course. I’ll repeat the question.",
        "understand": "No problem. I’ll ask that in a simpler way.",
        "wrong": "Okay. I’ll correct that. Please tell me the answer again in your own words.",
        "silence": "I didn’t hear an answer. You can speak again or tap an answer on the screen.",
        "unclear": "I couldn’t understand that clearly. Please say it again, or tap an answer on the screen.",
        "fallback": "Voice is having trouble. You can continue by tapping the answer on the screen, or ask a staff member for help.",
        "language": "Sure. I’ll continue in {language}.",
    },
    "hi-IN": {
        "repeat": "बिल्कुल। मैं सवाल दोबारा पूछता हूँ।",
        "understand": "कोई बात नहीं। मैं इसे आसान तरीके से पूछता हूँ।",
        "wrong": "ठीक है। मैं इसे सुधारता हूँ। कृपया अपने शब्दों में फिर से बताएं।",
        "silence": "मुझे आपका जवाब सुनाई नहीं दिया। आप फिर से बोल सकते हैं या स्क्रीन पर जवाब चुन सकते हैं।",
        "unclear": "मैं आपका जवाब साफ़ तौर पर समझ नहीं पाया। कृपया फिर से बोलें या स्क्रीन पर जवाब चुनें।",
        "fallback": "आवाज़ में थोड़ी दिक्कत हो रही है। आप स्क्रीन पर जवाब चुनकर आगे बढ़ सकते हैं या स्टाफ की मदद ले सकते हैं।",
        "language": "ज़रूर। अब मैं {language} में बात करूंगा।",
    },
}

LANGUAGE_LABELS = {
    "en-IN": "English", "hi-IN": "हिन्दी", "bn-IN": "বাংলা", "ta-IN": "தமிழ்",
    "te-IN": "తెలుగు", "mr-IN": "मराठी", "gu-IN": "ગુજરાતી", "kn-IN": "ಕನ್ನಡ",
}


def _matches(text: str, patterns: list[str]) -> bool:
    value = (text or "").strip().lower()
    return any(re.search(pattern, value, re.I) for pattern in patterns)


def detect_requested_language(text: str) -> str | None:
    value = (text or "").strip().lower()
    for language, patterns in LANGUAGE_PATTERNS:
        if any(re.search(pattern, value, re.I) for pattern in patterns):
            return language
    # Also allow an explicit single-word language response.
    compact = re.sub(r"[^\w\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0C80-\u0CFF\s-]", " ", value).strip()
    return LANGUAGE_ALIASES.get(compact)


def analyze_repair(text: str | None, *, event: str = "answer", attempt: int = 0, current_language: str = "en-IN") -> Dict[str, Any]:
    raw = (text or "").strip()
    event = (event or "answer").lower().strip()
    requested_language = detect_requested_language(raw)

    if requested_language and requested_language != current_language:
        return {"action": "switch_language", "reason": "explicit_language_request", "requested_language": requested_language,
                "response_key": "language", "confidence": "high", "fallback_to_touch": False}

    if event in {"silence", "timeout", "no_speech"} or not raw:
        action = "voice_retry" if attempt < 2 else "touch_fallback"
        return {"action": action, "reason": "no_speech", "response_key": "silence" if attempt < 2 else "fallback",
                "confidence": "high", "fallback_to_touch": attempt >= 2}

    if _matches(raw, REPEAT_PATTERNS):
        return {"action": "repeat_question", "reason": "repeat_request", "response_key": "repeat",
                "confidence": "high", "fallback_to_touch": False}
    if _matches(raw, UNDERSTAND_PATTERNS):
        return {"action": "simplify_question", "reason": "patient_did_not_understand", "response_key": "understand",
                "confidence": "high", "fallback_to_touch": False}
    if _matches(raw, WRONG_PATTERNS):
        return {"action": "request_correction", "reason": "patient_rejected_interpretation", "response_key": "wrong",
                "confidence": "high", "fallback_to_touch": False}

    # Very short non-answer utterances are often recognition noise or a hesitant
    # response. Do not discard them; ask for clarification.
    words = raw.split()
    if len(words) <= 1 and raw.lower() not in {"yes", "no", "haan", "han", "nahin", "nahi", "हां", "हाँ", "नहीं", "nope", "okay", "ok"}:
        action = "voice_retry" if attempt < 2 else "touch_fallback"
        return {"action": action, "reason": "very_short_unclear_input", "response_key": "unclear" if attempt < 2 else "fallback",
                "confidence": "medium", "fallback_to_touch": attempt >= 2}

    return {"action": "accept_answer", "reason": "no_repair_needed", "response_key": None,
            "confidence": "high", "fallback_to_touch": False}


def localized_response(repair: Dict[str, Any], language: str) -> str:
    lang = language if language in RESPONSES else "en-IN"
    key = repair.get("response_key")
    if key == "language":
        requested = repair.get("requested_language", language)
        label = LANGUAGE_LABELS.get(requested, requested)
        return RESPONSES[lang]["language"].format(language=label)
    return RESPONSES[lang].get(key or "unclear", RESPONSES[lang]["unclear"])
