"""Phase AI-1F: end-to-end adaptive interview orchestrator.

Connects AI-1A..1E into one conservative backend flow. It does not diagnose,
prescribe, or assign a doctor. It coordinates state, clinical extraction,
adaptive question selection, repair handling, and safety interruption.
"""
from __future__ import annotations
from typing import Any, Dict

REPAIR_ACTIONS = {
    "repeat_question", "simplify_question", "request_correction",
    "voice_retry", "touch_fallback", "switch_language"
}


def choose_action(*, repair: Dict[str, Any] | None, next_question: Dict[str, Any] | None,
                  risk_level: str = "none") -> str:
    """Return a small, frontend-safe action vocabulary."""
    if risk_level in {"urgent", "emergency"}:
        return "triage_interrupt"
    action = (repair or {}).get("action")
    if action in REPAIR_ACTIONS:
        return action
    if next_question:
        return "ask_question"
    return "complete_interview"


def build_turn_response(*, state: Dict[str, Any], repair: Dict[str, Any] | None = None,
                        next_question: Dict[str, Any] | None = None,
                        risk_level: str = "none", red_flags: list | None = None,
                        localized_repair_message: str | None = None) -> Dict[str, Any]:
    """Normalize all AI stages into one response contract."""
    action = choose_action(repair=repair, next_question=next_question, risk_level=risk_level)
    return {
        "orchestrated": True,
        "version": "AI-1F.1",
        "action": action,
        "repair": repair,
        "repair_message": localized_repair_message,
        "next_question": next_question,
        "risk_level": risk_level or "none",
        "red_flags": red_flags or [],
        "state": state,
        "disclaimer": "AI assists with clinical history intake and workflow. It is not a diagnosis or treatment system."
    }
