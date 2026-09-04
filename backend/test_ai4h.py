import json
from types import SimpleNamespace
from ai4h_final_clinical_gate import build_final_clinical_gate


def test_routine_ready_without_blockers():
    p = SimpleNamespace(id=1)
    c = SimpleNamespace(id=10)
    out = build_final_clinical_gate(p, c, summary={"x": 1}, risk={"risk_level": "none", "document_review": {"pending_items": []}})
    assert out["ready_for_routine_consultation"] is True
    assert out["disposition"] == "clinician_review"
    assert out["safety"]["diagnosis"] is False


def test_emergency_blocks_routine_flow():
    p = SimpleNamespace(id=1); c = SimpleNamespace(id=10)
    out = build_final_clinical_gate(p, c, risk={"risk_level": "emergency", "document_review": {"pending_items": []}})
    assert out["ready_for_routine_consultation"] is False
    assert out["disposition"] == "immediate_triage"
    assert out["safety"]["emergency_interrupts_routine_flow"] is True


def test_pending_document_verification_blocks_routine_readiness():
    p = SimpleNamespace(id=1); c = SimpleNamespace(id=10)
    out = build_final_clinical_gate(p, c, risk={"risk_level": "none", "document_review": {"pending_items": [{"label": "Hb"}]}})
    assert out["ready_for_routine_consultation"] is False
    assert out["disposition"] == "verification_required"
    assert any(x["code"] == "DOCUMENT_VERIFICATION_PENDING" for x in out["blockers"])


def test_advisory_flags_are_warnings_not_autonomous_decisions():
    p = SimpleNamespace(id=1); c = SimpleNamespace(id=10)
    out = build_final_clinical_gate(
        p, c,
        risk={"risk_level": "none", "document_review": {"pending_items": []}},
        decision_support={"review_prompts": [{"x": 1}]},
        medication={"flags": [{"x": 1}]},
        investigations={"flags": [{"x": 1}]},
        copilot={"review_alerts": [{"x": 1}]},
    )
    assert out["ready_for_routine_consultation"] is True
    assert len(out["warnings"]) == 4
    assert out["safety"]["autonomous_decision"] is False


def test_no_false_safety_claim():
    p = SimpleNamespace(id=1); c = SimpleNamespace(id=10)
    out = build_final_clinical_gate(p, c, risk={"risk_level": "none", "document_review": {"pending_items": []}})
    assert any("does not establish" in x.lower() for x in out["limitations"])
