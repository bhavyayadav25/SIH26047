import json
from types import SimpleNamespace
from ai4b_clinical_risk import build_risk_assessment


def patient(profile=None):
    return SimpleNamespace(id=1, name="Test", profile=profile or SimpleNamespace(allergies="", conditions="", medications=""))


def consultation(text, risk="none", flags=None):
    return SimpleNamespace(
        id=7, created_at=None, title="Current visit",
        structured_data=json.dumps({"chief_complaint": text}),
        nlp_data=json.dumps({"positive_symptoms": [text]}),
        risk_level=risk, red_flags=json.dumps(flags or []),
    )


def document(verified=True):
    return SimpleNamespace(
        id=9, filename="report.pdf", verification_status="Verified" if verified else "Pending",
        verified_data=json.dumps({"items":[{"label":"Finding","value":"chest pain","evidence":"Patient reports chest pain","verified":verified}]}),
        structured_extraction="{}",
    )


def test_emergency_concerning_chest_pattern():
    c = consultation("chest pain and shortness of breath")
    out = build_risk_assessment(patient(), [c], [])
    assert out["risk_level"] == "emergency"
    assert out["interrupt_routine_flow"] is True
    assert out["requires_human_review"] is True


def test_verified_document_can_trigger_rule():
    out = build_risk_assessment(patient(), [], [document(True)])
    assert out["risk_level"] == "none"
    # A single generic chest-pain finding is intentionally not an emergency by itself.
    assert out["document_review"]["verified_items"] == 1


def test_unverified_document_cannot_trigger_rule():
    out = build_risk_assessment(patient(), [], [document(False)])
    assert out["risk_level"] == "none"
    assert len(out["document_review"]["pending_items"]) == 1
    assert out["source_scope"]["unverified_document_items_used_for_alerting"] is False


def test_no_match_does_not_claim_safety():
    out = build_risk_assessment(patient(), [consultation("mild sore throat")], [])
    assert out["risk_level"] == "none"
    assert "does not establish clinical safety" in out["message"]
    assert out["limitations"]
