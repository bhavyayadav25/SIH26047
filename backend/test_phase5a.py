from pathlib import Path
from phase5a_integration_audit import build_integration_audit


def test_integration_audit_passes_on_current_backend():
    import main
    result = build_integration_audit(main.app, main.engine, Path(main.BASE_DIR))
    assert result["status"] == "PASS"
    assert result["failed_checks"] == []


def test_required_ai_modules_are_present():
    import main
    result = build_integration_audit(main.app, main.engine, Path(main.BASE_DIR))
    assert result["checks"]["modules"]["ok"] is True
    assert result["checks"]["modules"]["missing"] == []


def test_required_schema_and_routes_are_present():
    import main
    result = build_integration_audit(main.app, main.engine, Path(main.BASE_DIR))
    assert result["checks"]["schema"]["ok"] is True
    assert result["checks"]["routes"]["ok"] is True
    assert result["checks"]["database"]["ok"] is True


def test_audit_is_non_clinical_and_non_mutating_by_contract():
    import main
    result = build_integration_audit(main.app, main.engine, Path(main.BASE_DIR))
    boundary = result["clinical_safety_boundary"]
    assert boundary["reads_clinical_data"] is False
    assert boundary["mutates_clinical_data"] is False
    assert boundary["diagnosis"] is False
    assert boundary["prescribing"] is False
    assert boundary["autonomous_decision"] is False
