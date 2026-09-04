import json
from datetime import datetime
from types import SimpleNamespace
from ai4e_investigation_intelligence import build_investigation_intelligence


def patient():
    return SimpleNamespace(id=1)


def doc(doc_id, date, label, value, unit="mg/dL", verified=True, doc_type="Lab Report"):
    items = [{"category":"lab_result","label":label,"value":value,"unit":unit,"verified":verified,"evidence":f"{label}: {value} {unit}"}]
    return SimpleNamespace(id=doc_id, filename=f"lab{doc_id}.pdf", classification=doc_type,
                           document_type=doc_type, created_at=datetime.fromisoformat(date),
                           verified_data=json.dumps({"items":items}))


def test_only_verified_results_are_used():
    out = build_investigation_intelligence(patient(), [doc(1,"2026-01-01T10:00:00","Hb","11.2",verified=False)])
    assert out["investigations"] == []
    assert out["safety"]["uses_verified_document_items_only"] is True


def test_longitudinal_numeric_trend_is_descriptive():
    out = build_investigation_intelligence(patient(), [
        doc(1,"2026-01-01T10:00:00","Hemoglobin","11.2","g/dL"),
        doc(2,"2026-06-01T10:00:00","Hemoglobin","10.4","g/dL"),
    ])
    assert len(out["trends"]) == 1
    assert out["trends"][0]["direction"] == "decreased"
    assert out["trends"][0]["requires_clinician_judgment"] is True


def test_different_units_are_not_auto_converted():
    out = build_investigation_intelligence(patient(), [
        doc(1,"2026-01-01T10:00:00","Glucose","100","mg/dL"),
        doc(2,"2026-06-01T10:00:00","Glucose","5.6","mmol/L"),
    ])
    assert out["trends"] == []
    assert any(x["type"] == "unit_inconsistency" for x in out["review_alerts"])
    assert out["safety"]["automatic_unit_conversion"] is False


def test_no_abnormality_claim_without_reference_range():
    out = build_investigation_intelligence(patient(), [doc(1,"2026-01-01T10:00:00","Creatinine","1.4","mg/dL")])
    assert out["safety"]["normal_abnormal_classification_performed"] is False
    assert not any("abnormal" in str(x).lower() for x in out["review_alerts"])


def test_source_date_is_not_called_test_date():
    out = build_investigation_intelligence(patient(), [doc(1,"2026-01-01T10:00:00","TSH","2.1","mIU/L")])
    assert out["investigations"][0]["date_basis"] == "document_created_at"


def test_empty_history_is_explicit():
    out = build_investigation_intelligence(patient(), [])
    assert out["investigations"] == []
    assert out["review_alerts"][0]["type"] == "information_gap"
