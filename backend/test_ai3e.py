from ai3e_document_verification import build_verification_summary, apply_document_verification


def sample():
    return {"schema_version":"AI-3D.1", "needs_review":True, "items":[
        {"category":"measurement","label":"Blood pressure","value":"120/80","confidence":0.95,"needs_review":False,"evidence":"BP: 120/80 mmHg"},
        {"category":"medication","label":"Medicine","value":"Paracetamol","confidence":0.78,"needs_review":True,"evidence":"Paracetamol 500 mg"},
    ]}


def test_summary():
    out=build_verification_summary(sample())
    assert out["total_items"]==2
    assert out["items_needing_review"]==1


def test_full_verification_and_correction():
    out=apply_document_verification(sample(), [
        {"index":0,"verified":True},
        {"index":1,"verified":True,"value":"Paracetamol 500 mg"},
    ])
    assert out["verification_status"]=="Verified"
    assert out["needs_review"] is False
    assert out["items"][1]["corrected"] is True


def test_partial_verification_remains_reviewable():
    out=apply_document_verification(sample(), [{"index":1,"verified":True}])
    assert out["verification_status"]=="Partially Verified"
    assert out["needs_review"] is True


def test_rejects_unconfirmed_item():
    try:
        apply_document_verification(sample(), [{"index":0,"verified":False}])
        assert False
    except ValueError:
        assert True
