from ai3d_medical_extractor import extract_structured_medical_data

def test_lab_extraction():
    text = "LAB REPORT\nHemoglobin: 12.4 g/dL\nGlucose: 108 mg/dL\nCreatinine: 0.9 mg/dL\nDate: 31/08/2026"
    out = extract_structured_medical_data(text, "Lab Report")
    assert out["item_count"] >= 4
    labels = {x["label"] for x in out["items"]}
    assert "Hemoglobin" in labels
    assert any(x["category"] == "date" for x in out["items"])

def test_prescription_extraction():
    text = "PRESCRIPTION\nParacetamol 500 mg\nTake one tablet twice daily after food\nAmoxicillin 500 mg\n01/09/2026"
    out = extract_structured_medical_data(text, "Prescription")
    meds = [x for x in out["items"] if x["category"] == "medication"]
    assert len(meds) >= 2
    assert any("twice daily" in x["value"].lower() for x in out["items"] if x["category"] == "instruction")

def test_no_inference_from_empty_text():
    out = extract_structured_medical_data("", "Lab Report")
    assert out["items"] == []
    assert out["needs_review"] is True

def test_evidence_attached():
    out = extract_structured_medical_data("BP: 120/80 mmHg", "Consultation Note")
    item = next(x for x in out["items"] if x["label"] == "Blood pressure")
    assert item["value"] == "120/80"
    assert item["evidence"]
    assert 0 < item["confidence"] <= 1
