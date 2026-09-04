from ai3c_document_classifier import classify_document


def test_prescription():
    r = classify_document("PRESCRIPTION Rx\nTab Amoxicillin 500 mg\nDose: one tablet twice daily")
    assert r.document_class == "Prescription"
    assert r.confidence >= 0.72
    assert not r.needs_review


def test_lab_report():
    r = classify_document("LABORATORY REPORT\nHemoglobin 12.4 g/dL\nCreatinine 0.9 mg/dL\nReference range")
    assert r.document_class == "Lab Report"
    assert r.confidence >= 0.72


def test_imaging_report():
    r = classify_document("MRI BRAIN\nFindings: no acute abnormality.\nImpression: normal study.")
    assert r.document_class == "Imaging Report"


def test_discharge_summary():
    r = classify_document("DISCHARGE SUMMARY\nAdmission Date: 01/08/2026\nHospital Course\nDischarge medications")
    assert r.document_class == "Discharge Summary"


def test_ambiguous_text_is_conservative():
    r = classify_document("Medical document. Patient: Rahul. Please review.")
    assert r.needs_review
    assert r.document_class == "Other"


def test_direct_api_style_payload():
    r = classify_document("Referral letter. Kindly evaluate this patient for cardiology review.")
    assert r.document_class == "Referral Letter"
    assert r.evidence
