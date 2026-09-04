import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi.testclient import TestClient
import main


def auth(client, email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_consultation_owned_by_doctor_and_completes():
    client = TestClient(main.app)
    patient = auth(client, 'patient@sih26047.local', 'patient123')
    doctor = auth(client, 'doctor@sih26047.local', 'doctor123')
    r = client.post('/api/encounters', headers=patient, json={'patient_id':1,'department':'AI5D-Test','priority':'normal','reason':'Consultation test'})
    assert r.status_code == 200, r.text
    eid = r.json()['encounter']['id']
    assert client.post(f'/api/encounters/{eid}/status', headers=doctor, json={'status':'called'}).status_code == 200
    r = client.post(f'/api/doctor/encounters/{eid}/consultation', headers=doctor, json={'encounter_id':eid,'title':'Test consultation'})
    assert r.status_code == 200, r.text
    cid = r.json()['consultation']['consultation_id']
    r = client.put(f'/api/doctor/consultations/{cid}/record', headers=doctor, json={'sections':{'history':'Cough for 3 days','examination':'Doctor-observed finding'},'doctor_notes':'Review completed.'})
    assert r.status_code == 200, r.text
    r = client.post(f'/api/doctor/consultations/{cid}/complete', headers=doctor, json={'sections':{'history':'Cough for 3 days','examination':'Doctor-observed finding','assessment':'Doctor-entered assessment','diagnosis':'Doctor-entered diagnosis','plan':'Doctor-entered plan','prescription':'Doctor-entered prescription','follow_up':'Doctor-entered follow-up'}})
    assert r.status_code == 200, r.text
    assert r.json()['consultation']['status'] == 'completed'
    assert r.json()['clinical_decision_source'] == 'doctor_entered'


def test_patient_cannot_edit_consultation():
    client = TestClient(main.app)
    patient = auth(client, 'patient@sih26047.local', 'patient123')
    r = client.put('/api/doctor/consultations/1/record', headers=patient, json={'sections':{'diagnosis':'bad'}})
    assert r.status_code == 403


def test_completed_consultation_is_locked():
    client = TestClient(main.app)
    doctor = auth(client, 'doctor@sih26047.local', 'doctor123')
    # Use an existing completed record if available; otherwise create one.
    db = main.SessionLocal()
    try:
        c = db.query(main.Consultation).filter(main.Consultation.consultation_status == 'completed').first()
        cid = c.id if c else None
    finally:
        db.close()
    if cid is None:
        patient = auth(client, 'patient@sih26047.local', 'patient123')
        r = client.post('/api/encounters', headers=patient, json={'patient_id':1,'department':'AI5D-Lock','priority':'normal','reason':'Lock test'})
        eid = r.json()['encounter']['id']
        client.post(f'/api/encounters/{eid}/status', headers=doctor, json={'status':'called'})
        r = client.post(f'/api/doctor/encounters/{eid}/consultation', headers=doctor, json={'encounter_id':eid})
        cid = r.json()['consultation']['consultation_id']
        client.post(f'/api/doctor/consultations/{cid}/complete', headers=doctor, json={'sections':{'diagnosis':'Doctor entered'}})
    r = client.put(f'/api/doctor/consultations/{cid}/record', headers=doctor, json={'sections':{'diagnosis':'changed'}})
    assert r.status_code == 409
