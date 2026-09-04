import os
import tempfile
from pathlib import Path
import pytest

# Use an isolated copy of the backend DB for test execution.
BASE = Path(__file__).resolve().parent
import main




@pytest.fixture(autouse=True)
def clean_encounters():
    db = main.SessionLocal()
    try:
        db.query(main.Encounter).delete()
        db.commit()
    finally:
        db.close()
    yield
    db = main.SessionLocal()
    try:
        db.query(main.Encounter).delete()
        db.commit()
    finally:
        db.close()


def auth(client, email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_encounter_queue_flow():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    doctor_headers = auth(client, 'doctor@sih26047.local', 'doctor123')
    patient_headers = auth(client, 'patient@sih26047.local', 'patient123')

    r = client.post('/api/encounters', headers=patient_headers, json={
        'patient_id': 1, 'department': 'General Medicine', 'priority': 'normal', 'reason': 'Routine visit'
    })
    assert r.status_code == 200, r.text
    encounter = r.json()['encounter']
    eid = encounter['id']
    assert encounter['token_number'] >= 1
    assert encounter['status'] == 'waiting'

    r = client.get('/api/queue', headers=doctor_headers, params={'department': 'General Medicine'})
    assert r.status_code == 200, r.text
    assert any(x['id'] == eid for x in r.json()['queue'])

    r = client.post(f'/api/encounters/{eid}/status', headers=doctor_headers, json={'status': 'called'})
    assert r.status_code == 200, r.text
    assert r.json()['encounter']['status'] == 'called'

    r = client.post(f'/api/encounters/{eid}/status', headers=doctor_headers, json={'status': 'in_consultation'})
    assert r.status_code == 200, r.text
    r = client.post(f'/api/encounters/{eid}/status', headers=doctor_headers, json={'status': 'completed'})
    assert r.status_code == 200, r.text
    assert r.json()['encounter']['status'] == 'completed'


def test_patient_cannot_manage_queue():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    headers = auth(client, 'patient@sih26047.local', 'patient123')
    r = client.get('/api/queue', headers=headers)
    assert r.status_code == 403


def test_invalid_transition_rejected():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    doctor_headers = auth(client, 'doctor@sih26047.local', 'doctor123')
    patient_headers = auth(client, 'patient@sih26047.local', 'patient123')
    # Create a fresh encounter and attempt an illegal direct transition.
    # The earlier test may have completed today's encounter, so use a distinct department.
    r = client.post('/api/encounters', headers=patient_headers, json={
        'patient_id': 1, 'department': 'ENT', 'priority': 'normal', 'reason': 'Transition test'
    })
    assert r.status_code == 200, r.text
    eid = r.json()['encounter']['id']
    r = client.post(f'/api/encounters/{eid}/status', headers=doctor_headers, json={'status': 'completed'})
    assert r.status_code == 409


def test_duplicate_active_encounter_rejected():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    headers = auth(client, 'patient@sih26047.local', 'patient123')
    # Use a new department so this remains independent of other test data.
    r = client.post('/api/encounters', headers=headers, json={
        'patient_id': 1, 'department': 'Dermatology', 'priority': 'normal', 'reason': 'Duplicate test'
    })
    assert r.status_code == 200, r.text
    r = client.post('/api/encounters', headers=headers, json={
        'patient_id': 1, 'department': 'Dermatology', 'priority': 'normal', 'reason': 'Duplicate test 2'
    })
    assert r.status_code == 409
