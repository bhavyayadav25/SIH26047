import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi.testclient import TestClient
import main


def auth(client, email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_triage_dashboard_is_role_protected_and_operational():
    client = TestClient(main.app)
    patient = auth(client, 'patient@sih26047.local', 'patient123')
    doctor = auth(client, 'doctor@sih26047.local', 'doctor123')
    r = client.get('/api/triage/dashboard', headers=patient)
    assert r.status_code == 403
    r = client.post('/api/encounters', headers=patient, json={'patient_id': 1, 'department': 'AI5E-Test', 'priority': 'urgent', 'reason': 'Triage test'})
    assert r.status_code == 200, r.text
    eid = r.json()['encounter']['id']
    r = client.get('/api/triage/dashboard', headers=doctor, params={'department': 'AI5E-Test'})
    assert r.status_code == 200, r.text
    body = r.json()['triage_dashboard']
    assert body['count'] >= 1
    assert body['queue'][0]['encounter_id'] == eid
    assert body['queue'][0]['priority'] == 'urgent'
    assert body['safety']['ai_does_not_assign_priority'] is True
    assert client.post(f'/api/encounters/{eid}/status', headers=doctor, json={'status':'cancelled'}).status_code == 200


def test_triage_actions_are_human_and_audited():
    client = TestClient(main.app)
    patient = auth(client, 'patient@sih26047.local', 'patient123')
    triage = auth(client, 'doctor@sih26047.local', 'doctor123')
    r = client.post('/api/encounters', headers=patient, json={'patient_id': 1, 'department': 'AI5E-Action', 'priority': 'normal', 'reason': 'Action test'})
    assert r.status_code == 200, r.text
    eid = r.json()['encounter']['id']
    r = client.post(f'/api/triage/encounters/{eid}/action', headers=triage, json={'action': 'acknowledge', 'notes': 'Reviewed by triage'})
    assert r.status_code == 200, r.text
    assert r.json()['triage_status'] == 'acknowledged'
    r = client.post(f'/api/triage/encounters/{eid}/action', headers=triage, json={'action': 'escalate', 'priority': 'emergency', 'notes': 'Escalated for immediate human review'})
    assert r.status_code == 200, r.text
    assert r.json()['triage_status'] == 'escalated'
    assert r.json()['priority'] == 'emergency'
    assert client.post(f'/api/encounters/{eid}/status', headers=triage, json={'status':'cancelled'}).status_code == 200


def test_triage_cannot_act_on_closed_encounter():
    client = TestClient(main.app)
    patient = auth(client, 'patient@sih26047.local', 'patient123')
    triage = auth(client, 'doctor@sih26047.local', 'doctor123')
    r = client.post('/api/encounters', headers=patient, json={'patient_id': 1, 'department': 'AI5E-Closed', 'priority': 'normal', 'reason': 'Closed test'})
    assert r.status_code == 200, r.text
    eid = r.json()['encounter']['id']
    # Mark cancelled via queue transition.
    r = client.post(f'/api/encounters/{eid}/status', headers=triage, json={'status': 'cancelled'})
    assert r.status_code == 200, r.text
    r = client.post(f'/api/triage/encounters/{eid}/action', headers=triage, json={'action': 'acknowledge'})
    assert r.status_code == 409
