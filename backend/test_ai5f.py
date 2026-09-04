from fastapi.testclient import TestClient
import main
import uuid

client = TestClient(main.app)


def auth(email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_admin_is_role_protected():
    patient = auth('patient@sih26047.local', 'patient123')
    r = client.get('/api/admin/departments', headers=patient)
    assert r.status_code == 403


def test_admin_can_manage_department():
    admin = auth('admin@sih26047.local', 'admin123')
    name = 'AI5F-Cardiology-' + uuid.uuid4().hex[:8]
    r = client.post('/api/admin/departments', headers=admin, json={'name': name, 'specialty': 'Cardiology', 'active': True})
    assert r.status_code == 200, r.text
    dep_id = r.json()['department']['id']
    r = client.put(f'/api/admin/departments/{dep_id}', headers=admin, json={'name': name, 'specialty': 'Cardiology & Heart', 'active': True})
    assert r.status_code == 200, r.text
    r = client.get('/api/admin/departments', headers=admin)
    assert any(x['id'] == dep_id and x['active'] for x in r.json()['departments'])


def test_admin_can_create_doctor_and_manage_availability():
    admin = auth('admin@sih26047.local', 'admin123')
    email = 'ai5f.doctor@example.local'
    r = client.post('/api/admin/doctors', headers=admin, json={
        'name': 'AI5F Doctor', 'email': email, 'password': 'doctor456',
        'specialty': 'Cardiology', 'department': 'AI5F-Cardiology', 'registration_number': 'AI5F-001'
    })
    assert r.status_code in (200, 409), r.text
    if r.status_code == 200:
        doctor_id = r.json()['doctor']['id']
    else:
        doctors = client.get('/api/admin/doctors', headers=admin).json()['doctors']
        doctor_id = next(x['id'] for x in doctors if x['email'] == email)
    r = client.post('/api/admin/availability', headers=admin, json={'doctor_id': doctor_id, 'day_of_week': 'Mon', 'start_time': '09:00', 'end_time': '13:00', 'active': True})
    assert r.status_code in (200, 409), r.text
    r = client.get('/api/admin/availability', headers=admin, params={'doctor_id': doctor_id})
    assert r.status_code == 200
    assert any(x['doctor_id'] == doctor_id for x in r.json()['availability'])


def test_invalid_opd_time_is_rejected():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.post('/api/admin/opd-config', headers=admin, json={'department': 'AI5F-Cardiology', 'working_days': ['Mon'], 'start_time': '18:00', 'end_time': '09:00', 'active': True})
    assert r.status_code == 422


def test_admin_can_save_opd_and_hospital_config():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.post('/api/admin/opd-config', headers=admin, json={'department': 'AI5F-Cardiology', 'working_days': ['Mon','Wed','Fri'], 'start_time': '09:00', 'end_time': '17:00', 'active': True})
    assert r.status_code == 200, r.text
    r = client.put('/api/admin/hospital', headers=admin, json={'hospital_name': 'MediKiosk Demo Hospital', 'facility_code': 'AI5F-FACILITY', 'timezone': 'Asia/Kolkata', 'default_department': 'General Medicine', 'active': True})
    assert r.status_code == 200, r.text
    r = client.get('/api/admin/hospital', headers=admin)
    assert r.status_code == 200
    assert r.json()['hospital']['facility_code'] == 'AI5F-FACILITY'


def test_admin_can_create_routing_rule():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.post('/api/admin/routing', headers=admin, json={'department': 'AI5F-Cardiology', 'specialty': 'Cardiology', 'doctor_id': None, 'priority': 10, 'active': True})
    assert r.status_code == 200, r.text
    rid = r.json()['routing_rule_id']
    r = client.get('/api/admin/routing', headers=admin)
    assert r.status_code == 200
    assert any(x['id'] == rid for x in r.json()['routing_rules'])


def test_admin_audit_endpoint_sees_ai5f_schema_and_routes():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/system/integration-audit', headers=admin)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['status'] == 'PASS', data
    assert data['failed_checks'] == []
    assert 'AI-5F' in data['phase_inventory']
    assert 'departments' in data['checks']['schema']['items'][0]['table'] or any(i['table'] == 'departments' and i['present'] for i in data['checks']['schema']['items'])
