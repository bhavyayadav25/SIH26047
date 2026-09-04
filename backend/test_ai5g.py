from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def auth(email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_analytics_is_admin_only():
    patient = auth('patient@sih26047.local', 'patient123')
    r = client.get('/api/admin/analytics', headers=patient)
    assert r.status_code == 403


def test_admin_analytics_returns_aggregate_shape():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/admin/analytics', headers=admin)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['schema_version'] == 'AI-5G.1'
    assert 'overview' in data
    assert 'queue' in data
    assert 'performance' in data
    assert 'daily_encounter_volume' in data
    assert data['safety_boundary']['diagnosis'] is False
    assert data['data_quality']['ai_correction_rate'] is None


def test_analytics_date_validation():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/admin/analytics', headers=admin, params={'start_date': '2026-10-01', 'end_date': '2026-09-01'})
    assert r.status_code == 422


def test_analytics_rejects_bad_date_format():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/admin/analytics', headers=admin, params={'start_date': '09/01/2026'})
    assert r.status_code == 422


def test_integration_audit_includes_ai5g():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/system/integration-audit', headers=admin)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['status'] == 'PASS', data
    assert data['failed_checks'] == []
    assert 'AI-5G' in data['phase_inventory']
    assert any(x['phase'] == 'AI-5G' and x['present'] for x in data['checks']['modules']['items'])
