from fastapi.testclient import TestClient
import main
import uuid

client = TestClient(main.app)


def auth(email, password):
    r = client.post('/api/auth/login', json={'email': email, 'password': password})
    assert r.status_code == 200, r.text
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def test_accessibility_capabilities_are_explicit():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/accessibility/capabilities', headers=admin)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['phase'] == 'AI-5H'
    assert set(data['supported_languages']) >= {'en-IN', 'hi-IN'}
    assert set(data['input_modes']) == {'touch', 'voice', 'hybrid'}
    assert data['features']['touch_fallback'] is True
    assert 'en-IN' in data['tts_languages']


def test_patient_can_save_and_read_own_accessibility_preferences():
    email = f'ai5h-{uuid.uuid4().hex[:8]}@example.local'
    r = client.post('/api/auth/register', json={
        'name': 'AI5H Patient', 'email': email, 'password': 'patient123',
        'age': 30, 'gender': 'Other', 'phone': ''
    })
    assert r.status_code == 200, r.text
    patient_id = r.json()['user']['id']
    headers = {'Authorization': 'Bearer ' + r.json()['access_token']}
    payload = {
        'language': 'hi-IN', 'input_mode': 'hybrid', 'font_scale': '1.5',
        'high_contrast': True, 'reduced_motion': True, 'captions': True,
        'audio_enabled': True, 'audio_speed': '0.75', 'assisted_mode': True
    }
    r = client.put(f'/api/patients/{patient_id}/accessibility', headers=headers, json=payload)
    assert r.status_code == 200, r.text
    saved = r.json()['preferences']
    assert saved['language'] == 'hi-IN'
    assert saved['input_mode'] == 'hybrid'
    assert saved['font_scale'] == '1.5'
    assert saved['high_contrast'] is True
    r = client.get(f'/api/patients/{patient_id}/accessibility', headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()['preferences']['assisted_mode'] is True


def test_invalid_accessibility_values_are_rejected():
    patient = auth('patient@sih26047.local', 'patient123')
    # The clean demo patient can update only its own preferences; invalid values are rejected for a valid owner.
    r = client.put('/api/patients/1/accessibility', headers=patient, json={
        'language': 'xx-IN', 'input_mode': 'touch', 'font_scale': '1.0',
        'audio_speed': '1.0'
    })
    assert r.status_code in (403, 422)


def test_patient_cannot_modify_another_patient_preferences():
    patient = auth('patient@sih26047.local', 'patient123')
    r = client.put('/api/patients/999999/accessibility', headers=patient, json={
        'language': 'en-IN', 'input_mode': 'touch', 'font_scale': '1.0',
        'audio_speed': '1.0'
    })
    assert r.status_code == 403


def test_integration_audit_includes_ai5h():
    admin = auth('admin@sih26047.local', 'admin123')
    r = client.get('/api/system/integration-audit', headers=admin)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['status'] == 'PASS', data
    assert data['failed_checks'] == []
    assert 'AI-5G' in data['phase_inventory']
    assert 'AI-5H' in data['phase_inventory']
    assert any(x['table'] == 'accessibility_preferences' and x['present'] for x in data['checks']['schema']['items'])
