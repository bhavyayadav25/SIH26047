"""Phase 5A: cumulative backend integration audit.

This module is intentionally diagnostic rather than clinical. It inspects the
running FastAPI application, SQLite schema, required AI modules, filesystem
paths, and critical API contracts. It does not mutate clinical data or the
application database.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sqlalchemy import inspect, text

SCHEMA_VERSION = "AI-5H.1"

REQUIRED_MODULES = {
    "AI-1F": "ai1f_orchestrator.py",
    "AI-2": "ai2_safety.py",
    "AI-3C": "ai3c_document_classifier.py",
    "AI-3D": "ai3d_medical_extractor.py",
    "AI-3E": "ai3e_document_verification.py",
    "AI-3F": "ai3f_clinical_timeline.py",
    "AI-3G": "ai3g_explainability.py",
    "AI-3H": "ai3h_clinical_handoff.py",
    "AI-4A": "ai4a_clinical_summary.py",
    "AI-4B": "ai4b_clinical_risk.py",
    "AI-4C": "ai4c_clinical_decision_support.py",
    "AI-4D": "ai4d_medication_intelligence.py",
    "AI-4E": "ai4e_investigation_intelligence.py",
    "AI-4F": "ai4f_clinical_question_assistant.py",
    "AI-4G": "ai4g_consultation_copilot.py",
    "AI-4H": "ai4h_final_clinical_gate.py",
    "AI-5B": "ai5b_encounter_queue.py",
    "AI-5C": "ai5c_doctor_workspace.py",
    "AI-5D": "ai5d_consultation.py",
    "AI-5E": "ai5e_triage.py",
    "AI-5G": "ai5g_analytics.py",
}

REQUIRED_TABLES = {
    "users": {"id", "name", "email", "password_hash", "role"},
    "patient_profiles": {"id", "user_id"},
    "medical_documents": {
        "id", "patient_id", "filename", "document_type", "extracted_text",
        "classification", "structured_extraction", "verification_status",
        "verified_data", "verified_by", "verified_at",
    },
    "consultations": {
        "id", "patient_id", "summary", "risk_level", "red_flags",
        "doctor_review", "structured_data", "nlp_data", "ai_summary",
    },
    "interview_states": {"id", "patient_id", "session_id", "status", "structured_data"},
    "voice_turns": {"id", "patient_id", "session_id", "transcript"},
    "consent_records": {"id", "patient_id", "granted", "scope"},
    "audit_events": {"id", "user_id", "action", "resource", "created_at"},
    "user_sessions": {"id", "user_id", "token_hash", "expires_at", "revoked_at"},
    "encounters": {"id", "patient_id", "doctor_id", "department", "visit_date", "token_number", "priority", "status", "created_at"},
    "departments": {"id", "name", "specialty", "active"},
    "doctor_profiles": {"id", "user_id", "specialty", "department", "active"},
    "opd_configurations": {"id", "department", "working_days", "start_time", "end_time", "active"},
    "doctor_availability": {"id", "doctor_id", "day_of_week", "start_time", "end_time", "active"},
    "routing_rules": {"id", "department", "specialty", "doctor_id", "priority", "active"},
    "hospital_configuration": {"id", "hospital_name", "facility_code", "timezone", "default_department", "active"},
    "accessibility_preferences": {"id", "patient_id", "language", "input_mode", "font_scale", "high_contrast", "reduced_motion", "captions", "audio_enabled", "audio_speed", "assisted_mode"},
}

REQUIRED_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("GET", "/api/health"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/register"),
    ("POST", "/api/documents/upload"),
    ("POST", "/api/documents/classify"),
    ("POST", "/api/documents/extract"),
    ("GET", "/api/patients/{patient_id}/timeline"),
    ("GET", "/api/patients/{patient_id}/clinical-summary"),
    ("GET", "/api/patients/{patient_id}/risk-assessment"),
    ("GET", "/api/patients/{patient_id}/decision-support"),
    ("GET", "/api/patients/{patient_id}/medication-intelligence"),
    ("GET", "/api/patients/{patient_id}/investigation-intelligence"),
    ("POST", "/api/patients/{patient_id}/clinical-question"),
    ("GET", "/api/doctor/consultations/{consultation_id}/copilot"),
    ("GET", "/api/doctor/consultations/{consultation_id}/clinical-gate"),
    ("POST", "/api/encounters"),
    ("GET", "/api/encounters/{encounter_id}"),
    ("GET", "/api/queue"),
    ("POST", "/api/encounters/{encounter_id}/status"),
    ("GET", "/api/admin/departments"),
    ("POST", "/api/admin/departments"),
    ("GET", "/api/admin/doctors"),
    ("POST", "/api/admin/doctors"),
    ("GET", "/api/admin/opd-config"),
    ("GET", "/api/admin/availability"),
    ("GET", "/api/admin/routing"),
    ("GET", "/api/admin/hospital"),
    ("GET", "/api/admin/analytics"),
    ("GET", "/api/accessibility/capabilities"),
    ("GET", "/api/patients/{patient_id}/accessibility"),
    ("PUT", "/api/patients/{patient_id}/accessibility"),
    ("GET", "/api/voice/status"),
    ("POST", "/api/voice/transcribe"),
    ("POST", "/api/voice/speak"),
)


def _route_pairs(app: Any) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        for method in methods:
            pairs.add((method.upper(), route.path))
    return pairs


def _module_checks(base_dir: Path) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for phase, filename in REQUIRED_MODULES.items():
        path = base_dir / filename
        present = path.is_file()
        importable = False
        if present:
            try:
                spec = importlib.util.spec_from_file_location(f"_audit_{phase.replace('-', '_')}", path)
                importable = bool(spec and spec.loader)
            except Exception:
                importable = False
        if not present or not importable:
            missing.append(phase)
        items.append({"phase": phase, "file": filename, "present": present, "importable": importable})
    return {"ok": not missing, "missing": missing, "items": items}


def _schema_checks(engine: Any) -> Dict[str, Any]:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    items: List[Dict[str, Any]] = []
    missing_tables: List[str] = []
    missing_columns: Dict[str, List[str]] = {}
    for table, required_columns in REQUIRED_TABLES.items():
        present = table in tables
        actual = {c["name"] for c in inspector.get_columns(table)} if present else set()
        absent = sorted(required_columns - actual)
        if not present:
            missing_tables.append(table)
        if absent:
            missing_columns[table] = absent
        items.append({"table": table, "present": present, "missing_columns": absent})
    return {
        "ok": not missing_tables and not missing_columns,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "items": items,
    }


def _database_probe(engine: Any) -> Dict[str, Any]:
    try:
        with engine.connect() as conn:
            value = conn.execute(text("SELECT 1")).scalar_one()
        return {"ok": value == 1, "query": "SELECT 1"}
    except Exception as exc:  # diagnostic endpoint should report, not crash
        return {"ok": False, "error": type(exc).__name__}


def _filesystem_checks(base_dir: Path) -> Dict[str, Any]:
    upload_dir = base_dir / "uploads"
    voice_tmp = upload_dir / "voice_tmp"
    data_documents = base_dir / "data" / "documents"
    checks = [
        {"path": "uploads", "exists": upload_dir.is_dir()},
        {"path": "uploads/voice_tmp", "exists": voice_tmp.is_dir()},
        {"path": "data/documents", "exists": data_documents.is_dir()},
        {"path": "sih26047.db", "exists": (base_dir / "sih26047.db").is_file()},
    ]
    return {"ok": all(x["exists"] for x in checks), "items": checks}


def _route_checks(app: Any) -> Dict[str, Any]:
    actual = _route_pairs(app)
    missing = [
        {"method": method, "path": path}
        for method, path in REQUIRED_ROUTES
        if (method, path) not in actual
    ]
    return {
        "ok": not missing,
        "missing": missing,
        "required_count": len(REQUIRED_ROUTES),
        "registered_api_routes": len({p for p in actual if p[1].startswith("/api/")}),
    }


def _config_checks(app: Any) -> Dict[str, Any]:
    origins = set()
    for middleware in app.user_middleware:
        options = getattr(middleware, "kwargs", {}) or {}
        origins.update(options.get("allow_origins") or [])
    expected = {"http://localhost:5173", "http://127.0.0.1:5173"}
    return {
        "ok": expected.issubset(origins),
        "cors_dev_origins_present": sorted(expected.intersection(origins)),
        "note": "Development CORS origins are checked; production deployment should use an explicit trusted origin list.",
    }


def build_integration_audit(app: Any, engine: Any, base_dir: Path) -> Dict[str, Any]:
    modules = _module_checks(base_dir)
    schema = _schema_checks(engine)
    database = _database_probe(engine)
    filesystem = _filesystem_checks(base_dir)
    routes = _route_checks(app)
    config = _config_checks(app)

    checks = {
        "modules": modules,
        "database": database,
        "schema": schema,
        "filesystem": filesystem,
        "routes": routes,
        "configuration": config,
    }
    failed = [name for name, result in checks.items() if not result.get("ok", False)]
    status = "PASS" if not failed else "FAIL"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "failed_checks": failed,
        "checks": checks,
        "phase_inventory": ["AI-1F", "AI-2", "AI-3A", "AI-3B", "AI-3C", "AI-3D", "AI-3E", "AI-3F", "AI-3G", "AI-3H", "AI-4A", "AI-4B", "AI-4C", "AI-4D", "AI-4E", "AI-4F", "AI-4G", "AI-4H", "AI-5A", "AI-5B", "AI-5C", "AI-5D", "AI-5E", "AI-5F", "AI-5G", "AI-5H"],
        "clinical_safety_boundary": {
            "purpose": "integration diagnostics only",
            "reads_clinical_data": False,
            "mutates_clinical_data": False,
            "diagnosis": False,
            "prescribing": False,
            "autonomous_decision": False,
        },
        "production_note": "A passing audit means the prototype's checked integration contracts are present; it is not a production security, medical validation, or regulatory certification.",
    }
