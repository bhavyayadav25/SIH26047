from __future__ import annotations
from datetime import datetime, date
from pathlib import Path
import hashlib
import re
import json
import os
import uuid
import secrets
import tempfile
import subprocess
from typing import Optional, List, Dict, Any
from fastapi import Request

from clinical_nlu import extract_clinical_entities
from conversation_repair import analyze_repair, localized_response
from ai1f_orchestrator import build_turn_response
from ai2_safety import evaluate_safety, PRIORITY as SAFETY_PRIORITY
from ai3a.integration import register_ai3a
from ai3c_document_classifier import classify_document
from ai3d_medical_extractor import extract_structured_medical_data
from ai3e_document_verification import build_verification_summary, apply_document_verification
from ai3f_clinical_timeline import build_timeline
from ai3g_explainability import explain_document, explain_timeline
from ai3h_clinical_handoff import build_clinical_handoff
from ai4a_clinical_summary import build_clinical_summary
from ai4b_clinical_risk import build_risk_assessment
from ai4c_clinical_decision_support import build_decision_support
from ai4d_medication_intelligence import build_medication_intelligence
from ai4e_investigation_intelligence import build_investigation_intelligence
from ai4f_clinical_question_assistant import answer_clinical_question
from ai4g_consultation_copilot import build_consultation_copilot
from ai4h_final_clinical_gate import build_final_clinical_gate
from phase5a_integration_audit import build_integration_audit
from ai5b_encounter_queue import (normalize_department, normalize_priority, normalize_status, can_transition, queue_sort_key, serialize_encounter, ROLE_CAN_MANAGE_QUEUE)
from ai5c_doctor_workspace import build_doctor_workspace
from ai5d_consultation import build_consultation_payload, normalize_sections, consultation_summary
from ai5e_triage import (TRIAGE_ROLES, normalize_triage_action, triage_action_status, serialize_triage_item, build_triage_dashboard)
from ai5g_analytics import build_analytics_dashboard

# Phase AI-1D: optional local speech stack. The server remains usable when these
# packages/models are not installed; voice endpoints return a clear setup error.
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

# Phase 5C: lightweight local NLP/ML layer (no external model/API required)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, inspect, text as sql_text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'sih26047.db'}"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

VOICE_TMP_DIR = UPLOAD_DIR / "voice_tmp"
VOICE_TMP_DIR.mkdir(exist_ok=True)
WHISPER_MODEL_SIZE = os.getenv("MEDIKIOSK_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("MEDIKIOSK_WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("MEDIKIOSK_WHISPER_COMPUTE_TYPE", "int8")
TTS_VOICE_MAP = {
    "en-IN": os.getenv("MEDIKIOSK_TTS_EN_IN", "en-IN-NeerjaNeural"),
    "hi-IN": os.getenv("MEDIKIOSK_TTS_HI_IN", "hi-IN-SwaraNeural"),
}
_whisper_model = None

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile = relationship("PatientProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    consultations = relationship("Consultation", back_populates="patient", cascade="all, delete-orphan")


class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    blood_group = Column(String, nullable=True)
    allergies = Column(Text, nullable=True)
    conditions = Column(Text, nullable=True)
    medications = Column(Text, nullable=True)
    address = Column(Text, nullable=True)
    user = relationship("User", back_populates="profile")


class MedicalDocument(Base):
    __tablename__ = "medical_documents"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    document_type = Column(String, nullable=False, default="Other")
    mime_type = Column(String, nullable=True)
    stored_path = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    extracted_data = Column(Text, nullable=True)
    classification = Column(String, nullable=True)
    classification_confidence = Column(String, nullable=True)
    classification_method = Column(String, nullable=True)
    classification_evidence = Column(Text, nullable=True)
    classification_needs_review = Column(Integer, default=0)
    structured_extraction = Column(Text, nullable=True)
    extraction_needs_review = Column(Integer, default=1)
    extraction_method = Column(String, nullable=True)
    verification_status = Column(String, default="Pending")
    verified_data = Column(Text, nullable=True)
    verification_notes = Column(Text, nullable=True)
    verified_by = Column(Integer, nullable=True)
    verified_at = Column(DateTime, nullable=True)
    status = Column(String, default="Processed")
    created_at = Column(DateTime, default=datetime.utcnow)


class AyushAssessment(Base):
    __tablename__ = "ayush_assessments"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    system = Column(String, default="Ayurveda")
    assessment_type = Column(String, default="Dashavidha Pariksha")
    responses = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    status = Column(String, default="Completed")
    created_at = Column(DateTime, default=datetime.utcnow)


class ConsentRecord(Base):
    __tablename__ = "consent_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    consent_type = Column(String, nullable=False, default="Clinical case-taking")
    version = Column(String, nullable=False, default="5B.1")
    language = Column(String, nullable=False, default="en-IN")
    granted = Column(Integer, default=0)
    audio_explained = Column(Integer, default=0)
    scope = Column(Text, nullable=False, default="AI-assisted case taking, document processing, AYUSH intake, physician review")
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)


class FHIRExportRecord(Base):
    __tablename__ = "fhir_export_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exported_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resource_type = Column(String, nullable=False, default="Bundle")
    bundle_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    role = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class InterviewState(Base):
    """Persistent state for one AI clinical-intake encounter.

    AI-1A makes the conversation state server-owned instead of trusting the
    browser to resend the complete history on every answer. This is an
    encounter/session memory layer, not a diagnostic model.
    """
    __tablename__ = "interview_states"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    language = Column(String, nullable=False, default="en-IN")
    pathway = Column(String, nullable=True)
    current_question_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")
    structured_data = Column(Text, nullable=False, default="{}")
    answered_question_ids = Column(Text, nullable=False, default="[]")
    conversation = Column(Text, nullable=False, default="[]")
    risk_level = Column(String, nullable=False, default="none")
    red_flags = Column(Text, nullable=False, default="[]")
    version = Column(String, nullable=False, default="AI-2.1")
    repair_count = Column(Integer, nullable=False, default=0)
    voice_failure_count = Column(Integer, nullable=False, default=0)
    last_input_mode = Column(String, nullable=True, default="text")
    last_repair_action = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class VoiceTurn(Base):
    """Metadata for one voice interaction. Raw patient audio is not persisted by default."""
    __tablename__ = "voice_turns"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    language = Column(String, nullable=False, default="en-IN")
    direction = Column(String, nullable=False, default="input")
    transcript = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="completed")
    provider = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AccessibilityPreference(Base):
    """Per-patient kiosk accessibility preferences. No clinical data is stored here."""
    __tablename__ = "accessibility_preferences"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    language = Column(String, nullable=False, default="en-IN")
    input_mode = Column(String, nullable=False, default="touch")
    font_scale = Column(String, nullable=False, default="1.0")
    high_contrast = Column(Integer, nullable=False, default=0)
    reduced_motion = Column(Integer, nullable=False, default=0)
    captions = Column(Integer, nullable=False, default=1)
    audio_enabled = Column(Integer, nullable=False, default=1)
    audio_speed = Column(String, nullable=False, default="1.0")
    assisted_mode = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Encounter(Base):
    """One hospital visit/OPD episode. Queue state is operational, not clinical."""
    __tablename__ = "encounters"
    __table_args__ = (UniqueConstraint("visit_date", "department", "token_number", name="uq_encounter_daily_department_token"),)
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    department = Column(String, nullable=False, default="General Medicine", index=True)
    visit_date = Column(String, nullable=False, index=True)
    token_number = Column(Integer, nullable=False)
    priority = Column(String, nullable=False, default="normal", index=True)
    status = Column(String, nullable=False, default="waiting", index=True)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    called_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    triage_status = Column(String, nullable=False, default="unreviewed", index=True)
    triage_notes = Column(Text, nullable=True)
    triage_updated_by = Column(Integer, nullable=True)
    triage_updated_at = Column(DateTime, nullable=True)


class Consultation(Base):
    __tablename__ = "consultations"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    status = Column(String, default="Completed")
    risk_level = Column(String, default="none")
    red_flags = Column(Text, nullable=True)
    doctor_review = Column(String, default="Pending")
    doctor_notes = Column(Text, nullable=True)
    structured_data = Column(Text, nullable=True)
    nlp_data = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)
    ai_summary_generated_at = Column(DateTime, nullable=True)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=True, index=True)
    consultation_status = Column(String, default="draft", index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("User", back_populates="consultations")


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    specialty = Column(String, nullable=False, default="General Medicine")
    active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    specialty = Column(String, nullable=False, default="General Medicine")
    department = Column(String, nullable=False, default="General Medicine", index=True)
    registration_number = Column(String, nullable=True)
    active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class OPDConfiguration(Base):
    __tablename__ = "opd_configurations"
    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, nullable=False, unique=True, index=True)
    working_days = Column(String, nullable=False, default="Mon,Tue,Wed,Thu,Fri")
    start_time = Column(String, nullable=False, default="09:00")
    end_time = Column(String, nullable=False, default="17:00")
    active = Column(Integer, nullable=False, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow)


class DoctorAvailability(Base):
    __tablename__ = "doctor_availability"
    __table_args__ = (UniqueConstraint("doctor_id", "day_of_week", "start_time", "end_time", name="uq_doctor_availability_window"),)
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    day_of_week = Column(String, nullable=False)
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RoutingRule(Base):
    __tablename__ = "routing_rules"
    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, nullable=False, index=True)
    specialty = Column(String, nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    priority = Column(Integer, nullable=False, default=100)
    active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class HospitalConfiguration(Base):
    __tablename__ = "hospital_configuration"
    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String, nullable=False, default="SIH26047 Demo Hospital")
    facility_code = Column(String, nullable=False, default="SIH26047-DEMO-FACILITY")
    timezone = Column(String, nullable=False, default="Asia/Kolkata")
    default_department = Column(String, nullable=False, default="General Medicine")
    active = Column(Integer, nullable=False, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# Lightweight schema migration so Phase 4B works with the existing Phase 3 SQLite DB.
def ensure_phase4b_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(consultations)"))}
        if "risk_level" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN risk_level VARCHAR DEFAULT 'none'"))
        if "red_flags" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN red_flags TEXT"))
        if "doctor_review" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN doctor_review VARCHAR DEFAULT 'Pending'"))
        if "doctor_notes" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN doctor_notes TEXT"))
        if "structured_data" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN structured_data TEXT"))
        if "nlp_data" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN nlp_data TEXT"))
        if "ai_summary" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN ai_summary TEXT"))
        if "ai_summary_generated_at" not in columns:
            conn.execute(sql_text("ALTER TABLE consultations ADD COLUMN ai_summary_generated_at DATETIME"))

ensure_phase4b_columns()


def ensure_ai5d_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(consultations)"))}
        additions = {
            "encounter_id": "INTEGER",
            "consultation_status": "VARCHAR DEFAULT 'draft'",
            "updated_at": "DATETIME",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(sql_text(f"ALTER TABLE consultations ADD COLUMN {name} {ddl}"))

ensure_ai5d_columns()

def ensure_ai5e_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(encounters)"))}
        additions = {
            "triage_status": "VARCHAR DEFAULT 'unreviewed'",
            "triage_notes": "TEXT",
            "triage_updated_by": "INTEGER",
            "triage_updated_at": "DATETIME",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(sql_text(f"ALTER TABLE encounters ADD COLUMN {name} {ddl}"))

ensure_ai5e_columns()

def ensure_ai3c_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(medical_documents)"))}
        additions = {
            "classification": "VARCHAR",
            "classification_confidence": "VARCHAR",
            "classification_method": "VARCHAR",
            "classification_evidence": "TEXT",
            "classification_needs_review": "INTEGER DEFAULT 0",
            "structured_extraction": "TEXT",
            "extraction_needs_review": "INTEGER DEFAULT 0",
            "extraction_method": "VARCHAR",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(sql_text(f"ALTER TABLE medical_documents ADD COLUMN {name} {ddl}"))

ensure_ai3c_columns()

def ensure_ai3e_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(medical_documents)"))}
        additions = {
            "verification_status": "VARCHAR DEFAULT 'Pending'",
            "verified_data": "TEXT",
            "verification_notes": "TEXT",
            "verified_by": "INTEGER",
            "verified_at": "DATETIME",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(sql_text(f"ALTER TABLE medical_documents ADD COLUMN {name} {ddl}"))

ensure_ai3e_columns()

def ensure_phase1e_columns():
    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(sql_text("PRAGMA table_info(interview_states)"))}
        additions = {
            "repair_count": "INTEGER DEFAULT 0",
            "voice_failure_count": "INTEGER DEFAULT 0",
            "last_input_mode": "VARCHAR",
            "last_repair_action": "VARCHAR",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(sql_text(f"ALTER TABLE interview_states ADD COLUMN {name} {ddl}"))

ensure_phase1e_columns()

# Phase 5A/5B tables are created alongside the existing Phase 3-4 schema.
Base.metadata.create_all(bind=engine)

def ensure_phase5b_indexes():
    with engine.begin() as conn:
        # Existing databases created before AI-5B need the same token uniqueness guarantee.
        conn.execute(sql_text("CREATE UNIQUE INDEX IF NOT EXISTS uq_encounter_daily_department_token ON encounters (visit_date, department, token_number)"))

ensure_phase5b_indexes()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def make_profile(db, user, age=None, gender=None, phone=None):
    profile = PatientProfile(
        user_id=user.id, age=age, gender=gender, phone=phone,
        blood_group="", allergies="", conditions="", medications="", address=""
    )
    db.add(profile)
    return profile


def seed_demo_data():
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.email == "patient@sih26047.local").first()
        if not patient:
            patient = User(
                name="Rahul Sharma", email="patient@sih26047.local",
                password_hash=hash_password("patient123"), role="patient"
            )
            db.add(patient)
            db.flush()
            make_profile(db, patient, 42, "Male", "+91 90000 00000")

        doctor = db.query(User).filter(User.email == "doctor@sih26047.local").first()
        if not doctor:
            doctor = User(
                name="Dr. Ananya Verma", email="doctor@sih26047.local",
                password_hash=hash_password("doctor123"), role="doctor"
            )
            db.add(doctor)
            db.flush()
        admin = db.query(User).filter(User.email == "admin@sih26047.local").first()
        if not admin:
            admin = User(
                name="Hospital Administrator", email="admin@sih26047.local",
                password_hash=hash_password("admin123"), role="admin"
            )
            db.add(admin)
            db.flush()
        doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == doctor.id).first()
        if not doctor_profile:
            db.add(DoctorProfile(user_id=doctor.id, specialty="General Medicine", department="General Medicine", active=1))
        dept = db.query(Department).filter(Department.name == "General Medicine").first()
        if not dept:
            db.add(Department(name="General Medicine", specialty="General Medicine", active=1))
        opd = db.query(OPDConfiguration).filter(OPDConfiguration.department == "General Medicine").first()
        if not opd:
            db.add(OPDConfiguration(department="General Medicine", working_days="Mon,Tue,Wed,Thu,Fri", start_time="09:00", end_time="17:00", active=1))
        hospital = db.query(HospitalConfiguration).first()
        if not hospital:
            db.add(HospitalConfiguration())
        db.commit()
    finally:
        db.close()


seed_demo_data()

# Phase 5H / Final MVP security foundation. Demo sessions are short-lived bearer
# tokens stored only as hashes; the client never sends the raw token to the DB.
SESSION_MINUTES = int(os.getenv("SIH_SESSION_MINUTES", "120"))
PUBLIC_PATHS = {"/api/health", "/api/auth/login", "/api/auth/register", "/docs", "/openapi.json", "/redoc"}

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def create_session(db, user_id: int):
    token = secrets.token_urlsafe(32)
    rec = UserSession(user_id=user_id, token_hash=_hash_token(token), expires_at=datetime.utcnow().replace(microsecond=0))
    from datetime import timedelta
    rec.expires_at = datetime.utcnow() + timedelta(minutes=SESSION_MINUTES)
    db.add(rec)
    return token

def audit(db, user_id, role, action, resource, request_id=None):
    db.add(AuditEvent(user_id=user_id, role=role, action=action, resource=resource, request_id=request_id))

def authenticate_request(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in again.")
    token = auth[7:].strip()
    db = SessionLocal()
    try:
        rec = db.query(UserSession).filter(UserSession.token_hash == _hash_token(token), UserSession.revoked_at.is_(None)).first()
        if not rec or rec.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
        user = db.query(User).filter(User.id == rec.user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User session is invalid.")
        rec.last_seen_at = datetime.utcnow()
        user_id, user_role = user.id, user.role
        db.commit()
        return {"id": user_id, "role": user_role}
    finally:
        db.close()

app = FastAPI(
    title="SIH26047 Clinical AI API",
    version="8.0.0",
    description="SIH26047 Phase 5B: cumulative clinical prototype with validation, physician AI synthesis with local AI/NLP symptom understanding, consent, AYUSH, documents, and FHIR-ready interoperability, including AI-3C document classification."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# AI-3A unified document-intake module.
register_ai3a(app, storage_dir=str(BASE_DIR / "data" / "documents"))

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # CORS preflight must remain public. All application APIs except auth/health
    # require a valid short-lived bearer session.
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS or not request.url.path.startswith("/api/"):
        response = await call_next(request)
    else:
        try:
            authenticate_request(request)
            response = await call_next(request)
        except HTTPException as exc:
            from starlette.responses import JSONResponse
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else response.headers.get("Cache-Control", "")
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
    return response


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2)
    email: str
    password: str = Field(min_length=6)
    age: int = Field(ge=1, le=120)
    gender: str
    phone: str = ""


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=2)
    age: int = Field(ge=1, le=120)
    gender: str = ""
    phone: str = ""
    blood_group: str = ""
    allergies: str = ""
    conditions: str = ""
    medications: str = ""
    address: str = ""


class InterviewAnswer(BaseModel):
    patient_id: int
    session_id: str
    question_id: str
    answer: str = Field(min_length=1)
    answers: dict = Field(default_factory=dict)
    language: str = "en-IN"


class ClinicalNLURequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "en-IN"
    question_id: Optional[str] = None


class ConversationRepairRequest(BaseModel):
    patient_id: int
    session_id: str
    text: str = ""
    event: str = "answer"
    attempt: int = Field(default=0, ge=0, le=10)
    language: str = "en-IN"
    input_mode: str = "voice"


class InterviewComplete(BaseModel):
    patient_id: int
    session_id: str
    title: str
    answers: dict
    structured: dict


class ConsentRequest(BaseModel):
    patient_id: int
    language: str = "en-IN"
    audio_explained: bool = False
    consent_type: str = "Clinical case-taking"
    granted: bool = True


class AccessibilityPreferenceRequest(BaseModel):
    language: str = "en-IN"
    input_mode: str = "touch"
    font_scale: str = "1.0"
    high_contrast: bool = False
    reduced_motion: bool = False
    captions: bool = True
    audio_enabled: bool = True
    audio_speed: str = "1.0"
    assisted_mode: bool = False


class ClinicalQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class FHIRExportRequest(BaseModel):
    patient_id: int
    exported_by: Optional[int] = None
    abha_address: Optional[str] = None

class ABDMExportRequest(BaseModel):
    patient_id: int
    exported_by: Optional[int] = None
    abha_address: Optional[str] = None
    facility_id: str = "SIH26047-DEMO-FACILITY"
    practitioner_id: str = "SIH26047-DEMO-HPR"
    consent_reference: Optional[str] = None

class EncounterCreateRequest(BaseModel):
    patient_id: int
    department: str = Field(default="General Medicine", min_length=1, max_length=80)
    priority: str = Field(default="normal", min_length=1, max_length=20)
    reason: str = Field(default="", max_length=500)
    doctor_id: Optional[int] = None


class EncounterStatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=30)
    doctor_id: Optional[int] = None


class TriageActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=30)
    notes: str = Field(default="", max_length=3000)
    priority: Optional[str] = Field(default=None, max_length=20)


class ConsultationStartRequest(BaseModel):
    encounter_id: int
    title: str = Field(default="Clinical consultation", min_length=1, max_length=120)


class ConsultationUpdateRequest(BaseModel):
    sections: dict = Field(default_factory=dict)
    doctor_notes: str = Field(default="", max_length=5000)


class ConsultationCompleteRequest(BaseModel):
    sections: dict = Field(default_factory=dict)
    doctor_notes: str = Field(default="", max_length=5000)
    doctor_review: str = Field(default="Completed", min_length=1, max_length=40)



class AdminDepartmentRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    specialty: str = Field(default="General Medicine", min_length=2, max_length=100)
    active: bool = True


class AdminDoctorRequest(BaseModel):
    user_id: Optional[int] = None
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    email: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=6, max_length=200)
    specialty: str = Field(default="General Medicine", min_length=2, max_length=100)
    department: str = Field(default="General Medicine", min_length=2, max_length=100)
    registration_number: str = Field(default="", max_length=100)
    active: bool = True


class AdminOPDRequest(BaseModel):
    department: str = Field(min_length=2, max_length=100)
    working_days: List[str] = Field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])
    start_time: str = Field(default="09:00", min_length=5, max_length=5)
    end_time: str = Field(default="17:00", min_length=5, max_length=5)
    active: bool = True


class AdminAvailabilityRequest(BaseModel):
    doctor_id: int
    day_of_week: str = Field(min_length=3, max_length=9)
    start_time: str = Field(min_length=5, max_length=5)
    end_time: str = Field(min_length=5, max_length=5)
    active: bool = True


class AdminRoutingRequest(BaseModel):
    department: str = Field(min_length=2, max_length=100)
    specialty: str = Field(min_length=2, max_length=100)
    doctor_id: Optional[int] = None
    priority: int = Field(default=100, ge=0, le=10000)
    active: bool = True


class AdminHospitalRequest(BaseModel):
    hospital_name: str = Field(min_length=2, max_length=200)
    facility_code: str = Field(min_length=2, max_length=100)
    timezone: str = Field(default="Asia/Kolkata", min_length=3, max_length=80)
    default_department: str = Field(default="General Medicine", min_length=2, max_length=100)
    active: bool = True


def user_response(user):
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}


def profile_response(user):
    p = user.profile
    return {
        "id": user.id, "name": user.name, "email": user.email, "role": user.role,
        "age": p.age if p else None, "gender": p.gender if p else "",
        "phone": p.phone if p else "", "blood_group": p.blood_group if p else "",
        "allergies": p.allergies if p else "", "conditions": p.conditions if p else "",
        "medications": p.medications if p else "", "address": p.address if p else ""
    }


@app.get("/")
def root():
    return {"project": "SIH26047 Clinical AI", "phase": "Phase 5H - Voice & Accessibility", "status": "running"}


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "SIH26047 Final MVP backend is running", "version": "8.7.0"}


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload.email.strip().lower()).first()
        if not user or user.password_hash != hash_password(payload.password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = create_session(db, user.id)
        audit(db, user.id, user.role, "login", "auth")
        db.commit()
        return {"message": "Login successful", "user": user_response(user), "access_token": token, "token_type": "bearer", "expires_in_minutes": SESSION_MINUTES}
    finally:
        db.close()


@app.post("/api/auth/register")
def register(payload: RegisterRequest):
    db = SessionLocal()
    try:
        email = payload.email.strip().lower()
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        user = User(
            name=payload.name.strip(), email=email,
            password_hash=hash_password(payload.password), role="patient"
        )
        db.add(user)
        db.flush()
        make_profile(db, user, payload.age, payload.gender, payload.phone)
        token = create_session(db, user.id)
        audit(db, user.id, user.role, "register", "auth")
        db.commit()
        db.refresh(user)
        return {"message": "Patient account created", "user": user_response(user), "access_token": token, "token_type": "bearer", "expires_in_minutes": SESSION_MINUTES}
    finally:
        db.close()


@app.post("/api/auth/logout")
def logout(request: Request):
    auth = request.headers.get("Authorization", "")
    db = SessionLocal()
    try:
        rec = db.query(UserSession).filter(UserSession.token_hash == _hash_token(auth[7:].strip()) if auth.startswith("Bearer ") else "").first()
        if rec:
            rec.revoked_at = datetime.utcnow()
            audit(db, rec.user_id, None, "logout", "auth")
            db.commit()
        return {"message": "Signed out securely."}
    finally:
        db.close()

@app.get("/api/system/integration-audit")
def integration_audit(request: Request):
    user = authenticate_request(request)
    if user["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Integration audit is restricted to authorized clinical/administrative roles.")
    result = build_integration_audit(app, engine, BASE_DIR)
    db = SessionLocal()
    try:
        audit(db, user["id"], user["role"], "integration_audit", "system")
        db.commit()
    finally:
        db.close()
    return result


@app.get("/api/security/status")
def security_status(request: Request):
    user = authenticate_request(request)
    db = SessionLocal()
    try:
        user_id = user["id"]
        user_role = user["role"]
        active = db.query(UserSession).filter(UserSession.user_id==user_id, UserSession.revoked_at.is_(None), UserSession.expires_at>datetime.utcnow()).count()
        audits = db.query(AuditEvent).filter(AuditEvent.user_id==user_id).count()
        return {"role":user_role,"session_active":True,"active_sessions":active,"audit_events":audits,"controls":["Bearer session authentication","Short-lived sessions","Role-aware accounts","Consent gate for ABDM export","Audit logging","Security response headers"],"production_note":"Prototype security foundation; production deployment still requires HTTPS, managed secrets, hardened identity provider, encryption/key management and formal security assessment."}
    finally:
        db.close()

@app.get("/api/audit/me")
def audit_me(request: Request):
    user = authenticate_request(request)
    db=SessionLocal()
    try:
        rows=db.query(AuditEvent).filter(AuditEvent.user_id==user["id"]).order_by(AuditEvent.created_at.desc()).limit(50).all()
        return {"events":[{"id":r.id,"action":r.action,"resource":r.resource,"created_at":r.created_at.isoformat()} for r in rows]}
    finally: db.close()

@app.get("/api/patients")
def get_patients():
    db = SessionLocal()
    try:
        patients = db.query(User).filter(User.role == "patient").order_by(User.id).all()
        return {"patients": [
            {"id": p.id, "name": p.name, "email": p.email,
             "age": p.profile.age if p.profile else None,
             "gender": p.profile.gender if p.profile else ""}
            for p in patients
        ]}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not user:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return profile_response(user)
    finally:
        db.close()


@app.put("/api/patients/{patient_id}")
def update_patient(patient_id: int, payload: ProfileUpdate):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not user:
            raise HTTPException(status_code=404, detail="Patient not found.")
        user.name = payload.name.strip()
        p = user.profile
        if not p:
            p = PatientProfile(user_id=user.id)
            db.add(p)
        p.age, p.gender, p.phone = payload.age, payload.gender, payload.phone
        p.blood_group, p.allergies = payload.blood_group, payload.allergies
        p.conditions, p.medications, p.address = payload.conditions, payload.medications, payload.address
        db.commit()
        db.refresh(user)
        return {"message": "Profile updated successfully", "patient": profile_response(user)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/consultations")
def get_consultations(patient_id: int):
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.id == patient_id, User.role == "patient").first():
            raise HTTPException(status_code=404, detail="Patient not found.")
        items = db.query(Consultation).filter(
            Consultation.patient_id == patient_id
        ).order_by(Consultation.created_at.desc()).all()
        return {"consultations": [
            {"id": i.id, "title": i.title, "summary": i.summary,
             "status": i.status, "risk_level": i.risk_level or "none",
             "red_flags": json.loads(i.red_flags) if i.red_flags else [],
             "created_at": i.created_at.isoformat()}
            for i in items
        ]}
    finally:
        db.close()


# ---------- Phase 5B: Encounter + Queue workflow ----------
@app.get("/api/doctor/encounters/{encounter_id}/workspace")
def get_ai5c_doctor_workspace(encounter_id: int, request: Request):
    """AI-5C: read-only clinician workspace assembled from persisted data and existing AI layers."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Doctor workspace is restricted to clinical staff.")
    db = SessionLocal()
    try:
        encounter = db.query(Encounter).filter(Encounter.id == encounter_id).first()
        if not encounter:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        patient = db.query(User).filter(User.id == encounter.patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient.id).order_by(MedicalDocument.created_at.desc()).all()
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient.id).order_by(Consultation.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        summary = build_clinical_summary(patient, consultations, documents, timeline)
        risk = build_risk_assessment(patient, consultations, documents, summary)
        decision_support = build_decision_support(patient, consultations, documents, summary, risk)
        medication = build_medication_intelligence(patient, documents, consultations)
        investigations = build_investigation_intelligence(patient, documents)
        copilot = {}
        clinical_gate = {}
        if consultations:
            latest = consultations[0]
            copilot = build_consultation_copilot(patient, latest, documents, summary, risk, medication, investigations)
            clinical_gate = build_final_clinical_gate(patient, latest, summary, risk, decision_support, medication, investigations, copilot)
        workspace = build_doctor_workspace(patient, encounter, documents, consultations, summary, risk, decision_support, medication, investigations, copilot, clinical_gate, timeline)
        audit(db, actor["id"], actor["role"], "doctor_workspace_viewed", f"encounter:{encounter.id}")
        db.commit()
        return {"patient_id": patient.id, "encounter_id": encounter.id, "workspace": workspace}
    finally:
        db.close()


@app.post("/api/encounters")
def create_encounter(payload: EncounterCreateRequest, request: Request):
    actor = authenticate_request(request)
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == payload.patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        if actor["role"] == "patient" and actor["id"] != payload.patient_id:
            raise HTTPException(status_code=403, detail="Patients can create encounters only for themselves.")
        if actor["role"] not in ({"patient"} | ROLE_CAN_MANAGE_QUEUE):
            raise HTTPException(status_code=403, detail="Not authorized to create encounters.")
        try:
            department = normalize_department(payload.department)
            priority = normalize_priority(payload.priority)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if payload.doctor_id is not None:
            doctor = db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first()
            if not doctor:
                raise HTTPException(status_code=404, detail="Doctor not found.")
        else:
            doctor = None
        today = date.today().isoformat()
        active = db.query(Encounter).filter(Encounter.patient_id == payload.patient_id, Encounter.visit_date == today, Encounter.status.in_(["waiting", "called", "in_consultation"])).first()
        if active:
            raise HTTPException(status_code=409, detail="Patient already has an active encounter for today.")
        # SQLite serializes writes; the transaction prevents two committed tokens
        # from being generated from the same observed maximum.
        max_token = db.query(Encounter).filter(Encounter.visit_date == today, Encounter.department == department).order_by(Encounter.token_number.desc()).first()
        token = (max_token.token_number + 1) if max_token else 1
        encounter = Encounter(patient_id=payload.patient_id, doctor_id=payload.doctor_id, department=department, visit_date=today, token_number=token, priority=priority, status="waiting", reason=payload.reason.strip())
        db.add(encounter)
        audit(db, actor["id"], actor["role"], "encounter_created", f"encounter:{payload.patient_id}")
        db.commit(); db.refresh(encounter)
        return {"message": "Encounter created", "encounter": serialize_encounter(encounter, patient, doctor)}
    finally:
        db.close()


@app.get("/api/encounters/{encounter_id}")
def get_encounter(encounter_id: int, request: Request):
    actor = authenticate_request(request)
    db = SessionLocal()
    try:
        encounter = db.query(Encounter).filter(Encounter.id == encounter_id).first()
        if not encounter:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        if actor["role"] == "patient" and actor["id"] != encounter.patient_id:
            raise HTTPException(status_code=403, detail="Not authorized to view this encounter.")
        patient = db.query(User).filter(User.id == encounter.patient_id).first()
        doctor = db.query(User).filter(User.id == encounter.doctor_id).first() if encounter.doctor_id else None
        return {"encounter": serialize_encounter(encounter, patient, doctor)}
    finally:
        db.close()


@app.get("/api/queue")
def get_queue(request: Request, department: str = "General Medicine", visit_date: Optional[str] = None, status: str = "waiting,called,in_consultation"):
    actor = authenticate_request(request)
    if actor["role"] not in ROLE_CAN_MANAGE_QUEUE:
        raise HTTPException(status_code=403, detail="Queue access is restricted to clinical/administrative roles.")
    db = SessionLocal()
    try:
        try:
            department = normalize_department(department)
            statuses = [normalize_status(x) for x in status.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not statuses:
            raise HTTPException(status_code=422, detail="At least one queue status is required.")
        target_date = visit_date or date.today().isoformat()
        rows = db.query(Encounter).filter(Encounter.department == department, Encounter.visit_date == target_date, Encounter.status.in_(statuses)).all()
        rows.sort(key=queue_sort_key)
        items=[]
        for e in rows:
            patient=db.query(User).filter(User.id==e.patient_id).first()
            doctor=db.query(User).filter(User.id==e.doctor_id).first() if e.doctor_id else None
            item=serialize_encounter(e, patient, doctor)
            item["queue_position"] = len(items)+1
            items.append(item)
        return {"department": department, "visit_date": target_date, "count": len(items), "queue": items}
    finally:
        db.close()


@app.post("/api/encounters/{encounter_id}/status")
def update_encounter_status(encounter_id: int, payload: EncounterStatusRequest, request: Request):
    actor = authenticate_request(request)
    if actor["role"] not in ROLE_CAN_MANAGE_QUEUE:
        raise HTTPException(status_code=403, detail="Only clinical/administrative staff can change queue status.")
    db = SessionLocal()
    try:
        encounter = db.query(Encounter).filter(Encounter.id == encounter_id).first()
        if not encounter:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        try:
            target = normalize_status(payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if target == encounter.status:
            raise HTTPException(status_code=409, detail="Encounter is already in that status.")
        if not can_transition(encounter.status, target):
            raise HTTPException(status_code=409, detail=f"Invalid status transition: {encounter.status} -> {target}.")
        if target in {"in_consultation", "completed"} and actor["role"] not in {"doctor", "triage", "admin"}:
            raise HTTPException(status_code=403, detail="Not authorized for consultation status changes.")
        if payload.doctor_id is not None:
            doctor = db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first()
            if not doctor:
                raise HTTPException(status_code=404, detail="Doctor not found.")
            encounter.doctor_id = payload.doctor_id
        now = datetime.utcnow()
        encounter.status = target
        if target == "called":
            encounter.called_at = now
        if target == "completed":
            encounter.completed_at = now
        audit(db, actor["id"], actor["role"], "encounter_status_changed", f"encounter:{encounter.id}")
        db.commit(); db.refresh(encounter)
        patient=db.query(User).filter(User.id==encounter.patient_id).first()
        doctor=db.query(User).filter(User.id==encounter.doctor_id).first() if encounter.doctor_id else None
        return {"message":"Encounter status updated", "encounter":serialize_encounter(encounter, patient, doctor)}
    finally:
        db.close()


# ---------- Phase 4E: Medical document intelligence ----------
ALLOWED_DOC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}
DOCUMENT_TYPES = ["Prescription", "Lab Report", "Discharge Summary", "Imaging Report", "Other"]


def detect_document_type(filename: str, text: str, requested: str = "Other") -> str:
    if requested in DOCUMENT_TYPES and requested != "Other":
        return requested
    t = (filename + " " + text[:4000]).lower()
    if any(x in t for x in ["prescription", "rx", "tablet", "capsule", "dose", "mg"]): return "Prescription"
    if any(x in t for x in ["laboratory", "lab report", "haemoglobin", "hemoglobin", "glucose", "creatinine", "cbc"]): return "Lab Report"
    if any(x in t for x in ["discharge summary", "discharged", "admission", "hospital course"]): return "Discharge Summary"
    if any(x in t for x in ["x-ray", "xray", "ct scan", "mri", "ultrasound", "impression:"]): return "Imaging Report"
    return "Other"


def extract_document_text(path: Path, mime_type: str) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(errors="ignore")[:100000]
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)[:100000]
        except Exception:
            return ""
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(path))[:100000]
        except Exception:
            return ""
    return ""


def extract_medical_entities(text: str, document_type: str):
    raw = text or ""
    findings = []
    patterns = [
        ("Hemoglobin", r"(?:ha?emoglobin|hb)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:g/?dl)?"),
        ("Blood glucose", r"(?:blood glucose|glucose|blood sugar)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?"),
        ("Creatinine", r"creatinine\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?"),
        ("Blood pressure", r"(?:blood pressure|bp)\s*[:=]?\s*(\d{2,3}\s*/\s*\d{2,3})"),
        ("Temperature", r"(?:temperature|temp)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:°?c|f)?"),
        ("Pulse", r"(?:pulse|heart rate|hr)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?"),
    ]
    for label, pattern in patterns:
        m = re.search(pattern, raw, re.I)
        if m:
            findings.append({"type": "measurement", "label": label, "value": m.group(1).strip(), "source": "OCR/text extraction"})
    # Simple medication candidates from common prescription lines.
    for line in raw.splitlines():
        line=line.strip()
        if re.search(r"\b\d+\s*(?:mg|mcg|ml)\b", line, re.I) and len(line) < 180:
            findings.append({"type": "medication_candidate", "label": "Medication", "value": line, "source": "OCR/text extraction"})
    # Date candidates help create a timeline without inventing a date.
    dates = re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", raw)
    if dates:
        findings.append({"type": "date", "label": "Document date", "value": dates[0], "source": "Text extraction"})
    return findings


@app.post("/api/documents/upload")
async def upload_document(patient_id: int = Form(...), document_type: str = Form("Other"), file: UploadFile = File(...)):
    if document_type not in DOCUMENT_TYPES:
        document_type = "Other"
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_DOC_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Supported files: PDF, PNG, JPG, JPEG, WEBP or TXT.")
        data = await file.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Please upload a file smaller than 10 MB.")
        safe_name = f"{uuid.uuid4().hex}{ext}"
        target = UPLOAD_DIR / safe_name
        target.write_bytes(data)
        text = extract_document_text(target, file.content_type or "")
        classification = classify_document(text, file.filename or "document", document_type)
        # Preserve the old document_type contract while allowing AI-3C to correct
        # an inaccurate user-selected/legacy label when evidence is strong.
        dtype = classification.document_class if classification.document_class != "Other" else detect_document_type(file.filename or "document", text, document_type)
        extraction = extract_structured_medical_data(text, dtype)
        findings = extract_medical_entities(text, dtype)
        # Keep legacy findings for frontend compatibility and expose AI-3D structured items alongside them.
        findings = findings + extraction.get("items", [])
        status = "Processed" if text.strip() else "Uploaded — OCR unavailable or no readable text"
        doc = MedicalDocument(patient_id=patient_id, filename=file.filename or safe_name, document_type=dtype,
                              mime_type=file.content_type or "", stored_path=str(target), extracted_text=text,
                              extracted_data=json.dumps(findings),
                              classification=classification.document_class,
                              classification_confidence=str(classification.confidence),
                              classification_method=classification.method,
                              classification_evidence=json.dumps(classification.evidence),
                              classification_needs_review=int(classification.needs_review),
                              structured_extraction=json.dumps(extraction),
                              extraction_needs_review=int(extraction.get("needs_review", True)),
                              extraction_method="AI-3D explainable local extractor",
                              verification_status="Pending",
                              status=status)
        db.add(doc); db.commit(); db.refresh(doc)
        return {"message":"Document processed", "document": {"id":doc.id,"filename":doc.filename,"document_type":doc.document_type,
                "status":doc.status,"created_at":doc.created_at.isoformat(),"findings":findings,
                "classification": {"document_class": classification.document_class, "confidence": classification.confidence,
                                   "needs_review": classification.needs_review, "method": classification.method,
                                   "evidence": classification.evidence, "scores": classification.scores},
                "extraction": extraction,
                "text_preview":(text[:1000] if text else "No text could be extracted. You can still keep the document for practitioner review.")}}
    finally:
        db.close()


class DocumentClassificationRequest(BaseModel):
    text: str = Field(default="", max_length=100000)
    filename: str = Field(default="document", max_length=255)
    requested_type: str = Field(default="Other", max_length=80)


@app.post("/api/documents/classify")
def classify_document_text(payload: DocumentClassificationRequest):
    result = classify_document(payload.text, payload.filename, payload.requested_type)
    return {"classification": {
        "document_class": result.document_class, "confidence": result.confidence,
        "needs_review": result.needs_review, "method": result.method,
        "evidence": result.evidence, "scores": result.scores
    }}


@app.post("/api/documents/{document_id}/classify")
def classify_stored_document(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        result = classify_document(doc.extracted_text or "", doc.filename or "document", doc.document_type or "Other")
        doc.classification = result.document_class
        doc.classification_confidence = str(result.confidence)
        doc.classification_method = result.method
        doc.classification_evidence = json.dumps(result.evidence)
        doc.classification_needs_review = int(result.needs_review)
        if result.document_class != "Other" and result.confidence >= 0.72:
            doc.document_type = result.document_class
        extraction = extract_structured_medical_data(doc.extracted_text or "", doc.document_type or result.document_class)
        doc.structured_extraction = json.dumps(extraction)
        doc.extraction_needs_review = int(extraction.get("needs_review", True))
        doc.extraction_method = "AI-3D explainable local extractor"
        db.commit()
        return {"message": "Document classified", "classification": {
            "document_class": result.document_class, "confidence": result.confidence,
            "needs_review": result.needs_review, "method": result.method,
            "evidence": result.evidence, "scores": result.scores
        }}
    finally:
        db.close()


class DocumentExtractionRequest(BaseModel):
    text: str = Field(default="", max_length=100000)
    document_type: str = Field(default="Other", max_length=80)


@app.post("/api/documents/extract")
def extract_document_text_data(payload: DocumentExtractionRequest):
    extraction = extract_structured_medical_data(payload.text, payload.document_type)
    return {"extraction": extraction}


@app.post("/api/documents/{document_id}/extract")
def extract_stored_document(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        extraction = extract_structured_medical_data(doc.extracted_text or "", doc.document_type or "Other")
        doc.structured_extraction = json.dumps(extraction)
        doc.extraction_needs_review = int(extraction.get("needs_review", True))
        doc.extraction_method = "AI-3D explainable local extractor"
        legacy = json.loads(doc.extracted_data or "[]")
        doc.extracted_data = json.dumps(legacy + extraction.get("items", []))
        db.commit()
        return {"message": "Document structured data extracted", "extraction": extraction}
    finally:
        db.close()


class DocumentVerificationRequest(BaseModel):
    verified_items: List[Dict[str, Any]] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)
    verified_by: Optional[int] = None


@app.get("/api/documents/{document_id}/verification")
def get_document_verification(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        extraction = json.loads(doc.structured_extraction or "{}")
        summary = build_verification_summary(extraction)
        return {
            "document_id": doc.id,
            "verification_status": doc.verification_status or "Pending",
            "summary": summary,
            "verified_data": json.loads(doc.verified_data or "null"),
            "verification_notes": doc.verification_notes or "",
            "verified_by": doc.verified_by,
            "verified_at": doc.verified_at.isoformat() if doc.verified_at else None,
        }
    finally:
        db.close()


@app.post("/api/documents/{document_id}/verify")
def verify_stored_document(document_id: int, payload: DocumentVerificationRequest):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        extraction = json.loads(doc.structured_extraction or "{}")
        if not extraction:
            raise HTTPException(status_code=400, detail="No AI-3D extraction is available to verify.")
        try:
            verified = apply_document_verification(extraction, payload.verified_items)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        doc.verified_data = json.dumps(verified)
        doc.verification_notes = payload.notes.strip() or None
        doc.verified_by = payload.verified_by
        doc.verified_at = datetime.utcnow()
        doc.verification_status = verified.get("verification_status", "Pending")
        db.commit()
        return {
            "message": "Document extraction verified",
            "document_id": doc.id,
            "verification_status": doc.verification_status,
            "verified_data": verified,
            "verification_notes": doc.verification_notes or "",
            "verified_by": doc.verified_by,
            "verified_at": doc.verified_at.isoformat(),
        }
    finally:
        db.close()


@app.get("/api/documents/{document_id}/explainability")
def get_document_explainability(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"document_id": doc.id, "explainability": explain_document(doc)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/timeline")
def get_patient_timeline(patient_id: int):
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).all()
        timeline = build_timeline(consultations, documents)
        return {"patient_id": patient_id, "timeline": timeline}
    finally:
        db.close()


@app.get("/api/documents/{document_id}/timeline-context")
def get_document_timeline_context(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == doc.patient_id).all()
        consultations = db.query(Consultation).filter(Consultation.patient_id == doc.patient_id).all()
        timeline = build_timeline(consultations, documents)
        return {"document_id": document_id, "patient_id": doc.patient_id, "timeline": timeline, "explainability": explain_timeline(timeline)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/clinical-summary")
def get_ai4a_clinical_summary(patient_id: int):
    """AI-4A: consolidated clinician-facing summary from stored, traceable data."""
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        return {"patient_id": patient_id, "clinical_summary": build_clinical_summary(patient, consultations, documents, timeline)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/risk-assessment")
def get_ai4b_risk_assessment(patient_id: int, request: Request):
    """AI-4B: clinician-facing conservative risk/red-flag assessment."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Clinical risk assessment is restricted to clinical staff.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        summary = build_clinical_summary(patient, consultations, documents, timeline)
        return {"patient_id": patient_id, "risk_assessment": build_risk_assessment(patient, consultations, documents, summary)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/decision-support")
def get_ai4c_decision_support(patient_id: int, request: Request):
    """AI-4C: conservative clinician decision-support prompts and record checks."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Decision support is restricted to clinical staff.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        summary = build_clinical_summary(patient, consultations, documents, timeline)
        risk = build_risk_assessment(patient, consultations, documents, summary)
        support = build_decision_support(patient, consultations, documents, summary, risk)
        return {"patient_id": patient_id, "decision_support": support}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/medication-intelligence")
def get_ai4d_medication_intelligence(patient_id: int, request: Request):
    """AI-4D: conservative medication reconciliation for clinical staff."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Medication intelligence is restricted to clinical staff.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
        return {"patient_id": patient_id, "medication_intelligence": build_medication_intelligence(patient, documents, consultations)}
    finally:
        db.close()


@app.get("/api/doctor/consultations/{consultation_id}/copilot")
def get_ai4g_consultation_copilot(consultation_id: int, request: Request):
    """AI-4G: clinician-only consultation copilot; read-only draft assistance."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Consultation copilot is restricted to clinical staff.")
    db = SessionLocal()
    try:
        consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        patient = db.query(User).filter(User.id == consultation.patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient.id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient.id).order_by(MedicalDocument.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        summary = build_clinical_summary(patient, consultations, documents, timeline)
        risk = build_risk_assessment(patient, consultations, documents, summary)
        medication = build_medication_intelligence(patient, documents, consultations)
        investigations = build_investigation_intelligence(patient, documents)
        copilot = build_consultation_copilot(patient, consultation, documents, summary, risk, medication, investigations)
        return {"patient_id": patient.id, "consultation_id": consultation.id, "consultation_copilot": copilot}
    finally:
        db.close()


@app.post("/api/patients/{patient_id}/clinical-question")
def post_ai4f_clinical_question(patient_id: int, payload: ClinicalQuestionRequest, request: Request):
    """AI-4F: source-grounded clinical question answering for clinical staff."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Clinical question assistant is restricted to clinical staff.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
        return {"patient_id": patient_id, "clinical_question": answer_clinical_question(patient, consultations, documents, payload.question)}
    finally:
        db.close()


@app.get("/api/doctor/consultations/{consultation_id}/clinical-gate")
def get_ai4h_clinical_gate(consultation_id: int, request: Request):
    """AI-4H: final clinician-facing safety/readiness gate; read-only."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Final clinical gate is restricted to clinical staff.")
    db = SessionLocal()
    try:
        consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        patient = db.query(User).filter(User.id == consultation.patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient.id).order_by(Consultation.created_at.desc()).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient.id).order_by(MedicalDocument.created_at.desc()).all()
        timeline = build_timeline(consultations, documents)
        summary = build_clinical_summary(patient, consultations, documents, timeline)
        risk = build_risk_assessment(patient, consultations, documents, summary)
        support = build_decision_support(patient, consultations, documents, summary, risk)
        medication = build_medication_intelligence(patient, documents, consultations)
        investigations = build_investigation_intelligence(patient, documents)
        copilot = build_consultation_copilot(patient, consultation, documents, summary, risk, medication, investigations)
        gate = build_final_clinical_gate(patient, consultation, summary, risk, support, medication, investigations, copilot)
        return {"patient_id": patient.id, "consultation_id": consultation.id, "clinical_gate": gate}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/investigation-intelligence")
def get_ai4e_investigation_intelligence(patient_id: int, request: Request):
    """AI-4E: conservative longitudinal investigation intelligence for clinical staff."""
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Investigation intelligence is restricted to clinical staff.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.asc()).all()
        return {"patient_id": patient_id, "investigation_intelligence": build_investigation_intelligence(patient, documents)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/clinical-handoff")
def get_patient_clinical_handoff(patient_id: int):
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).all()
        return {"patient_id": patient_id, "clinical_handoff": build_clinical_handoff(patient_id, consultations, documents)}
    finally:
        db.close()


@app.get("/api/documents/{document_id}/clinical-handoff")
def get_document_clinical_handoff(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(MedicalDocument).filter(MedicalDocument.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        consultations = db.query(Consultation).filter(Consultation.patient_id == doc.patient_id).all()
        documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == doc.patient_id).all()
        return {"document_id": document_id, "patient_id": doc.patient_id, "clinical_handoff": build_clinical_handoff(doc.patient_id, consultations, documents)}
    finally:
        db.close()


@app.get("/api/patients/{patient_id}/documents")
def get_patient_documents(patient_id: int):
    db=SessionLocal()
    try:
        if not db.query(User).filter(User.id==patient_id, User.role=="patient").first(): raise HTTPException(status_code=404, detail="Patient not found.")
        docs=db.query(MedicalDocument).filter(MedicalDocument.patient_id==patient_id).order_by(MedicalDocument.created_at.desc()).all()
        return {"documents":[{"id":d.id,"filename":d.filename,"document_type":d.document_type,"status":d.status,
                "created_at":d.created_at.isoformat(),"findings":json.loads(d.extracted_data or "[]"),
                "classification":{"document_class":d.classification or d.document_type,"confidence":float(d.classification_confidence or 0),"needs_review":bool(d.classification_needs_review),"method":d.classification_method,"evidence":json.loads(d.classification_evidence or "[]")},"extraction":json.loads(d.structured_extraction or "{}"),"verification":{"status":d.verification_status or "Pending","verified_data":json.loads(d.verified_data or "null"),"notes":d.verification_notes or "","verified_by":d.verified_by,"verified_at":d.verified_at.isoformat() if d.verified_at else None},
                "text_preview":(d.extracted_text or "")[:800]} for d in docs]}
    finally: db.close()


@app.get("/api/doctor/patients/{patient_id}/workspace")
def doctor_patient_workspace(patient_id: int):
    db=SessionLocal()
    try:
        patient=db.query(User).filter(User.id==patient_id,User.role=="patient").first()
        if not patient: raise HTTPException(status_code=404, detail="Patient not found.")
        consultations=db.query(Consultation).filter(Consultation.patient_id==patient_id).order_by(Consultation.created_at.desc()).all()
        docs=db.query(MedicalDocument).filter(MedicalDocument.patient_id==patient_id).order_by(MedicalDocument.created_at.desc()).all()
        timeline=[]
        for c in consultations:
            timeline.append({"date":c.created_at.isoformat(),"type":"Consultation","title":c.title,"detail":c.summary,"risk_level":c.risk_level or "none"})
        for d in docs:
            timeline.append({"date":d.created_at.isoformat(),"type":"Document","title":d.filename,"detail":d.document_type,"risk_level":"none"})
        timeline.sort(key=lambda x:x["date"], reverse=True)
        return {"patient":profile_response(patient),
                "consultations":[{"id":c.id,"title":c.title,"summary":c.summary,"status":c.status,"risk_level":c.risk_level or "none",
                  "red_flags":json.loads(c.red_flags or "[]"),"doctor_review":c.doctor_review or "Pending","doctor_notes":c.doctor_notes or "","ai_summary":json.loads(c.ai_summary) if c.ai_summary else None,"ai_summary_generated_at":c.ai_summary_generated_at.isoformat() if c.ai_summary_generated_at else None,"created_at":c.created_at.isoformat()} for c in consultations],
                "documents":[{"id":d.id,"filename":d.filename,"document_type":d.document_type,"status":d.status,"created_at":d.created_at.isoformat(),"findings":json.loads(d.extracted_data or "[]"),"classification":{"document_class":d.classification or d.document_type,"confidence":float(d.classification_confidence or 0),"needs_review":bool(d.classification_needs_review),"method":d.classification_method,"evidence":json.loads(d.classification_evidence or "[]")},"extraction":json.loads(d.structured_extraction or "{}"),"text_preview":(d.extracted_text or "")[:1200]} for d in docs],
                "timeline":timeline}
    finally: db.close()


class DoctorReviewUpdate(BaseModel):
    doctor_review: str = "Reviewed"
    doctor_notes: str = ""


@app.put("/api/doctor/consultations/{consultation_id}/review")
def review_consultation(consultation_id: int, payload: DoctorReviewUpdate):
    db=SessionLocal()
    try:
        c=db.query(Consultation).filter(Consultation.id==consultation_id).first()
        if not c: raise HTTPException(status_code=404, detail="Consultation not found.")
        c.doctor_review=payload.doctor_review[:40]
        c.doctor_notes=payload.doctor_notes[:5000]
        db.commit(); db.refresh(c)
        return {"message":"Doctor review saved","consultation_id":c.id,"doctor_review":c.doctor_review,"doctor_notes":c.doctor_notes}
    finally: db.close()


# ---------- Phase 5E: triage dashboard and human-controlled actions ----------
@app.get("/api/triage/dashboard")
def get_ai5e_triage_dashboard(request: Request, department: str = "General Medicine", visit_date: Optional[str] = None):
    """AI-5E: operational triage view with AI-4B signals surfaced for human review."""
    actor = authenticate_request(request)
    if actor["role"] not in TRIAGE_ROLES:
        raise HTTPException(status_code=403, detail="Triage dashboard is restricted to clinical/administrative roles.")
    db = SessionLocal()
    try:
        try:
            department = normalize_department(department)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        target_date = visit_date or date.today().isoformat()
        rows = db.query(Encounter).filter(
            Encounter.department == department,
            Encounter.visit_date == target_date,
            Encounter.status.in_(["waiting", "called", "in_consultation"]),
        ).all()
        items = []
        for encounter in rows:
            patient = db.query(User).filter(User.id == encounter.patient_id, User.role == "patient").first()
            if not patient:
                continue
            doctor = db.query(User).filter(User.id == encounter.doctor_id).first() if encounter.doctor_id else None
            consultations = db.query(Consultation).filter(Consultation.patient_id == patient.id).order_by(Consultation.created_at.desc()).all()
            documents = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient.id).order_by(MedicalDocument.created_at.desc()).all()
            timeline = build_timeline(consultations, documents)
            summary = build_clinical_summary(patient, consultations, documents, timeline)
            risk = build_risk_assessment(patient, consultations, documents, summary)
            items.append(serialize_triage_item(encounter, patient, doctor, risk))
        return {"patient_scope": "current operational queue", "triage_dashboard": build_triage_dashboard(items, department, target_date)}
    finally:
        db.close()


@app.post("/api/triage/encounters/{encounter_id}/action")
def act_on_ai5e_triage(encounter_id: int, payload: TriageActionRequest, request: Request):
    """AI-5E: acknowledgement/escalation/resolution is always a human action."""
    actor = authenticate_request(request)
    if actor["role"] not in TRIAGE_ROLES:
        raise HTTPException(status_code=403, detail="Only triage/clinical administrative staff can act on triage items.")
    try:
        action = normalize_triage_action(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db = SessionLocal()
    try:
        encounter = db.query(Encounter).filter(Encounter.id == encounter_id).first()
        if not encounter:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        if encounter.status in {"completed", "cancelled"}:
            raise HTTPException(status_code=409, detail="Completed or cancelled encounters cannot be triaged.")
        if payload.priority is not None:
            try:
                encounter.priority = normalize_priority(payload.priority)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
        target = triage_action_status(action)
        encounter.triage_status = target
        encounter.triage_notes = payload.notes.strip()
        encounter.triage_updated_by = actor["id"]
        encounter.triage_updated_at = datetime.utcnow()
        audit(db, actor["id"], actor["role"], f"triage_{action}", f"encounter:{encounter.id}")
        db.commit(); db.refresh(encounter)
        patient = db.query(User).filter(User.id == encounter.patient_id).first()
        doctor = db.query(User).filter(User.id == encounter.doctor_id).first() if encounter.doctor_id else None
        return {
            "message": f"Triage action '{action}' recorded",
            "encounter_id": encounter.id,
            "triage_status": encounter.triage_status,
            "priority": encounter.priority,
            "triage_notes": encounter.triage_notes or "",
            "human_action": True,
            "actor_role": actor["role"],
            "encounter": serialize_encounter(encounter, patient, doctor),
        }
    finally:
        db.close()


# ---------- Phase 5D: clinician-owned consultation workflow ----------
@app.post("/api/doctor/encounters/{encounter_id}/consultation")
def start_ai5d_consultation(encounter_id: int, payload: ConsultationStartRequest, request: Request):
    actor = authenticate_request(request)
    if actor["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can start a consultation.")
    if payload.encounter_id != encounter_id:
        raise HTTPException(status_code=400, detail="Encounter ID mismatch.")
    db = SessionLocal()
    try:
        encounter = db.query(Encounter).filter(Encounter.id == encounter_id).first()
        if not encounter:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        if encounter.status not in {"called", "in_consultation"}:
            raise HTTPException(status_code=409, detail="Encounter must be called or in consultation before starting.")
        if encounter.doctor_id not in (None, actor["id"]):
            raise HTTPException(status_code=403, detail="Encounter is assigned to another doctor.")
        if encounter.doctor_id is None:
            encounter.doctor_id = actor["id"]
        if encounter.status == "called":
            encounter.status = "in_consultation"
            encounter.called_at = encounter.called_at or datetime.utcnow()

        existing = db.query(Consultation).filter(Consultation.encounter_id == encounter_id).order_by(Consultation.created_at.desc()).first()
        if existing and (existing.consultation_status or "draft") != "completed":
            return {"consultation": build_consultation_payload(existing, encounter), "created": False}

        record = Consultation(
            patient_id=encounter.patient_id,
            encounter_id=encounter.id,
            title=payload.title.strip(),
            summary="Consultation draft.",
            status="In Consultation",
            consultation_status="in_progress",
            doctor_review="Pending",
            structured_data=json.dumps({"consultation": {}}, ensure_ascii=False),
            updated_at=datetime.utcnow(),
        )
        db.add(record)
        audit(db, actor["id"], actor["role"], "consultation_started", f"encounter:{encounter.id}")
        db.commit(); db.refresh(record)
        return {"consultation": build_consultation_payload(record, encounter), "created": True}
    finally:
        db.close()


@app.get("/api/doctor/consultations/{consultation_id}/record")
def get_ai5d_consultation(consultation_id: int, request: Request):
    actor = authenticate_request(request)
    if actor["role"] not in {"doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Consultation record is restricted to clinical staff.")
    db = SessionLocal()
    try:
        record = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        encounter = db.query(Encounter).filter(Encounter.id == record.encounter_id).first() if record.encounter_id else None
        if actor["role"] == "doctor" and encounter and encounter.doctor_id not in (None, actor["id"]):
            raise HTTPException(status_code=403, detail="Consultation is assigned to another doctor.")
        return {"consultation": build_consultation_payload(record, encounter)}
    finally:
        db.close()


@app.put("/api/doctor/consultations/{consultation_id}/record")
def update_ai5d_consultation(consultation_id: int, payload: ConsultationUpdateRequest, request: Request):
    actor = authenticate_request(request)
    if actor["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can edit consultation records.")
    db = SessionLocal()
    try:
        record = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        if record.consultation_status == "completed":
            raise HTTPException(status_code=409, detail="Completed consultations are locked for this prototype.")
        encounter = db.query(Encounter).filter(Encounter.id == record.encounter_id).first() if record.encounter_id else None
        if encounter and encounter.doctor_id not in (None, actor["id"]):
            raise HTTPException(status_code=403, detail="Consultation is assigned to another doctor.")
        try:
            sections = normalize_sections(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        record.structured_data = json.dumps({"consultation": sections}, ensure_ascii=False)
        record.summary = consultation_summary(sections)
        record.doctor_notes = payload.doctor_notes
        record.status = "In Consultation"
        record.consultation_status = "in_progress"
        record.updated_at = datetime.utcnow()
        audit(db, actor["id"], actor["role"], "consultation_draft_saved", f"consultation:{record.id}")
        db.commit(); db.refresh(record)
        return {"message": "Consultation draft saved", "consultation": build_consultation_payload(record, encounter)}
    finally:
        db.close()


@app.post("/api/doctor/consultations/{consultation_id}/complete")
def complete_ai5d_consultation(consultation_id: int, payload: ConsultationCompleteRequest, request: Request):
    actor = authenticate_request(request)
    if actor["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can complete a consultation.")
    db = SessionLocal()
    try:
        record = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        if record.consultation_status == "completed":
            raise HTTPException(status_code=409, detail="Consultation is already completed.")
        encounter = db.query(Encounter).filter(Encounter.id == record.encounter_id).first() if record.encounter_id else None
        if encounter and encounter.doctor_id not in (None, actor["id"]):
            raise HTTPException(status_code=403, detail="Consultation is assigned to another doctor.")
        try:
            sections = normalize_sections(payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        # Completing a consultation is a clinician action. We store exactly what
        # the doctor submitted; nothing here derives a diagnosis or treatment.
        record.structured_data = json.dumps({"consultation": sections}, ensure_ascii=False)
        record.summary = consultation_summary(sections)
        record.doctor_notes = payload.doctor_notes
        record.doctor_review = payload.doctor_review[:40]
        record.status = "Completed"
        record.consultation_status = "completed"
        record.updated_at = datetime.utcnow()
        if encounter:
            if encounter.status == "in_consultation":
                encounter.status = "completed"
            encounter.completed_at = encounter.completed_at or datetime.utcnow()
        audit(db, actor["id"], actor["role"], "consultation_completed", f"consultation:{record.id}")
        db.commit(); db.refresh(record)
        return {
            "message": "Consultation completed",
            "consultation": build_consultation_payload(record, encounter),
            "clinical_decision_source": "doctor_entered",
        }
    finally:
        db.close()


# ---------- Phase 5A AYUSH / Ayurveda assessment ----------
AYUSH_DASHAVIDHA = [
    {"id":"prakriti","title":"Prakriti (constitution)","question":"What is your usual body and temperament pattern? If known, describe your predominant constitution (Vata, Pitta, Kapha or combination).","hint":"Self-reported or practitioner-informed; do not force a constitution if unknown.","type":"single","options":["Vata","Pitta","Kapha","Vata-Pitta","Pitta-Kapha","Vata-Kapha","Tridoshic / balanced","Not sure"]},
    {"id":"vikriti","title":"Vikriti (current imbalance)","question":"Compared with your usual state, what has changed recently? Select the patterns you notice most.","hint":"This records the patient's reported current state; it is not an AI diagnosis.","type":"multi","options":["Dryness / irregularity","Heat / acidity","Heaviness / congestion","Sleep disturbance","Appetite change","Bowel change","Stress / restlessness","No major change","Not sure"]},
    {"id":"sara","title":"Sara (tissue quality)","question":"How would you describe the general strength and quality of your body tissues, as far as you know?","hint":"For prototype intake only; clinician assessment may be needed.","type":"single","options":["Strong / well nourished","Average","Low / easily fatigued","Not sure"]},
    {"id":"samhanana","title":"Samhanana (body build)","question":"How would you describe your overall body build and physical development?","hint":"Choose the closest self-description.","type":"single","options":["Well proportioned / strong","Average build","Thin / delicate","Heavy / broad","Not sure"]},
    {"id":"pramana","title":"Pramana (body measurements)","question":"Do you have recent height, weight, waist measurement or other body measurements you would like to record?","hint":"Use known measurements where available.","type":"text"},
    {"id":"satmya","title":"Satmya (adaptability / suitability)","question":"Which foods, routines, climate or daily habits generally suit you well, and which tend to cause discomfort?","hint":"Describe your usual tolerances and preferences.","type":"text"},
    {"id":"satva","title":"Satva (mental resilience)","question":"How would you describe your usual mental resilience and ability to cope with stress?","hint":"This is a self-reported wellbeing context, not a psychiatric assessment.","type":"single","options":["Generally resilient","Usually okay with some stress","Often overwhelmed","Prefer not to say"]},
    {"id":"ahara_shakti","title":"Ahara Shakti (digestive / food capacity)","question":"How is your appetite and digestion generally?","hint":"Consider appetite, meal tolerance and digestive comfort.","type":"single","options":["Good and regular","Variable","Low appetite","Heavy / slow digestion","Not sure"]},
    {"id":"vyayama_shakti","title":"Vyayama Shakti (exercise capacity)","question":"How much physical activity can you comfortably do in your usual day?","hint":"Choose your usual capacity, not your best day.","type":"single","options":["High","Moderate","Low","Very low / easily fatigued","Not sure"]},
    {"id":"vaya","title":"Vaya (age / life stage)","question":"What is your age, and if you wish, which life stage best describes you?","hint":"Age is also available from your patient profile.","type":"text"},
]

AYUSH_INTRO = "Welcome to the AYUSH / Ayurveda intake. This guided module records Dashavidha Pariksha-related information for practitioner review. It does not diagnose or determine a dosha on its own."
AYUSH_HI = {
 "prakriti":("प्रकृति (constitution)","आपके शरीर और स्वभाव का सामान्य पैटर्न कैसा है? यदि आपको पता हो, तो अपनी प्रमुख प्रकृति बताएं (वात, पित्त, कफ या मिश्रित)।"),
 "vikriti":("विकृति (वर्तमान असंतुलन)","आपकी सामान्य स्थिति की तुलना में हाल में क्या बदलाव आया है? जो पैटर्न महसूस हों उन्हें चुनें।"),
 "sara":("सार (धातु/ऊतक गुणवत्ता)","जहाँ तक आपको पता हो, अपने शरीर की सामान्य ताकत और पोषण की स्थिति कैसी है?"),
 "samhanana":("संहनन (शारीरिक बनावट)","आप अपनी सामान्य शारीरिक बनावट और विकास का वर्णन कैसे करेंगे?"),
 "pramana":("प्रमाण (शारीरिक माप)","क्या आपके पास हाल का कद, वजन, कमर का माप या अन्य शारीरिक माप हैं जिन्हें दर्ज करना चाहेंगे?"),
 "satmya":("सात्म्य (अनुकूलता)","कौन से भोजन, दिनचर्या, मौसम या आदतें आपको सामान्यतः अनुकूल रहती हैं और किनसे परेशानी होती है?"),
 "satva":("सत्त्व (मानसिक दृढ़ता)","तनाव से निपटने और मानसिक रूप से संभलने की आपकी सामान्य क्षमता कैसी है?"),
 "ahara_shakti":("आहार शक्ति (भोजन/पाचन क्षमता)","आपकी भूख और पाचन सामान्यतः कैसा रहता है?"),
 "vyayama_shakti":("व्यायाम शक्ति (शारीरिक क्षमता)","आप अपनी सामान्य दिनचर्या में कितनी शारीरिक गतिविधि आराम से कर सकते हैं?"),
 "vaya":("वय (आयु/जीवन अवस्था)","आपकी आयु क्या है और यदि चाहें तो कौन-सी जीवन अवस्था आपको सबसे अधिक उपयुक्त लगती है?"),
}

AYUSH_TRANSLATIONS = {
 "hi-IN": {
  "prakriti": ("प्रकृति (constitution)", "आपके शरीर और स्वभाव का सामान्य पैटर्न कैसा है? यदि आपको पता हो, तो अपनी प्रमुख प्रकृति बताएं (वात, पित्त, कफ या मिश्रित)।", "यदि पता न हो तो 'पता नहीं' चुनें।", ["वात","पित्त","कफ","वात-पित्त","पित्त-कफ","वात-कफ","त्रिदोषिक / संतुलित","पता नहीं"]),
  "vikriti": ("विकृति (वर्तमान बदलाव)", "आपकी सामान्य स्थिति की तुलना में हाल में क्या बदलाव आया है? जो पैटर्न महसूस हों उन्हें चुनें।", "जो बातें आपको सच लगती हैं उन्हें चुनें।", ["रूखापन / अनियमितता","गर्मी / अम्लता","भारीपन / जकड़न","नींद में बदलाव","भूख में बदलाव","मल त्याग में बदलाव","तनाव / बेचैनी","कोई बड़ा बदलाव नहीं","पता नहीं"]),
  "sara": ("सार (ऊतक गुणवत्ता)", "जहाँ तक आपको पता हो, अपने शरीर की सामान्य ताकत और पोषण की स्थिति कैसी है?", "यह केवल आपकी बताई हुई जानकारी है।", ["मजबूत / अच्छी तरह पोषित","सामान्य","कमज़ोर / जल्दी थकने वाला","पता नहीं"]),
  "samhanana": ("संहनन (शारीरिक बनावट)", "आप अपनी सामान्य शारीरिक बनावट और विकास का वर्णन कैसे करेंगे?", "सबसे करीब वाला विकल्प चुनें।", ["अच्छी बनावट / मजबूत","सामान्य बनावट","पतली / नाज़ुक","भारी / चौड़ी","पता नहीं"]),
  "pramana": ("प्रमाण (शारीरिक माप)", "क्या आपके पास हाल का कद, वजन, कमर का माप या अन्य शारीरिक माप हैं जिन्हें दर्ज करना चाहेंगे?", "जो माप पता हों, लिखें।", None),
  "satmya": ("सात्म्य (अनुकूलता)", "कौन से भोजन, दिनचर्या, मौसम या आदतें आपको सामान्यतः अनुकूल रहती हैं और किनसे परेशानी होती है?", "अपने अनुभव के अनुसार बताएं।", None),
  "satva": ("सत्त्व (मानसिक दृढ़ता)", "तनाव से निपटने और मानसिक रूप से संभलने की आपकी सामान्य क्षमता कैसी है?", "यह स्वयं बताई गई जानकारी है, निदान नहीं।", ["आम तौर पर संभाल लेता/लेती हूँ","कुछ तनाव के साथ ठीक रहता/रहती हूँ","अक्सर बहुत दबाव महसूस होता है","बताना नहीं चाहता/चाहती"]),
  "ahara_shakti": ("आहार शक्ति (भूख और पाचन)", "आपकी भूख और पाचन सामान्यतः कैसा रहता है?", "अपने सामान्य अनुभव के अनुसार चुनें।", ["अच्छा और नियमित","बदलता रहता है","भूख कम","भारी / धीमा पाचन","पता नहीं"]),
  "vyayama_shakti": ("व्यायाम शक्ति (शारीरिक क्षमता)", "आप अपनी सामान्य दिनचर्या में कितनी शारीरिक गतिविधि आराम से कर सकते हैं?", "अपने सामान्य दिन की क्षमता बताएं।", ["अधिक","मध्यम","कम","बहुत कम / जल्दी थकता हूँ","पता नहीं"]),
  "vaya": ("वय (आयु / जीवन अवस्था)", "आपकी आयु क्या है और यदि चाहें तो कौन-सी जीवन अवस्था आपको सबसे अधिक उपयुक्त लगती है?", "अपनी आयु लिखें।", None),
 },
 "bn-IN": {
  "prakriti": ("প্রকৃতি (constitution)", "আপনার শরীর ও স্বভাবের সাধারণ ধরন কেমন? জানা থাকলে আপনার প্রধান প্রকৃতি বলুন (বাত, পিত্ত, কফ বা মিশ্রণ)।", "না জানলে ‘জানি না’ নির্বাচন করুন।", ["বাত","পিত্ত","কফ","বাত-পিত্ত","পিত্ত-কফ","বাত-কফ","ত্রিদোষিক / ভারসাম্যপূর্ণ","জানি না"]),
  "vikriti": ("বিকৃতি (বর্তমান পরিবর্তন)", "আপনার স্বাভাবিক অবস্থার তুলনায় সম্প্রতি কী পরিবর্তন হয়েছে? যে ধরনগুলো অনুভব করেন সেগুলো নির্বাচন করুন।", "যা প্রযোজ্য সেগুলো নির্বাচন করুন।", ["শুষ্কতা / অনিয়ম","গরম / অম্লতা","ভারীভাব / congestion","ঘুমের পরিবর্তন","ক্ষুধার পরিবর্তন","পায়খানার পরিবর্তন","স্ট্রেস / অস্থিরতা","বড় কোনো পরিবর্তন নেই","জানি না"]),
  "sara": ("সার (টিস্যুর গুণমান)", "যতদূর জানেন, আপনার শরীরের সাধারণ শক্তি ও পুষ্টির অবস্থা কেমন?", "এটি আপনার নিজের দেওয়া তথ্য।", ["শক্তিশালী / ভালো পুষ্টি","স্বাভাবিক","কম / সহজে ক্লান্ত","জানি না"]),
  "samhanana": ("সংহনন (শরীরের গঠন)", "আপনার সামগ্রিক শরীরের গঠন ও শারীরিক বিকাশ কেমন বলে মনে হয়?", "সবচেয়ে কাছের বিকল্পটি বেছে নিন।", ["সুসম / শক্তিশালী","মাঝারি গঠন","পাতলা / নাজুক","ভারী / প্রশস্ত","জানি না"]),
  "pramana": ("প্রমাণ (শারীরিক মাপ)", "আপনার কি সাম্প্রতিক উচ্চতা, ওজন, কোমরের মাপ বা অন্য কোনো শারীরিক মাপ আছে যা নথিভুক্ত করতে চান?", "যে মাপগুলো জানেন লিখুন।", None),
  "satmya": ("সাত্ম্য (উপযোগিতা)", "কোন খাবার, দৈনন্দিন অভ্যাস, আবহাওয়া বা রুটিন আপনার জন্য ভালো, আর কোনগুলো অস্বস্তি তৈরি করে?", "নিজের অভিজ্ঞতা অনুযায়ী বলুন।", None),
  "satva": ("সত্ত্ব (মানসিক সহনশীলতা)", "স্ট্রেস সামলানো ও মানসিকভাবে স্থির থাকার আপনার সাধারণ ক্ষমতা কেমন?", "এটি স্ব-প্রতিবেদিত তথ্য, মানসিক রোগ নির্ণয় নয়।", ["সাধারণত সামলে নিতে পারি","কিছু স্ট্রেসে মোটামুটি ঠিক","প্রায়ই খুব চাপ অনুভব করি","বলতে চাই না"]),
  "ahara_shakti": ("আহার শক্তি (ক্ষুধা ও হজম)", "আপনার ক্ষুধা ও হজম সাধারণত কেমন?", "আপনার সাধারণ অভিজ্ঞতা অনুযায়ী বেছে নিন।", ["ভালো ও নিয়মিত","পরিবর্তনশীল","ক্ষুধা কম","ভারী / ধীর হজম","জানি না"]),
  "vyayama_shakti": ("ব্যায়াম শক্তি (শারীরিক সক্ষমতা)", "আপনার সাধারণ দিনে কতটা শারীরিক কাজ আরাম করে করতে পারেন?", "সাধারণ দিনের সক্ষমতা অনুযায়ী বলুন।", ["বেশি","মাঝারি","কম","খুব কম / সহজে ক্লান্ত","জানি না"]),
  "vaya": ("বয় (বয়স / জীবনপর্যায়)", "আপনার বয়স কত? চাইলে কোন জীবনপর্যায় আপনার সঙ্গে সবচেয়ে বেশি মেলে তাও বলুন।", "আপনার বয়স লিখুন।", None),
 },
 "gu-IN": {
  "prakriti": ("પ્રકૃતિ (constitution)", "તમારા શરીર અને સ્વભાવનો સામાન્ય પ્રકાર કેવો છે? ખબર હોય તો તમારી મુખ્ય પ્રકૃતિ જણાવો (વાત, પિત્ત, કફ અથવા મિશ્રણ).", "ખબર ન હોય તો ‘ખબર નથી’ પસંદ કરો.", ["વાત","પિત્ત","કફ","વાત-પિત્ત","પિત્ત-કફ","વાત-કફ","ત્રિદોષિક / સંતુલિત","ખબર નથી"]),
  "vikriti": ("વિકૃતિ (હાલના ફેરફાર)", "તમારી સામાન્ય સ્થિતિની સરખામણીમાં તાજેતરમાં શું બદલાયું છે? જે બાબતો અનુભવાય તે પસંદ કરો.", "લાગુ પડતી બધી બાબતો પસંદ કરો.", ["સૂકાપણું / અનિયમિતતા","ગરમી / એસિડિટી","ભારેપણું / congestion","ઊંઘમાં ફેરફાર","ભૂખમાં ફેરફાર","મળમાં ફેરફાર","તણાવ / બેચેની","મોટો ફેરફાર નથી","ખબર નથી"]),
  "sara": ("સાર (ટિશ્યૂ ગુણવત્તા)", "જેટલી તમને જાણ હોય, તમારા શરીરની સામાન્ય તાકાત અને પોષણની સ્થિતિ કેવી છે?", "આ તમારી પોતાની જણાવેલી માહિતી છે.", ["મજબૂત / સારી રીતે પોષાયેલ","સામાન્ય","ઓછી / સરળતાથી થાક","ખબર નથી"]),
  "samhanana": ("સંહનન (શારીરિક બંધારણ)", "તમારા શરીરના સામાન્ય બંધારણ અને વિકાસને તમે કેવી રીતે વર્ણવશો?", "સૌથી નજીકનો વિકલ્પ પસંદ કરો.", ["સુસંગત / મજબૂત","સરેરાશ બંધારણ","પાતળું / નાજુક","ભારે / પહોળું","ખબર નથી"]),
  "pramana": ("પ્રમાણ (શારીરિક માપ)", "શું તમારી પાસે તાજેતરની ઊંચાઈ, વજન, કમરનું માપ અથવા અન્ય શારીરિક માપ છે જે નોંધવા માંગો છો?", "જાણતા હો તે માપ લખો.", None),
  "satmya": ("સાત્મ્ય (અનુકૂળતા)", "કયા ખોરાક, દિનચર્યા, હવામાન અથવા આદતો તમને સામાન્ય રીતે અનુકૂળ આવે છે અને કઈ અસ્વસ્થતા કરે છે?", "તમારા અનુભવ પ્રમાણે જણાવો.", None),
  "satva": ("સત્વ (માનસિક સ્થિરતા)", "તણાવ સંભાળવાની અને માનસિક રીતે સ્થિર રહેવાની તમારી સામાન્ય ક્ષમતા કેવી છે?", "આ સ્વ-રિપોર્ટેડ માહિતી છે, નિદાન નથી.", ["સામાન્ય રીતે સારી રીતે સંભાળી લઉં છું","થોડા તણાવ સાથે ઠીક","ઘણી વાર ખૂબ દબાણ લાગે છે","કહેવું નથી"]),
  "ahara_shakti": ("આહાર શક્તિ (ભૂખ અને પાચન)", "તમારી ભૂખ અને પાચન સામાન્ય રીતે કેવું રહે છે?", "તમારા સામાન્ય અનુભવ પ્રમાણે પસંદ કરો.", ["સારું અને નિયમિત","બદલાતું રહે છે","ભૂખ ઓછી","ભારે / ધીમું પાચન","ખબર નથી"]),
  "vyayama_shakti": ("વ્યાયામ શક્તિ (શારીરિક ક્ષમતા)", "તમારી સામાન્ય દિનચર્યામાં તમે કેટલી શારીરિક પ્રવૃત્તિ આરામથી કરી શકો છો?", "તમારા સામાન્ય દિવસની ક્ષમતા જણાવો.", ["વધુ","મધ્યમ","ઓછી","ખૂબ ઓછી / સરળતાથી થાકી જાઉં છું","ખબર નથી"]),
  "vaya": ("વય (ઉંમર / જીવન તબક્કો)", "તમારી ઉંમર કેટલી છે? ઇચ્છો તો કયો જીવન તબક્કો તમને સૌથી વધુ યોગ્ય લાગે છે તે પણ કહો.", "તમારી ઉંમર લખો.", None),
 },
}

# Add compact, readable translations for the remaining Indian languages. Question text is
# localized while clinical field IDs remain stable. Options are intentionally kept simple
# where a full translation would reduce clarity for the prototype.
AYUSH_TRANSLATIONS.update({
 "ta-IN": {k:(v[0],v[1],v[2],v[3]) for k,v in {
  "prakriti":("பிரகிருதி (constitution)","உங்கள் உடல் மற்றும் இயல்பின் பொதுவான தன்மை எப்படி உள்ளது? தெரிந்தால் உங்கள் முக்கிய பிரகிருதியை கூறுங்கள் (வாதம், பித்தம், கபம் அல்லது கலவை).","தெரியாவிட்டால் ‘தெரியாது’ என்பதைத் தேர்வு செய்யுங்கள்.",["வாதம்","பித்தம்","கபம்","வாத-பித்தம்","பித்த-கபம்","வாத-கபம்","மூன்று தோஷ சமநிலை","தெரியாது"]),
  "vikriti":("விகிருதி (தற்போதைய மாற்றங்கள்)","உங்கள் வழக்கமான நிலையை ஒப்பிடும்போது சமீபத்தில் என்ன மாற்றம் ஏற்பட்டுள்ளது? நீங்கள் உணரும் அம்சங்களைத் தேர்வு செய்யுங்கள்.","பொருந்துவதைத் தேர்வு செய்யுங்கள்.",["வறட்சி / ஒழுங்கின்மை","சூடு / அமிலத்தன்மை","கனத்தன்மை / அடைப்பு","தூக்க மாற்றம்","பசியின் மாற்றம்","மல மாற்றம்","மன அழுத்தம் / அமைதியின்மை","முக்கிய மாற்றம் இல்லை","தெரியாது"]),
  "sara":("சார (திசு தரம்)","உங்கள் உடலின் பொதுவான வலிமை மற்றும் ஊட்டநிலை எப்படி உள்ளது?","இது நீங்கள் கூறும் தகவல் மட்டுமே.",["வலிமையான / நல்ல ஊட்டம்","சராசரி","குறைவு / எளிதில் சோர்வு","தெரியாது"]),
  "samhanana":("சம்ஹனன (உடல் அமைப்பு)","உங்கள் உடல் அமைப்பு மற்றும் வளர்ச்சியை எப்படி விவரிப்பீர்கள்?","மிகவும் பொருந்தும் விருப்பத்தைத் தேர்வு செய்யுங்கள்.",["சீரான / வலிமையான","சராசரி அமைப்பு","மெலிந்த / நுட்பமான","கனமான / அகலமான","தெரியாது"]),
  "pramana":("பிரமாண (உடல் அளவுகள்)","சமீபத்திய உயரம், எடை, இடுப்பு அளவு அல்லது வேறு உடல் அளவுகள் உள்ளனவா?","தெரிந்த அளவுகளை எழுதுங்கள்.",None),
  "satmya":("சாத்ம்ய (ஒத்துப்போகும் தன்மை)","எந்த உணவு, பழக்கம், காலநிலை அல்லது தினசரி நடைமுறை உங்களுக்கு ஏற்றது? எவை அசௌகரியம் தருகின்றன?","உங்கள் அனுபவத்தைப் பகிருங்கள்.",None),
  "satva":("சத்த்வ (மன உறுதி)","மன அழுத்தத்தை சமாளிக்கும் மற்றும் மனநிலையை சமநிலையில் வைத்திருக்கும் உங்கள் வழக்கமான திறன் எப்படி?","இது சுயமாக கூறிய தகவல்; நோயறிதல் அல்ல.",["பொதுவாக சமாளிக்க முடியும்","சில அழுத்தத்துடன் பரவாயில்லை","அடிக்கடி அதிக அழுத்தம்","சொல்ல விரும்பவில்லை"]),
  "ahara_shakti":("ஆஹார சக்தி (பசி / செரிமானம்)","உங்கள் பசி மற்றும் செரிமானம் பொதுவாக எப்படி உள்ளது?","உங்கள் வழக்கமான அனுபவத்தைத் தேர்வு செய்யுங்கள்.",["நல்லது மற்றும் ஒழுங்கானது","மாறுபடும்","பசி குறைவு","கனமான / மெதுவான செரிமானம்","தெரியாது"]),
  "vyayama_shakti":("வ்யாயாம சக்தி (உடல் திறன்)","உங்கள் வழக்கமான நாளில் எவ்வளவு உடல் செயல்பாட்டை வசதியாக செய்ய முடியும்?","உங்கள் வழக்கமான நாளை அடிப்படையாகக் கொள்ளுங்கள்.",["அதிகம்","மிதமானது","குறைவு","மிகக் குறைவு / எளிதில் சோர்வு","தெரியாது"]),
  "vaya":("வய (வயது / வாழ்க்கை நிலை)","உங்கள் வயது என்ன? விரும்பினால் எந்த வாழ்க்கை நிலையுடன் நீங்கள் அதிகம் பொருந்துகிறீர்கள் என்றும் கூறலாம்.","உங்கள் வயதை எழுதுங்கள்.",None),
 }.items()},
 "te-IN": {k:(v[0],v[1],v[2],v[3]) for k,v in {
  "prakriti":("ప్రకృతి (constitution)","మీ శరీరం మరియు స్వభావం సాధారణంగా ఎలా ఉంటాయి? తెలిసి ఉంటే మీ ప్రధాన ప్రకృతిని చెప్పండి (వాత, పిత్త, కఫ లేదా కలయిక).","తెలియకపోతే ‘తెలియదు’ ఎంచుకోండి.",["వాత","పిత్త","కఫ","వాత-పిత్త","పిత్త-కఫ","వాత-కఫ","త్రిదోష సమతుల్యం","తెలియదు"]),
  "vikriti":("వికృతి (ప్రస్తుత మార్పులు)","మీ సాధారణ స్థితితో పోలిస్తే ఇటీవల ఏమి మారింది? మీరు గమనించిన వాటిని ఎంచుకోండి.","వర్తించేవాటిని ఎంచుకోండి.",["పొడిబారడం / అసమానత","వేడి / ఆమ్లత్వం","భారంగా / congestion","నిద్రలో మార్పు","ఆకలిలో మార్పు","మల విసర్జనలో మార్పు","ఒత్తిడి / అస్థిరత","పెద్ద మార్పు లేదు","తెలియదు"]),
  "sara":("సార (కణజాల నాణ్యత)","మీకు తెలిసినంతవరకు, మీ శరీర సాధారణ బలం మరియు పోషక స్థితి ఎలా ఉంది?","ఇది మీరు చెప్పిన సమాచారం మాత్రమే.",["బలంగా / మంచి పోషణ","సగటు","తక్కువ / త్వరగా అలసిపోతాను","తెలియదు"]),
  "samhanana":("సంహనన (శరీర నిర్మాణం)","మీ శరీర నిర్మాణం మరియు శారీరక అభివృద్ధిని ఎలా వివరిస్తారు?","దగ్గరగా ఉన్న ఎంపికను ఎంచుకోండి.",["సమతుల్య / బలమైన","సగటు నిర్మాణం","సన్నగా / సున్నితంగా","భారీ / వెడల్పుగా","తెలియదు"]),
  "pramana":("ప్రమాణ (శరీర కొలతలు)","ఇటీవలి ఎత్తు, బరువు, నడుము కొలత లేదా ఇతర శరీర కొలతలు నమోదు చేయాలనుకుంటున్నారా?","తెలిసిన కొలతలను రాయండి.",None),
  "satmya":("సాత్మ్య (అనుకూలత)","ఏ ఆహారం, అలవాట్లు, వాతావరణం లేదా దినచర్య మీకు బాగా సరిపోతాయి? ఏవి అసౌకర్యం కలిగిస్తాయి?","మీ అనుభవం ప్రకారం చెప్పండి.",None),
  "satva":("సత్వ (మానసిక స్థైర్యం)","ఒత్తిడిని ఎదుర్కోవడం మరియు మానసికంగా స్థిరంగా ఉండటం మీ సాధారణ సామర్థ్యం ఎలా ఉంది?","ఇది స్వయంగా చెప్పిన సమాచారం; నిర్ధారణ కాదు.",["సాధారణంగా ఎదుర్కోగలను","కొంత ఒత్తిడితో బాగానే ఉంటాను","తరచుగా ఎక్కువ ఒత్తిడి","చెప్పదలుచుకోలేదు"]),
  "ahara_shakti":("ఆహార శక్తి (ఆకలి / జీర్ణం)","మీ ఆకలి మరియు జీర్ణక్రియ సాధారణంగా ఎలా ఉంటాయి?","మీ సాధారణ అనుభవం ప్రకారం ఎంచుకోండి.",["మంచిది మరియు క్రమబద్ధం","మారుతూ ఉంటుంది","ఆకలి తక్కువ","భారం / నెమ్మదిగా జీర్ణం","తెలియదు"]),
  "vyayama_shakti":("వ్యాయామ శక్తి (శారీరక సామర్థ్యం)","మీ సాధారణ రోజులో ఎంత శారీరక కార్యకలాపం సౌకర్యంగా చేయగలరు?","మీ సాధారణ రోజు సామర్థ్యాన్ని చెప్పండి.",["ఎక్కువ","మధ్యస్థం","తక్కువ","చాలా తక్కువ / త్వరగా అలసిపోతాను","తెలియదు"]),
  "vaya":("వయ (వయస్సు / జీవన దశ)","మీ వయస్సు ఎంత? కావాలంటే మీకు సరిపోయే జీవన దశను కూడా చెప్పవచ్చు.","మీ వయస్సు రాయండి.",None),
 }.items()},
 "mr-IN": {k:(v[0],v[1],v[2],v[3]) for k,v in {
  "prakriti":("प्रकृती (constitution)","तुमच्या शरीराचा आणि स्वभावाचा सामान्य प्रकार कसा आहे? माहिती असल्यास तुमची प्रमुख प्रकृती सांगा (वात, पित्त, कफ किंवा मिश्र).","माहिती नसेल तर ‘माहित नाही’ निवडा.",["वात","पित्त","कफ","वात-पित्त","पित्त-कफ","वात-कफ","त्रिदोषिक / संतुलित","माहित नाही"]),
  "vikriti":("विकृती (सध्यातील बदल)","तुमच्या नेहमीच्या स्थितीच्या तुलनेत अलीकडे काय बदलले आहे? जाणवणारे बदल निवडा.","लागू असलेले पर्याय निवडा.",["कोरडेपणा / अनियमितता","उष्णता / आम्लता","जडपणा / कोंडी","झोपेतील बदल","भुकेतील बदल","शौचातील बदल","ताण / अस्वस्थता","मोठा बदल नाही","माहित नाही"]),
  "sara":("सार (ऊतक गुणवत्ता)","तुम्हाला माहिती आहे त्यानुसार शरीराची सामान्य ताकद आणि पोषणाची स्थिती कशी आहे?","ही तुमची स्वतःची माहिती आहे.",["मजबूत / चांगले पोषण","सामान्य","कमी / लवकर थकतो","माहित नाही"]),
  "samhanana":("संहनन (शरीराची बांधणी)","तुमची एकूण शरीरबांधणी आणि शारीरिक विकास कसा आहे असे तुम्ही सांगाल?","सर्वात जवळचा पर्याय निवडा.",["सुदृढ / मजबूत","सरासरी बांधणी","बारीक / नाजूक","जड / रुंद","माहित नाही"]),
  "pramana":("प्रमाण (शारीरिक मोजमाप)","तुमच्याकडे अलीकडील उंची, वजन, कंबर किंवा इतर शारीरिक मोजमाप आहेत का?","माहित असलेली मोजमापे लिहा.",None),
  "satmya":("सात्म्य (अनुकूलता)","कोणते अन्न, दिनचर्या, हवामान किंवा सवयी तुम्हाला अनुकूल असतात आणि कोणत्यामुळे त्रास होतो?","तुमच्या अनुभवाप्रमाणे सांगा.",None),
  "satva":("सत्त्व (मानसिक स्थैर्य)","ताण हाताळण्याची आणि मानसिकदृष्ट्या स्थिर राहण्याची तुमची सामान्य क्षमता कशी आहे?","ही स्वतः सांगितलेली माहिती आहे; निदान नाही.",["सामान्यतः सांभाळू शकतो","काही ताण असूनही ठीक","अनेकदा खूप ताण जाणवतो","सांगू इच्छित नाही"]),
  "ahara_shakti":("आहार शक्ती (भूक / पचन)","तुमची भूक आणि पचन साधारणपणे कसे असते?","तुमच्या नेहमीच्या अनुभवाप्रमाणे निवडा.",["चांगले आणि नियमित","बदलते","भूक कमी","जड / मंद पचन","माहित नाही"]),
  "vyayama_shakti":("व्यायाम शक्ती (शारीरिक क्षमता)","तुमच्या नेहमीच्या दिवसात किती शारीरिक हालचाल आरामात करू शकता?","तुमच्या नेहमीच्या क्षमतेनुसार सांगा.",["जास्त","मध्यम","कमी","खूप कमी / लवकर थकतो","माहित नाही"]),
  "vaya":("वय (वय / जीवन टप्पा)","तुमचे वय किती आहे? इच्छित असल्यास कोणता जीवन टप्पा तुमच्याशी जास्त जुळतो ते सांगा.","तुमचे वय लिहा.",None),
 }.items()},
 "kn-IN": {k:(v[0],v[1],v[2],v[3]) for k,v in {
  "prakriti":("ಪ್ರಕೃತಿ (constitution)","ನಿಮ್ಮ ದೇಹ ಮತ್ತು ಸ್ವಭಾವದ ಸಾಮಾನ್ಯ ರೀತಿಯು ಹೇಗಿದೆ? ತಿಳಿದಿದ್ದರೆ ನಿಮ್ಮ ಪ್ರಮುಖ ಪ್ರಕೃತಿಯನ್ನು ಹೇಳಿ (ವಾತ, ಪಿತ್ತ, ಕಫ ಅಥವಾ ಮಿಶ್ರಣ).","ತಿಳಿದಿಲ್ಲದಿದ್ದರೆ ‘ತಿಳಿದಿಲ್ಲ’ ಆಯ್ಕೆಮಾಡಿ.",["ವಾತ","ಪಿತ್ತ","ಕಫ","ವಾತ-ಪಿತ್ತ","ಪಿತ್ತ-ಕಫ","ವಾತ-ಕಫ","ತ್ರಿದೋಷ / ಸಮತೋಲನ","ತಿಳಿದಿಲ್ಲ"]),
  "vikriti":("ವಿಕೃತಿ (ಪ್ರಸ್ತುತ ಬದಲಾವಣೆ)","ನಿಮ್ಮ ಸಾಮಾನ್ಯ ಸ್ಥಿತಿಗೆ ಹೋಲಿಸಿದರೆ ಇತ್ತೀಚೆಗೆ ಏನು ಬದಲಾಗಿದೆ? ನೀವು ಗಮನಿಸಿದವುಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ.","ಅನ್ವಯಿಸುವವುಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ.",["ಒಣತನ / ಅಸಮರ್ಪಕತೆ","ಬಿಸಿ / ಆಮ್ಲತೆ","ಭಾರ / congestion","ನಿದ್ರೆಯಲ್ಲಿ ಬದಲಾವಣೆ","ಹಸಿವಿನಲ್ಲಿ ಬದಲಾವಣೆ","ಮಲದಲ್ಲಿ ಬದಲಾವಣೆ","ಒತ್ತಡ / ಅಶಾಂತಿ","ದೊಡ್ಡ ಬದಲಾವಣೆ ಇಲ್ಲ","ತಿಳಿದಿಲ್ಲ"]),
  "sara":("ಸಾರ (ಕಣಜ ಗುಣಮಟ್ಟ)","ನಿಮಗೆ ತಿಳಿದಿರುವ ಮಟ್ಟಿಗೆ ನಿಮ್ಮ ದೇಹದ ಸಾಮಾನ್ಯ ಬಲ ಮತ್ತು ಪೋಷಣೆಯ ಸ್ಥಿತಿ ಹೇಗಿದೆ?","ಇದು ನೀವು ನೀಡಿದ ಮಾಹಿತಿ ಮಾತ್ರ.",["ಬಲವಾದ / ಉತ್ತಮ ಪೋಷಣೆ","ಸರಾಸರಿ","ಕಡಿಮೆ / ಬೇಗ ದಣಿವು","ತಿಳಿದಿಲ್ಲ"]),
  "samhanana":("ಸಂಹನನ (ದೇಹದ ಕಟ್ಟಳೆ)","ನಿಮ್ಮ ದೇಹದ ಒಟ್ಟಾರೆ ಕಟ್ಟಳೆ ಮತ್ತು ದೈಹಿಕ ಬೆಳವಣಿಗೆಯನ್ನು ಹೇಗೆ ವಿವರಿಸುತ್ತೀರಿ?","ಹತ್ತಿರವಾದ ಆಯ್ಕೆಯನ್ನು ಆರಿಸಿ.",["ಸಮಪ್ರಮಾಣ / ಬಲವಾದ","ಸರಾಸರಿ ಕಟ್ಟಳೆ","ಸಣ್ಣ / ಸೂಕ್ಷ್ಮ","ಭಾರವಾದ / ಅಗಲವಾದ","ತಿಳಿದಿಲ್ಲ"]),
  "pramana":("ಪ್ರಮಾಣ (ದೇಹದ ಅಳತೆ)","ಇತ್ತೀಚಿನ ಎತ್ತರ, ತೂಕ, ಸೊಂಟದ ಅಳತೆ ಅಥವಾ ಇತರ ದೇಹದ ಅಳತೆಗಳನ್ನು ದಾಖಲಿಸಲು ಬಯಸುವಿರಾ?","ತಿಳಿದಿರುವ ಅಳತೆಗಳನ್ನು ಬರೆಯಿರಿ.",None),
  "satmya":("ಸಾತ್ಮ್ಯ (ಹೊಂದಾಣಿಕೆ)","ಯಾವ ಆಹಾರ, ದಿನಚರಿ, ಹವಾಮಾನ ಅಥವಾ ಅಭ್ಯಾಸಗಳು ನಿಮಗೆ ಸಾಮಾನ್ಯವಾಗಿ ಹೊಂದಿಕೊಳ್ಳುತ್ತವೆ ಮತ್ತು ಯಾವವು ಅಸೌಕರ್ಯ ಉಂಟುಮಾಡುತ್ತವೆ?","ನಿಮ್ಮ ಅನುಭವದಂತೆ ಹೇಳಿ.",None),
  "satva":("ಸತ್ವ (ಮಾನಸಿಕ ಸ್ಥೈರ್ಯ)","ಒತ್ತಡವನ್ನು ನಿಭಾಯಿಸುವ ಮತ್ತು ಮಾನಸಿಕವಾಗಿ ಸ್ಥಿರವಾಗಿರುವ ನಿಮ್ಮ ಸಾಮಾನ್ಯ ಸಾಮರ್ಥ್ಯ ಹೇಗಿದೆ?","ಇದು ಸ್ವಯಂ ವರದಿ; ರೋಗನಿರ್ಣಯವಲ್ಲ.",["ಸಾಮಾನ್ಯವಾಗಿ ನಿಭಾಯಿಸಬಲ್ಲೆ","ಸ್ವಲ್ಪ ಒತ್ತಡದೊಂದಿಗೆ ಸರಿ","ಆಗಾಗ್ಗೆ ತುಂಬಾ ಒತ್ತಡ","ಹೇಳಲು ಇಷ್ಟವಿಲ್ಲ"]),
  "ahara_shakti":("ಆಹಾರ ಶಕ್ತಿ (ಹಸಿವು / ಜೀರ್ಣಕ್ರಿಯೆ)","ನಿಮ್ಮ ಹಸಿವು ಮತ್ತು ಜೀರ್ಣಕ್ರಿಯೆ ಸಾಮಾನ್ಯವಾಗಿ ಹೇಗಿರುತ್ತದೆ?","ನಿಮ್ಮ ಸಾಮಾನ್ಯ ಅನುಭವದಂತೆ ಆಯ್ಕೆಮಾಡಿ.",["ಉತ್ತಮ ಮತ್ತು ನಿಯಮಿತ","ಬದಲಾಗುತ್ತದೆ","ಹಸಿವು ಕಡಿಮೆ","ಭಾರ / ನಿಧಾನ ಜೀರ್ಣಕ್ರಿಯೆ","ತಿಳಿದಿಲ್ಲ"]),
  "vyayama_shakti":("ವ್ಯಾಯಾಮ ಶಕ್ತಿ (ದೈಹಿಕ ಸಾಮರ್ಥ್ಯ)","ನಿಮ್ಮ ಸಾಮಾನ್ಯ ದಿನದಲ್ಲಿ ಎಷ್ಟು ದೈಹಿಕ ಚಟುವಟಿಕೆಯನ್ನು ಆರಾಮವಾಗಿ ಮಾಡಬಹುದು?","ನಿಮ್ಮ ಸಾಮಾನ್ಯ ದಿನದ ಸಾಮರ್ಥ್ಯವನ್ನು ಹೇಳಿ.",["ಹೆಚ್ಚು","ಮಧ್ಯಮ","ಕಡಿಮೆ","ತುಂಬಾ ಕಡಿಮೆ / ಬೇಗ ದಣಿವು","ತಿಳಿದಿಲ್ಲ"]),
  "vaya":("ವಯ (ವಯಸ್ಸು / ಜೀವನ ಹಂತ)","ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು? ಬಯಸಿದರೆ ನಿಮಗೆ ಹೊಂದುವ ಜೀವನ ಹಂತವನ್ನೂ ಹೇಳಬಹುದು.","ನಿಮ್ಮ ವಯಸ್ಸನ್ನು ಬರೆಯಿರಿ.",None),
 }.items()},
})

def localized_ayush_questions(language):
    lang = language if language in AYUSH_TRANSLATIONS else "en-IN"
    translations = AYUSH_TRANSLATIONS.get(lang, {})
    out=[]
    for q in AYUSH_DASHAVIDHA:
        x=dict(q)
        if q["id"] in translations:
            title,text,hint,options=translations[q["id"]]
            x.update(title=title, question=text, hint=hint)
            if options is not None: x["options"]=options
        out.append(x)
    return out

class AyushAnswer(BaseModel):
    patient_id: int
    session_id: str
    question_id: str
    answer: str
    answers: dict = Field(default_factory=dict)
    language: str = "en-IN"

class AyushComplete(BaseModel):
    patient_id: int
    session_id: str
    responses: dict
    language: str = "en-IN"

@app.get("/api/ayush/questions")
def ayush_questions(language: str = "en-IN"):
    intro = "यह आयुष/आयुर्वेद मॉड्यूल दशविध परीक्षा से संबंधित जानकारी दर्ज करता है। यह स्वयं निदान या दोष निर्धारण नहीं करता।" if language == "hi-IN" else AYUSH_INTRO
    return {"intro": intro, "system":"Ayurveda", "assessment_type":"Dashavidha Pariksha", "questions": localized_ayush_questions(language), "language":language}

@app.post("/api/ayush/answer")
def ayush_answer(payload: AyushAnswer):
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Please provide an answer.")
    answered=dict(payload.answers or {})
    answered[payload.question_id]=payload.answer.strip()
    ids=[q["id"] for q in AYUSH_DASHAVIDHA]
    idx=ids.index(payload.question_id) if payload.question_id in ids else -1
    next_q=AYUSH_DASHAVIDHA[idx+1] if idx>=0 and idx+1<len(AYUSH_DASHAVIDHA) else None
    return {"message":"AYUSH response received","next_question":(localized_ayush_questions(payload.language)[idx+1] if next_q is not None else None),"completed":next_q is None,"progress":round(len(answered)/len(ids)*100),"question_number":len(answered)+(0 if next_q is None else 1),"total_questions":len(ids),"responses":answered,"language":payload.language}

@app.post("/api/ayush/complete")
def ayush_complete(payload: AyushComplete):
    db=SessionLocal()
    try:
        patient=db.query(User).filter(User.id==payload.patient_id, User.role=="patient").first()
        if not patient: raise HTTPException(status_code=404, detail="Patient not found.")
        lines=[]
        for q in AYUSH_DASHAVIDHA:
            val=payload.responses.get(q["id"])
            if val not in (None, ""): lines.append(f'{q["title"]}: {val}')
        summary=" | ".join(lines) if lines else "AYUSH assessment completed."
        rec=AyushAssessment(patient_id=patient.id,responses=json.dumps(payload.responses,ensure_ascii=False),summary=summary)
        db.add(rec)
        db.commit(); db.refresh(rec)
        return {"message":"AYUSH assessment saved","assessment_id":rec.id,"summary":summary,"responses":payload.responses}
    finally: db.close()

@app.get("/api/patients/{patient_id}/ayush")
def patient_ayush(patient_id:int):
    db=SessionLocal()
    try:
        rows=db.query(AyushAssessment).filter(AyushAssessment.patient_id==patient_id).order_by(AyushAssessment.created_at.desc()).all()
        return {"assessments":[{"id":r.id,"system":r.system,"assessment_type":r.assessment_type,"summary":r.summary,"responses":json.loads(r.responses or "{}"),"created_at":r.created_at.isoformat()} for r in rows]}
    finally: db.close()

# ---------- Phase 5B: consent + identity + FHIR-ready interoperability ----------
CONSENT_SCOPE = "AI-assisted case taking, document processing, AYUSH intake, physician review"

@app.get("/api/patients/{patient_id}/consent")
def get_consent(patient_id: int):
    db = SessionLocal()
    try:
        rows = db.query(ConsentRecord).filter(ConsentRecord.patient_id == patient_id).order_by(ConsentRecord.created_at.desc()).all()
        active = next((r for r in rows if r.granted and r.revoked_at is None), None)
        return {"active": bool(active), "consent": ({"id": active.id, "type": active.consent_type, "version": active.version, "language": active.language, "audio_explained": bool(active.audio_explained), "scope": active.scope, "created_at": active.created_at.isoformat()} if active else None), "history": [{"id":r.id,"granted":bool(r.granted),"revoked":bool(r.revoked_at),"created_at":r.created_at.isoformat()} for r in rows]}
    finally:
        db.close()

@app.post("/api/patients/{patient_id}/consent")
def save_consent(patient_id: int, payload: ConsentRequest):
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient: raise HTTPException(status_code=404, detail="Patient not found.")
        current = db.query(ConsentRecord).filter(ConsentRecord.patient_id == patient_id, ConsentRecord.revoked_at.is_(None), ConsentRecord.granted == 1).all()
        for c in current: c.revoked_at = datetime.utcnow()
        rec = ConsentRecord(patient_id=patient_id, consent_type=payload.consent_type, language=payload.language, granted=1 if payload.granted else 0, audio_explained=1 if payload.audio_explained else 0, scope=CONSENT_SCOPE)
        db.add(rec); db.commit(); db.refresh(rec)
        return {"message":"Consent recorded","active":bool(rec.granted),"consent_id":rec.id,"version":rec.version}
    finally:
        db.close()

@app.post("/api/patients/{patient_id}/consent/revoke")
def revoke_consent(patient_id: int):
    db = SessionLocal()
    try:
        rows = db.query(ConsentRecord).filter(ConsentRecord.patient_id == patient_id, ConsentRecord.granted == 1, ConsentRecord.revoked_at.is_(None)).all()
        for r in rows: r.revoked_at = datetime.utcnow()
        db.commit()
        return {"message":"Consent revoked","active":False}
    finally:
        db.close()

def fhir_reference(resource_type, resource_id):
    return {"reference": f"{resource_type}/{resource_id}"}

def build_fhir_bundle(db, patient_id: int):
    patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not patient: raise HTTPException(status_code=404, detail="Patient not found.")
    profile = patient.profile
    consultations = db.query(Consultation).filter(Consultation.patient_id == patient_id).order_by(Consultation.created_at.desc()).all()
    docs = db.query(MedicalDocument).filter(MedicalDocument.patient_id == patient_id).order_by(MedicalDocument.created_at.desc()).all()
    ayush = db.query(AyushAssessment).filter(AyushAssessment.patient_id == patient_id).order_by(AyushAssessment.created_at.desc()).all()
    entries=[]
    patient_resource={"resourceType":"Patient","id":f"sih-{patient.id}","identifier":[{"system":"https://sih26047.local/patient-id","value":str(patient.id)}],"name":[{"text":patient.name}],"gender":(profile.gender.lower() if profile and profile.gender and profile.gender.lower() in ["male","female","other","unknown"] else "unknown") }
    if profile and profile.age:
        patient_resource["extension"]=[{"url":"https://sih26047.local/fhir/age","valueInteger":profile.age}]
    entries.append({"fullUrl":f"urn:uuid:{patient_resource['id']}","resource":patient_resource})
    for c in consultations:
        entries.append({"fullUrl":f"urn:uuid:consultation-{c.id}","resource":{"resourceType":"DocumentReference","id":f"consultation-{c.id}","status":"current","subject":fhir_reference("Patient",patient_resource["id"]),"description":c.title,"date":c.created_at.isoformat(),"content":[{"attachment":{"contentType":"text/plain","title":"AI-assisted clinical history","data":c.summary.encode().hex()}}],"extension":[{"url":"https://sih26047.local/fhir/risk-level","valueString":c.risk_level or "none"},{"url":"https://sih26047.local/fhir/doctor-review","valueString":c.doctor_review or "Pending"}]}})
    for d in docs:
        entries.append({"fullUrl":f"urn:uuid:document-{d.id}","resource":{"resourceType":"DocumentReference","id":f"document-{d.id}","status":"current","subject":fhir_reference("Patient",patient_resource["id"]),"description":f"{d.document_type}: {d.filename}","date":d.created_at.isoformat(),"content":[{"attachment":{"title":d.filename,"contentType":d.mime_type or "application/octet-stream"}}]}})
    for a in ayush:
        entries.append({"fullUrl":f"urn:uuid:ayush-{a.id}","resource":{"resourceType":"QuestionnaireResponse","id":f"ayush-{a.id}","status":"completed","subject":fhir_reference("Patient",patient_resource["id"]),"authored":a.created_at.isoformat(),"questionnaire":"https://sih26047.local/fhir/Questionnaire/dashavidha-pariksha","item":[{"linkId":k,"text":k,"answer":[{"valueString":str(v)}]} for k,v in json.loads(a.responses or "{}").items()]}})
    return {"resourceType":"Bundle","id":f"sih26047-{patient.id}-{uuid.uuid4().hex[:10]}","type":"collection","timestamp":datetime.utcnow().isoformat(),"meta":{"profile":["http://hl7.org/fhir/StructureDefinition/Bundle"],"tag":[{"system":"https://sih26047.local","code":"SIH26047-PHASE5B","display":"Prototype FHIR-ready export"}]},"entry":entries}

@app.get("/api/doctor/patients/{patient_id}/fhir-preview")
def fhir_preview(patient_id: int):
    db=SessionLocal()
    try:
        return build_fhir_bundle(db, patient_id)
    finally: db.close()

@app.get("/api/doctor/patients/{patient_id}/abdm-readiness")
def abdm_readiness(patient_id: int):
    db=SessionLocal()
    try:
        patient=db.query(User).filter(User.id==patient_id, User.role=="patient").first()
        if not patient: raise HTTPException(status_code=404, detail="Patient not found.")
        consent=db.query(ConsentRecord).filter(ConsentRecord.patient_id==patient_id, ConsentRecord.granted==1, ConsentRecord.revoked_at.is_(None)).order_by(ConsentRecord.created_at.desc()).first()
        consultations=db.query(Consultation).filter(Consultation.patient_id==patient_id).count()
        docs=db.query(MedicalDocument).filter(MedicalDocument.patient_id==patient_id).count()
        ayush=db.query(AyushAssessment).filter(AyushAssessment.patient_id==patient_id).count()
        return {"patient_id":patient_id,"abha_status":"demo-not-linked","consent_active":bool(consent),"consent_id":consent.id if consent else None,"care_context_count":consultations+ayush,"document_count":docs,"facility_registry":"demo","hpr_registry":"demo","network_status":"sandbox-ready","production_connection":False,"checks":{"fhir_bundle":True,"consent_gate":bool(consent),"care_context":(consultations+ayush)>0,"abha_link":False,"hfr_link":False,"hpr_link":False}}
    finally: db.close()

def build_abdm_package(db, patient_id: int, abha_address: Optional[str] = None, facility_id: str = "SIH26047-DEMO-FACILITY", practitioner_id: str = "SIH26047-DEMO-HPR", consent_reference: Optional[str] = None):
    bundle=build_fhir_bundle(db, patient_id)
    patient=db.query(User).filter(User.id==patient_id, User.role=="patient").first()
    consent=db.query(ConsentRecord).filter(ConsentRecord.patient_id==patient_id, ConsentRecord.granted==1, ConsentRecord.revoked_at.is_(None)).order_by(ConsentRecord.created_at.desc()).first()
    if not consent: raise HTTPException(status_code=403, detail="Active patient consent is required for an ABDM package.")
    care_contexts=[]
    for e in bundle.get("entry",[]):
        r=e.get("resource",{})
        if r.get("resourceType") in ("DocumentReference","QuestionnaireResponse"):
            care_contexts.append({"reference":r.get("id"),"display":r.get("description") or r.get("resourceType")})
    package={"package_type":"ABDM sandbox-ready health record package","standard":"FHIR R4 / ABDM-aligned prototype","generated_at":datetime.utcnow().isoformat(),"abha":{"address":abha_address,"linked":bool(abha_address)},"patient":{"local_id":f"SIH26047-P-{patient.id:04d}","fhir_id":f"sih-{patient.id}"},"care_contexts":care_contexts,"consent":{"active":True,"record_id":consent.id,"reference":consent_reference or f"Consent/{consent.id}","version":consent.version},"facility":{"hfr_id":facility_id,"status":"demo"},"practitioner":{"hpr_id":practitioner_id,"status":"demo"},"exchange":{"mode":"sandbox-ready simulation","network_endpoint":None,"sent_to_abdm":False},"fhir_bundle":bundle,"limitations":["This prototype does not connect to the live ABDM network.","ABHA, HFR and HPR identifiers shown here are demo placeholders unless explicitly linked.","A production integration requires ABDM sandbox onboarding, authentication, consent workflows, security assessment and approved network endpoints."]}
    return package

@app.get("/api/doctor/patients/{patient_id}/abdm-package")
def abdm_package_preview(patient_id: int):
    db=SessionLocal()
    try: return build_abdm_package(db, patient_id)
    finally: db.close()

@app.post("/api/doctor/patients/{patient_id}/abdm-package")
def abdm_package_export(patient_id: int, payload: ABDMExportRequest):
    db=SessionLocal()
    try:
        package=build_abdm_package(db, patient_id, payload.abha_address, payload.facility_id, payload.practitioner_id, payload.consent_reference)
        rec=FHIRExportRecord(patient_id=patient_id, exported_by=payload.exported_by, resource_type="ABDM-Package", bundle_id=package["fhir_bundle"]["id"])
        db.add(rec); db.commit(); db.refresh(rec)
        return {"message":"ABDM sandbox-ready package generated","export_id":rec.id,"package":package}
    finally: db.close()

@app.post("/api/doctor/patients/{patient_id}/fhir-validate")
def validate_fhir(patient_id: int):
    db=SessionLocal()
    try:
        b=build_fhir_bundle(db, patient_id)
        errors=[]
        if b.get("resourceType")!="Bundle": errors.append("Bundle.resourceType must be Bundle")
        if b.get("type") not in ("collection","document","message"): errors.append("Bundle.type is missing or unsupported")
        for e in b.get("entry",[]):
            r=e.get("resource",{})
            if not r.get("resourceType"): errors.append("Entry resourceType is missing")
            if not r.get("id"): errors.append(f"{r.get('resourceType','Resource')} id is missing")
        return {"valid":not errors,"resource_count":len(b.get("entry",[])),"errors":errors,"profile":"FHIR R4 baseline / ABDM-aligned prototype","checked_at":datetime.utcnow().isoformat()}
    finally: db.close()

@app.post("/api/doctor/patients/{patient_id}/fhir-export")
def fhir_export(patient_id: int, payload: FHIRExportRequest):
    db=SessionLocal()
    try:
        bundle=build_fhir_bundle(db, patient_id)
        if payload.abha_address:
            bundle.setdefault("meta", {}).setdefault("tag", []).append({"system":"https://abdm.gov.in/abha-address","code":"linked-demo","display":payload.abha_address})
        rec=FHIRExportRecord(patient_id=patient_id, exported_by=payload.exported_by, resource_type="Bundle", bundle_id=bundle["id"])
        db.add(rec); db.commit(); db.refresh(rec)
        return {"message":"FHIR-ready bundle generated","export_id":rec.id,"bundle":bundle}
    finally: db.close()

# ---------- Phase 4A adaptive clinical interview engine ----------

COMMON_QUESTIONS = [
    {"id": "chief_complaint", "section": "Presenting complaint", "text": "What is the main problem or symptom that brought you here today? Please describe it in your own words."},
    {"id": "onset", "section": "History of present illness", "text": "When did this problem start? Did it begin suddenly or gradually?"},
    {"id": "location", "section": "History of present illness", "text": "Where exactly do you feel the problem or symptom? Does it spread anywhere else?"},
    {"id": "severity", "section": "History of present illness", "text": "How severe is it right now on a scale from 0 to 10, where 0 is no discomfort and 10 is the worst you can imagine?"},
    {"id": "character", "section": "History of present illness", "text": "How would you describe the symptom—for example sharp, dull, burning, throbbing, tight, heavy, or something else?"},
]

PATHWAYS = {
    "chest_pain": {
        "label": "Chest discomfort pathway",
        "keywords": ["chest pain", "chest discomfort", "chest pressure", "chest tightness", "pain in chest"],
        "questions": [
            {"id": "chest_radiation", "section": "Focused chest history", "text": "Does the discomfort spread to your arm, shoulder, jaw, back, or neck?"},
            {"id": "chest_exertion", "section": "Focused chest history", "text": "Does it get worse with walking, climbing stairs, or other physical activity? Does it improve with rest?"},
            {"id": "chest_breathlessness", "section": "Focused chest history", "text": "Are you having breathlessness, unusual sweating, dizziness, fainting, or a feeling that your heart is racing?"},
        ],
    },
    "headache": {
        "label": "Headache pathway",
        "keywords": ["headache", "head pain", "migraine", "pain in my head"],
        "questions": [
            {"id": "headache_visual", "section": "Focused headache history", "text": "Have you had blurred vision, flashing lights, weakness, numbness, difficulty speaking, or trouble walking with the headache?"},
            {"id": "headache_nausea", "section": "Focused headache history", "text": "Have you had nausea, vomiting, sensitivity to light or sound, or a similar headache before?"},
            {"id": "headache_trigger", "section": "Focused headache history", "text": "Did anything trigger or worsen the headache, such as exertion, coughing, lack of sleep, stress, or a recent injury?"},
        ],
    },
    "abdominal_pain": {
        "label": "Abdominal pain pathway",
        "keywords": ["stomach pain", "abdominal pain", "belly pain", "abdomen pain", "pain in my stomach"],
        "questions": [
            {"id": "abdominal_food", "section": "Focused abdominal history", "text": "Does the pain change after eating, passing stool, or taking any medicine?"},
            {"id": "abdominal_bowel", "section": "Focused abdominal history", "text": "Have you had vomiting, diarrhea, constipation, blood or black stool, or a change in your usual bowel habits?"},
            {"id": "abdominal_urinary", "section": "Focused abdominal history", "text": "Have you had pain or burning while urinating, blood in the urine, or pain going toward your back or groin?"},
        ],
    },
    "fever": {
        "label": "Fever pathway",
        "keywords": ["fever", "high temperature", "temperature", "feverish", "chills"],
        "questions": [
            {"id": "fever_pattern", "section": "Focused fever history", "text": "When you have checked your temperature, what was the highest reading? Does the fever come and go or stay present?"},
            {"id": "fever_infection", "section": "Focused fever history", "text": "Have you had cough, sore throat, breathing difficulty, vomiting, diarrhea, burning urine, rash, or any other signs of infection?"},
            {"id": "fever_exposure", "section": "Focused fever history", "text": "Have you recently travelled, eaten unusual food, been around someone who was ill, or had an insect or animal exposure?"},
        ],
    },
    "respiratory": {
        "label": "Breathing symptom pathway",
        "keywords": ["cough", "breathlessness", "shortness of breath", "difficulty breathing", "breathing problem", "wheezing"],
        "questions": [
            {"id": "respiratory_cough", "section": "Focused respiratory history", "text": "If you have a cough, is it dry or with phlegm? If there is phlegm, what colour is it? Have you noticed blood?"},
            {"id": "respiratory_activity", "section": "Focused respiratory history", "text": "Is the breathing difficulty worse with activity, when lying down, or at night? Does it happen suddenly or gradually?"},
            {"id": "respiratory_wheeze", "section": "Focused respiratory history", "text": "Have you had wheezing, chest tightness, fever, or known asthma or other lung problems?"},
        ],
    },
    "general": {
        "label": "General symptom pathway",
        "keywords": [],
        "questions": [
            {"id": "general_change", "section": "Focused symptom history", "text": "What makes the symptom better or worse? Have you tried anything for it, and did that help?"},
            {"id": "general_impact", "section": "Focused symptom history", "text": "How is this problem affecting your sleep, eating, work, movement, or normal daily activities?"},
        ],
    },
}

CLOSING_QUESTIONS = [
    {"id": "associated", "section": "Associated symptoms", "text": "Have you noticed any other symptoms along with this problem? Please mention anything that feels relevant."},
    {"id": "past_history", "section": "Past history", "text": "Do you have any existing medical conditions or have you had any important illnesses or surgeries in the past?"},
    {"id": "medications", "section": "Drug history", "text": "Are you currently taking any medicines, supplements, or other treatments?"},
    {"id": "allergies", "section": "Allergy history", "text": "Do you have any known medicine, food, or other allergies? If yes, what happens when you are exposed to them?"},
    {"id": "family_history", "section": "Family history", "text": "Does anyone in your close family have important medical conditions such as diabetes, high blood pressure, heart disease, asthma, or similar problems?"},
    {"id": "personal_history", "section": "Personal history", "text": "Is there anything about your daily habits that may be relevant, such as sleep, diet, exercise, tobacco, alcohol, occupation, or stress?"},
    {"id": "review_systems", "section": "Review of systems", "text": "Before we finish, have you recently had fever, unusual weight change, breathing difficulty, chest discomfort, vomiting, bowel or urinary changes, dizziness, or any other new symptom?"},
]



# ---------- Phase 4C.1 multilingual clinical interview layer ----------
SUPPORTED_LANGUAGES = {
    "en-IN": {"label": "English (India)", "short": "English"},
    "hi-IN": {"label": "हिन्दी (India)", "short": "हिन्दी"},
    "bn-IN": {"label": "বাংলা (India)", "short": "বাংলা"},
    "ta-IN": {"label": "தமிழ் (India)", "short": "தமிழ்"},
    "te-IN": {"label": "తెలుగు (India)", "short": "తెలుగు"},
    "mr-IN": {"label": "मराठी (India)", "short": "मराठी"},
    "gu-IN": {"label": "ગુજરાતી (India)", "short": "ગુજરાતી"},
    "kn-IN": {"label": "ಕನ್ನಡ (India)", "short": "ಕನ್ನಡ"},
}

# UI question translations. Clinical field IDs remain language-neutral so the
# physician-facing record stays standardized.
QUESTION_TRANSLATIONS = {
    "hi-IN": {
        "chief_complaint": "आज आपको यहां आने की मुख्य समस्या या तकलीफ़ क्या है? अपने शब्दों में बताएं।",
        "onset": "यह समस्या कब शुरू हुई? क्या यह अचानक शुरू हुई या धीरे-धीरे?",
        "location": "यह समस्या या तकलीफ़ आपको ठीक कहां महसूस होती है? क्या यह कहीं और फैलती है?",
        "severity": "अभी यह तकलीफ़ 0 से 10 में कितनी है, जहां 0 का मतलब कोई तकलीफ़ नहीं और 10 सबसे ज्यादा है?",
        "character": "आप इस तकलीफ़ को कैसे बताएंगे—जैसे तेज़, जलन, धड़कन जैसी, जकड़न, भारीपन या कुछ और?",
        "chest_radiation": "क्या यह तकलीफ़ हाथ, कंधे, जबड़े, पीठ या गर्दन तक फैलती है?",
        "chest_exertion": "क्या चलने, सीढ़ियां चढ़ने या मेहनत करने पर यह बढ़ती है? क्या आराम करने पर कम होती है?",
        "chest_breathlessness": "क्या सांस फूलना, असामान्य पसीना, चक्कर, बेहोशी या दिल तेजी से धड़कने जैसा महसूस हो रहा है?",
        "associated": "क्या इस समस्या के साथ कोई और लक्षण भी हैं? जो भी महत्वपूर्ण लगे, बताएं।",
        "past_history": "क्या आपको कोई पुरानी बीमारी है या पहले कोई महत्वपूर्ण बीमारी या ऑपरेशन हुआ है?",
        "medications": "क्या आप अभी कोई दवा, सप्लीमेंट या अन्य उपचार ले रहे हैं?",
        "allergies": "क्या आपको किसी दवा, भोजन या अन्य चीज़ से एलर्जी है? अगर हां, तो क्या होता है?",
        "family_history": "क्या आपके करीबी परिवार में मधुमेह, हाई ब्लड प्रेशर, हृदय रोग, अस्थमा या ऐसी कोई बीमारी है?",
        "personal_history": "आपकी रोज़मर्रा की आदतों में ऐसी कोई बात है जो महत्वपूर्ण हो सकती है—जैसे नींद, भोजन, व्यायाम, तंबाकू, शराब, काम या तनाव?",
        "review_systems": "अंत में, क्या हाल में बुखार, वजन में बदलाव, सांस लेने में तकलीफ़, सीने में परेशानी, उल्टी, मल या पेशाब में बदलाव, चक्कर या कोई नया लक्षण हुआ है?",
        "headache_visual": "क्या सिरदर्द के साथ धुंधला दिखना, चमकती रोशनी, कमजोरी, सुन्नपन, बोलने या चलने में परेशानी हुई है?",
        "headache_nausea": "क्या मितली, उल्टी, रोशनी या आवाज़ से परेशानी, या पहले ऐसा ही सिरदर्द हुआ है?",
        "headache_trigger": "क्या मेहनत, खांसी, नींद की कमी, तनाव या हाल की चोट से सिरदर्द शुरू या बढ़ा?",
        "abdominal_food": "क्या खाना खाने, मल त्यागने या कोई दवा लेने के बाद दर्द बदलता है?",
        "abdominal_bowel": "क्या उल्टी, दस्त, कब्ज, मल में खून या काला मल, या मल त्याग की आदत में बदलाव हुआ है?",
        "abdominal_urinary": "क्या पेशाब करते समय दर्द या जलन, पेशाब में खून, या दर्द पीठ या जांघ के बीच की ओर जाता है?",
        "fever_pattern": "तापमान नापने पर सबसे अधिक कितना आया? बुखार आता-जाता है या लगातार रहता है?",
        "fever_infection": "क्या खांसी, गले में दर्द, सांस लेने में परेशानी, उल्टी, दस्त, पेशाब में जलन, दाने या संक्रमण के अन्य लक्षण हैं?",
        "fever_exposure": "क्या हाल में यात्रा की, कोई असामान्य भोजन खाया, किसी बीमार व्यक्ति के संपर्क में आए या कीड़े/जानवर के संपर्क में आए?",
        "respiratory_cough": "अगर खांसी है, तो सूखी है या बलगम के साथ? बलगम हो तो उसका रंग क्या है? क्या उसमें खून दिखा?",
        "respiratory_activity": "सांस लेने में परेशानी मेहनत, लेटने या रात में बढ़ती है? यह अचानक होती है या धीरे-धीरे?",
        "respiratory_wheeze": "क्या घरघराहट, सीने में जकड़न, बुखार, अस्थमा या फेफड़ों की कोई बीमारी रही है?",
        "general_change": "इस लक्षण को क्या बेहतर या बदतर करता है? आपने इसके लिए कुछ किया या दवा ली? उससे फायदा हुआ?",
        "general_impact": "यह समस्या आपकी नींद, भोजन, काम, चलने-फिरने या रोज़मर्रा की गतिविधियों को कैसे प्रभावित कर रही है?",
    },
}

# Additional UI translations for the clinical interview. The backend keeps the IDs in
# English so analytics/doctor review remain language-neutral.
QUESTION_TRANSLATIONS.update({
 "bn-IN": {
  "chief_complaint":"আজ আপনাকে এখানে আসার প্রধান সমস্যা বা উপসর্গ কী? নিজের ভাষায় বলুন।","onset":"এই সমস্যা কবে শুরু হয়েছে? হঠাৎ নাকি ধীরে ধীরে শুরু হয়েছিল?","location":"সমস্যা বা উপসর্গটি ঠিক কোথায় অনুভব করেন? অন্য কোথাও ছড়িয়ে যায় কি?","severity":"এখন অস্বস্তি ০ থেকে ১০-এর মধ্যে কত, যেখানে ০ মানে নেই এবং ১০ সবচেয়ে বেশি?","character":"উপসর্গটি কেমন—তীক্ষ্ণ, মৃদু, জ্বালাপোড়া, ধুকপুক, চাপ বা অন্য কিছু?","associated":"এই সমস্যার সঙ্গে আর কোনো উপসর্গ হয়েছে কি? গুরুত্বপূর্ণ মনে হলে বলুন।","past_history":"আপনার কোনো পুরনো রোগ আছে, বা আগে গুরুত্বপূর্ণ অসুখ/অপারেশন হয়েছে কি?","medications":"আপনি কি এখন কোনো ওষুধ, সাপ্লিমেন্ট বা অন্য চিকিৎসা নিচ্ছেন?","allergies":"কোনো ওষুধ, খাবার বা অন্য কিছুর অ্যালার্জি আছে কি? থাকলে কী হয়?","family_history":"পরিবারের কাছের কারও ডায়াবেটিস, উচ্চ রক্তচাপ, হৃদরোগ, হাঁপানি বা অন্য গুরুত্বপূর্ণ রোগ আছে কি?","personal_history":"ঘুম, খাবার, ব্যায়াম, তামাক, অ্যালকোহল, কাজ বা স্ট্রেসের মতো দৈনন্দিন অভ্যাসে গুরুত্বপূর্ণ কিছু আছে কি?","review_systems":"শেষে, সম্প্রতি জ্বর, ওজনের পরিবর্তন, শ্বাসকষ্ট, বুকের অস্বস্তি, বমি, পায়খানা/প্রস্রাবে পরিবর্তন, মাথা ঘোরা বা অন্য নতুন উপসর্গ হয়েছে কি?"
 },
 "gu-IN": {
  "chief_complaint":"આજે અહીં આવવાનું મુખ્ય કારણ અથવા તકલીફ શું છે? તમારા પોતાના શબ્દોમાં જણાવો.","onset":"આ તકલીફ ક્યારે શરૂ થઈ? અચાનક કે ધીમે ધીમે?","location":"તકલીફ તમને ચોક્કસ ક્યાં થાય છે? તે બીજી જગ્યાએ ફેલાય છે?","severity":"હમણાં તકલીફ ૦ થી ૧૦માં કેટલી છે, જ્યાં ૦ એટલે કશું નહીં અને ૧૦ સૌથી વધારે?","character":"તકલીફ કેવી લાગે છે—તીક્ષ્ણ, ધીમી, બળતરા, ધબકારા, જકડાણ, ભારેપણું કે કંઈક બીજું?","associated":"આ તકલીફ સાથે બીજાં કોઈ લક્ષણો છે? મહત્વનું લાગે તે જણાવો.","past_history":"તમને કોઈ જૂની બીમારી છે અથવા પહેલાં કોઈ ગંભીર બીમારી/ઓપરેશન થયું છે?","medications":"તમે હાલમાં કોઈ દવા, સપ્લિમેન્ટ અથવા બીજી સારવાર લો છો?","allergies":"કોઈ દવા, ખોરાક અથવા અન્ય વસ્તુથી એલર્જી છે? હોય તો શું થાય છે?","family_history":"નજીકના પરિવારમાં ડાયાબિટીસ, હાઈ બ્લડ પ્રેશર, હૃદયરોગ, દમ અથવા બીજી મહત્વની બીમારી છે?","personal_history":"ઊંઘ, ખોરાક, કસરત, તમાકુ, દારૂ, કામ અથવા તણાવ જેવી દૈનિક આદતોમાં કંઈ મહત્વનું છે?","review_systems":"અંતમાં, તાજેતરમાં તાવ, વજનમાં ફેરફાર, શ્વાસની તકલીફ, છાતીમાં અસ્વસ્થતા, ઉલટી, મળ/પેશાબમાં ફેરફાર, ચક્કર અથવા બીજું નવું લક્ષણ થયું છે?"
 },
 "ta-IN": {
  "chief_complaint":"இன்று இங்கு வருவதற்கான முக்கிய பிரச்சனை அல்லது அறிகுறி என்ன? உங்கள் சொந்த வார்த்தைகளில் கூறுங்கள்.","onset":"இந்த பிரச்சனை எப்போது தொடங்கியது? திடீரென்று அல்லது மெதுவாக?","location":"இந்த பிரச்சனை அல்லது அறிகுறி சரியாக எங்கு உணரப்படுகிறது? வேறு இடத்துக்கு பரவுகிறதா?","severity":"இப்போது இந்த அசௌகரியம் 0 முதல் 10 வரை எவ்வளவு? 0 என்றால் இல்லை, 10 என்றால் மிகவும் அதிகம்.","character":"இந்த அறிகுறியை எப்படி விவரிப்பீர்கள்—கூர்மையானது, மந்தமானது, எரிச்சல், துடிப்பு, இறுக்கம், கனத்தது அல்லது வேறு ஏதாவது?","associated":"இந்த பிரச்சனையுடன் வேறு அறிகுறிகள் உள்ளனவா? முக்கியமாகத் தோன்றுவதைச் சொல்லுங்கள்.","past_history":"உங்களுக்கு ஏதேனும் பழைய நோய் உள்ளதா அல்லது முன்பு முக்கியமான நோய்/அறுவை சிகிச்சை நடந்ததா?","medications":"தற்போது ஏதேனும் மருந்து, சப்ப்ளிமென்ட் அல்லது வேறு சிகிச்சை எடுத்துக்கொள்கிறீர்களா?","allergies":"மருந்து, உணவு அல்லது வேறு ஏதாவது பொருளுக்கு ஒவ்வாமை உள்ளதா? இருந்தால் என்ன ஆகிறது?","family_history":"உங்கள் நெருங்கிய குடும்பத்தில் சர்க்கரை நோய், உயர் இரத்த அழுத்தம், இதய நோய், ஆஸ்துமா அல்லது வேறு முக்கிய நோய் உள்ளதா?","personal_history":"தூக்கம், உணவு, உடற்பயிற்சி, புகையிலை, மது, வேலை அல்லது மன அழுத்தம் போன்ற தினசரி பழக்கங்களில் முக்கியமான ஏதாவது உள்ளதா?","review_systems":"முடிப்பதற்கு முன், சமீபத்தில் காய்ச்சல், எடை மாற்றம், மூச்சுத்திணறல், மார்பு அசௌகரியம், வாந்தி, மலம்/சிறுநீர் மாற்றம், தலைசுற்றல் அல்லது வேறு புதிய அறிகுறி இருந்ததா?"
 },
 "te-IN": {
  "chief_complaint":"ఈరోజు ఇక్కడికి రావడానికి ప్రధాన సమస్య లేదా లక్షణం ఏమిటి? మీ మాటల్లో చెప్పండి.","onset":"ఈ సమస్య ఎప్పుడు మొదలైంది? అకస్మాత్తుగా లేదా క్రమంగా?","location":"ఈ సమస్య లేదా లక్షణం మీకు ఖచ్చితంగా ఎక్కడ అనిపిస్తుంది? మరెక్కడికైనా వ్యాపిస్తుందా?","severity":"ప్రస్తుతం ఈ అసౌకర్యం 0 నుంచి 10లో ఎంత? 0 అంటే లేదు, 10 అంటే చాలా ఎక్కువ.","character":"ఈ లక్షణాన్ని ఎలా వివరిస్తారు—తీవ్రంగా, మెల్లగా, మంటగా, కొట్టుకుంటున్నట్టు, బిగుతుగా, భారంగా లేదా వేరేలా?","associated":"ఈ సమస్యతో పాటు మరే ఇతర లక్షణాలు ఉన్నాయా? ముఖ్యంగా అనిపించినవి చెప్పండి.","past_history":"మీకు ఏదైనా పాత వ్యాధి ఉందా లేదా గతంలో ముఖ్యమైన అనారోగ్యం/ఆపరేషన్ జరిగిందా?","medications":"మీరు ప్రస్తుతం ఏదైనా మందులు, సప్లిమెంట్లు లేదా ఇతర చికిత్స తీసుకుంటున్నారా?","allergies":"ఏదైనా మందు, ఆహారం లేదా ఇతర వస్తువుకు అలర్జీ ఉందా? ఉంటే ఏమవుతుంది?","family_history":"మీ దగ్గరి కుటుంబంలో మధుమేహం, అధిక రక్తపోటు, గుండె జబ్బు, ఆస్తమా లేదా ఇతర ముఖ్యమైన వ్యాధి ఉందా?","personal_history":"నిద్ర, ఆహారం, వ్యాయామం, పొగాకు, మద్యం, పని లేదా ఒత్తిడి వంటి రోజువారీ అలవాట్లలో ముఖ్యమైన విషయం ఏదైనా ఉందా?","review_systems":"చివరగా, ఇటీవల జ్వరం, బరువు మార్పు, శ్వాస ఇబ్బంది, ఛాతి అసౌకర్యం, వాంతులు, మలం/మూత్ర మార్పులు, తల తిరగడం లేదా కొత్త లక్షణం ఏదైనా ఉందా?"
 },
 "mr-IN": {
  "chief_complaint":"आज येथे येण्याचे मुख्य कारण किंवा त्रास काय आहे? तुमच्या शब्दांत सांगा.","onset":"हा त्रास कधी सुरू झाला? अचानक की हळूहळू?","location":"हा त्रास किंवा लक्षण नेमके कुठे जाणवते? दुसरीकडे पसरते का?","severity":"सध्या हा त्रास 0 ते 10 मध्ये किती आहे? 0 म्हणजे नाही आणि 10 म्हणजे सर्वाधिक.","character":"हा त्रास कसा वाटतो—तीव्र, बोथट, जळजळ, ठणका, घट्टपणा, जडपणा किंवा काही वेगळा?","associated":"या त्रासासोबत आणखी काही लक्षणे आहेत का? महत्त्वाचे वाटत असल्यास सांगा.","past_history":"तुम्हाला कोणता जुना आजार आहे का किंवा पूर्वी महत्त्वाचा आजार/शस्त्रक्रिया झाली आहे का?","medications":"तुम्ही सध्या कोणती औषधे, सप्लिमेंट्स किंवा इतर उपचार घेत आहात का?","allergies":"औषध, अन्न किंवा इतर कशाची अॅलर्जी आहे का? असल्यास काय होते?","family_history":"जवळच्या कुटुंबात मधुमेह, उच्च रक्तदाब, हृदयरोग, दमा किंवा इतर महत्त्वाचा आजार आहे का?","personal_history":"झोप, आहार, व्यायाम, तंबाखू, दारू, काम किंवा ताण यांसारख्या दैनंदिन सवयींबद्दल काही महत्त्वाचे आहे का?","review_systems":"शेवटी, अलीकडे ताप, वजनातील बदल, श्वास घेण्यास त्रास, छातीत अस्वस्थता, उलटी, शौच/लघवीतील बदल, चक्कर किंवा इतर नवीन लक्षण झाले आहे का?"
 },
 "kn-IN": {
  "chief_complaint":"ಇಂದು ಇಲ್ಲಿಗೆ ಬರಲು ಮುಖ್ಯ ಕಾರಣವಾದ ಸಮಸ್ಯೆ ಅಥವಾ ಲಕ್ಷಣ ಏನು? ನಿಮ್ಮದೇ ಮಾತಿನಲ್ಲಿ ಹೇಳಿ.","onset":"ಈ ಸಮಸ್ಯೆ ಯಾವಾಗ ಆರಂಭವಾಯಿತು? ಏಕಾಏಕಿ ಅಥವಾ ನಿಧಾನವಾಗಿ?","location":"ಈ ಸಮಸ್ಯೆ ಅಥವಾ ಲಕ್ಷಣ ನಿಮಗೆ ನಿಖರವಾಗಿ ಎಲ್ಲಿ ಅನುಭವವಾಗುತ್ತದೆ? ಬೇರೆಡೆಗೆ ಹರಡುತ್ತದೆಯೇ?","severity":"ಈಗ ಅಸೌಕರ್ಯ 0 ರಿಂದ 10ರಲ್ಲಿ ಎಷ್ಟು? 0 ಎಂದರೆ ಇಲ್ಲ, 10 ಎಂದರೆ ಅತ್ಯಂತ ಹೆಚ್ಚು.","character":"ಈ ಲಕ್ಷಣವನ್ನು ಹೇಗೆ ವಿವರಿಸುತ್ತೀರಿ—ತೀಕ್ಷ್ಣ, ಮಂದ, ಉರಿ, ಬಡಿತದಂತೆ, ಬಿಗಿತ, ಭಾರ ಅಥವಾ ಬೇರೆ ರೀತಿಯಲ್ಲಿ?","associated":"ಈ ಸಮಸ್ಯೆಯ ಜೊತೆಗೆ ಬೇರೆ ಯಾವುದೇ ಲಕ್ಷಣಗಳಿವೆಯೇ? ಮುಖ್ಯವೆನಿಸಿದುದನ್ನು ಹೇಳಿ.","past_history":"ನಿಮಗೆ ಯಾವುದೇ ಹಳೆಯ ಕಾಯಿಲೆ ಇದೆಯೇ ಅಥವಾ ಹಿಂದೆ ಪ್ರಮುಖ ಕಾಯಿಲೆ/ಶಸ್ತ್ರಚಿಕಿತ್ಸೆ ಆಗಿದೆಯೇ?","medications":"ನೀವು ಈಗ ಯಾವುದೇ ಔಷಧಿ, ಸಪ್ಲಿಮೆಂಟ್ ಅಥವಾ ಬೇರೆ ಚಿಕಿತ್ಸೆ ಪಡೆಯುತ್ತಿದ್ದೀರಾ?","allergies":"ಯಾವುದೇ ಔಷಧಿ, ಆಹಾರ ಅಥವಾ ಬೇರೆ ವಸ್ತುವಿಗೆ ಅಲರ್ಜಿ ಇದೆಯೇ? ಇದ್ದರೆ ಏನಾಗುತ್ತದೆ?","family_history":"ನಿಮ್ಮ ಹತ್ತಿರದ ಕುಟುಂಬದಲ್ಲಿ ಮಧುಮೇಹ, ಅಧಿಕ ರಕ್ತದೊತ್ತಡ, ಹೃದಯ ಕಾಯಿಲೆ, ಆಸ್ತಮಾ ಅಥವಾ ಬೇರೆ ಪ್ರಮುಖ ಕಾಯಿಲೆ ಇದೆಯೇ?","personal_history":"ನಿದ್ರೆ, ಆಹಾರ, ವ್ಯಾಯಾಮ, ತಂಬಾಕು, ಮದ್ಯ, ಕೆಲಸ ಅಥವಾ ಒತ್ತಡದಂತಹ ದೈನಂದಿನ ಅಭ್ಯಾಸಗಳಲ್ಲಿ ಮುಖ್ಯವಾದ ಏನಾದರೂ ಇದೆಯೇ?","review_systems":"ಕೊನೆಯಲ್ಲಿ, ಇತ್ತೀಚೆಗೆ ಜ್ವರ, ತೂಕದ ಬದಲಾವಣೆ, ಉಸಿರಾಟದ ತೊಂದರೆ, ಎದೆ ಅಸೌಕರ್ಯ, ವಾಂತಿ, ಮಲ/ಮೂತ್ರ ಬದಲಾವಣೆ, ತಲೆಸುತ್ತು ಅಥವಾ ಬೇರೆ ಹೊಸ ಲಕ್ಷಣ ಇದೆಯೇ?"
 }
})

# Common clinical keywords in major Indian languages, including common Hinglish.
PATHWAY_KEYWORDS_MULTI = {
    "chest_pain": ["सीने में दर्द", "सीने में तकलीफ", "छाती में दर्द", "छाती में तकलीफ़", "বুকে ব্যথা", "மார்பு வலி", "ఛాతి నొప్పి", "छातीत दुखणे", "છાતીમાં દુખાવો", "ಎದೆ ನೋವು", "chest me dard", "seene mein dard", "seene me dard"],
    "headache": ["सिरदर्द", "सिर में दर्द", "মাথাব্যথা", "মাথায় ব্যথা", "தலைவலி", "తలనొప్పి", "डोकेदुखी", "માથાનો દુખાવો", "ತಲೆನೋವು", "sir dard", "sar dard"],
    "abdominal_pain": ["पेट में दर्द", "पेट दर्द", "पेट की तकलीफ", "পেটে ব্যথা", "வயிற்று வலி", "కడుపు నొప్పి", "पोटदुखी", "પેટમાં દુખાવો", "ಹೊಟ್ಟೆ ನೋವು", "pet me dard", "stomach me dard"],
    "fever": ["बुखार", "तेज बुखार", "तापमान", "জ্বর", "জ্বর আসছে", "காய்ச்சல்", "జ్వరం", "ताप", "ताप येणे", "તાવ", "ಜ್ವರ", "bukhar", "fever"],
    "respiratory": ["खांसी", "सांस फूलना", "सांस लेने में दिक्कत", "सांस की तकलीफ", "কাশি", "শ্বাসকষ্ট", "இருமல்", "மூச்சுத்திணறல்", "దగ్గు", "శ్వాస తీసుకోవడంలో ఇబ్బంది", "खोकला", "श्वास घेण्यास त्रास", "ઉધરસ", "શ્વાસ લેવામાં તકલીફ", "ಕೆಮ್ಮು", "ಉಸಿರಾಟದ ತೊಂದರೆ", "khansi", "saans phoolna", "saans lene me dikkat"],
}

NEGATION_MULTI = ["no ", "not ", "without ", "नहीं", "नही", "कोई नहीं", "नहीं है", "नाही", "नाही आहे", "নেই", "না ", "இல்லை", "కాదు", "లేదు", "नसणे", "નથી", "ಇಲ್ಲ"]

def localize_question(question: dict, language: str):
    q = dict(question)
    if language in QUESTION_TRANSLATIONS and q["id"] in QUESTION_TRANSLATIONS[language]:
        q["text"] = QUESTION_TRANSLATIONS[language][q["id"]]
    return q


def classify_pathway(text: str) -> str:
    lowered = text.lower().strip()
    for name, pathway in PATHWAYS.items():
        if any(keyword in lowered for keyword in pathway["keywords"]):
            return name
    for name, keywords in PATHWAY_KEYWORDS_MULTI.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return name
    return "general"


def build_question_flow(pathway_name: str):
    pathway = PATHWAYS.get(pathway_name, PATHWAYS["general"])
    return COMMON_QUESTIONS + pathway["questions"] + CLOSING_QUESTIONS


def extract_structured(question_id, answer):
    """Lightweight Phase 4A extraction. A model can replace this function later."""
    text = answer.strip()
    result = {"raw": text}
    direct_map = {
        "chief_complaint": "chief_complaint", "onset": "onset", "location": "location",
        "character": "character", "associated": "associated", "past_history": "past_history",
        "medications": "medications", "allergies": "allergies", "family_history": "family_history",
        "personal_history": "personal_history", "review_systems": "review_systems",
        "chest_radiation": "chest_radiation", "chest_exertion": "chest_exertion",
        "chest_breathlessness": "chest_breathlessness", "headache_visual": "headache_visual",
        "headache_nausea": "headache_nausea", "headache_trigger": "headache_trigger",
        "abdominal_food": "abdominal_food", "abdominal_bowel": "abdominal_bowel",
        "abdominal_urinary": "abdominal_urinary", "fever_pattern": "fever_pattern",
        "fever_infection": "fever_infection", "fever_exposure": "fever_exposure",
        "respiratory_cough": "respiratory_cough", "respiratory_activity": "respiratory_activity",
        "respiratory_wheeze": "respiratory_wheeze", "general_change": "general_change",
        "general_impact": "general_impact",
    }
    if question_id == "severity":
        nums = re.findall(r"\b(?:10|[0-9])\b", text)
        result["severity"] = int(nums[0]) if nums else text
    elif question_id in direct_map:
        result[direct_map[question_id]] = text

    # AI-1B: add conservative semantic evidence without overwriting explicit
    # question answers. The raw answer remains available for verification.
    semantic = extract_clinical_entities(text)
    if semantic.get("positive_symptoms"):
        result["detected_symptoms"] = semantic["positive_symptoms"]
    if semantic.get("negated_symptoms"):
        result["negated_symptoms"] = semantic["negated_symptoms"]
    if semantic.get("duration_mentions"):
        result["duration_mentions"] = semantic["duration_mentions"]
    if semantic.get("severity") is not None and "severity" not in result:
        result["severity_label"] = semantic["severity"]
    result["nlu"] = {
        "intent": semantic.get("intent", "general"),
        "evidence": semantic.get("evidence", []),
        "engine": semantic.get("engine"),
    }
    return result


# ---------- Phase 4B red-flag safety engine ----------
# This is a triage-support prototype, not a diagnostic system. Rules intentionally
# produce review alerts rather than diagnoses.

RED_FLAG_RULES = [
    {
        "id": "severe_breathing_difficulty",
        "level": "emergency",
        "label": "Severe breathing difficulty",
        "questions": {"chief_complaint", "chest_breathlessness", "respiratory_activity", "respiratory_wheeze", "review_systems"},
        "keywords": ["cannot breathe", "can't breathe", "unable to breathe", "struggling to breathe", "severe shortness of breath", "severe breathlessness"],
        "message": "The response describes severe breathing difficulty and should be reviewed urgently by clinical staff."
    },
    {
        "id": "chest_pain_radiation",
        "level": "urgent",
        "label": "Chest discomfort with possible radiation",
        "questions": {"chest_radiation"},
        "keywords": ["left arm", "right arm", "both arms", "jaw", "neck", "back", "shoulder"],
        "message": "Chest discomfort with reported spread to another area warrants prompt clinical review."
    },
    {
        "id": "chest_pain_associated_symptoms",
        "level": "urgent",
        "label": "Chest discomfort with concerning associated symptoms",
        "questions": {"chest_breathlessness", "review_systems"},
        "keywords": ["sweating", "cold sweat", "fainting", "passed out", "shortness of breath", "breathless", "difficulty breathing", "racing heart"],
        "message": "Chest-related symptoms with breathlessness, fainting, marked sweating, or palpitations warrant prompt clinical review."
    },
    {
        "id": "neurological_deficit",
        "level": "urgent",
        "label": "New neurological symptoms",
        "questions": {"headache_visual", "review_systems"},
        "keywords": ["weakness on one side", "one sided weakness", "one-sided weakness", "numbness on one side", "difficulty speaking", "cannot speak", "trouble speaking", "trouble walking", "new weakness", "new numbness"],
        "message": "New neurological symptoms should be reviewed promptly by clinical staff."
    },
    {
        "id": "sudden_severe_headache",
        "level": "urgent",
        "label": "Sudden severe headache",
        "questions": {"onset", "headache_trigger", "chief_complaint"},
        "keywords": ["worst headache", "worst ever", "sudden severe headache", "thunderclap", "explosive headache"],
        "message": "A sudden or exceptionally severe headache warrants prompt clinical assessment."
    },
    {
        "id": "gi_bleeding",
        "level": "urgent",
        "label": "Possible gastrointestinal bleeding",
        "questions": {"abdominal_bowel", "review_systems"},
        "keywords": ["vomiting blood", "blood in vomit", "black stool", "black stools", "blood in stool", "bloody stool"],
        "message": "Reported gastrointestinal bleeding requires prompt clinical review."
    },
    {
        "id": "severe_abdominal_symptoms",
        "level": "urgent",
        "label": "Severe abdominal symptoms",
        "questions": {"abdominal_bowel", "abdominal_urinary", "chief_complaint"},
        "keywords": ["severe abdominal pain", "severe stomach pain", "fainting with abdominal pain", "rigid abdomen"],
        "message": "Severe abdominal symptoms warrant prompt clinical assessment."
    },
    {
        "id": "persistent_high_fever",
        "level": "urgent",
        "label": "High or persistent fever",
        "questions": {"fever_pattern"},
        "keywords": ["very high fever", "persistent high fever", "104", "105", "106"],
        "message": "The reported fever pattern may require prompt clinical review, especially if the patient is very unwell."
    },
]


# Add common Hindi/Hinglish expressions to the safety layer without changing
# the rule IDs or clinical thresholds.
for _rule in RED_FLAG_RULES:
    _extra = {
        "severe_breathing_difficulty": ["सांस नहीं आ रही", "सांस लेने में बहुत दिक्कत", "बहुत ज्यादा सांस फूलना", "saans nahi aa rahi", "saans lene me bahut dikkat"],
        "chest_pain_radiation": ["बाएं हाथ", "बांह", "जबड़े", "गर्दन", "पीठ", "कंधे", "baaye haath", "baazu"],
        "chest_pain_associated_symptoms": ["पसीना", "बेहोशी", "सांस फूलना", "दिल तेज धड़कना", "pasina", "behoshi", "saans phoolna", "dil tez dhadakna"],
        "neurological_deficit": ["एक तरफ कमजोरी", "एक तरफ सुन्न", "बोलने में दिक्कत", "चलने में दिक्कत", "एक तरफ कमजोरी", "ek taraf kamzori", "bolne me dikkat"],
        "sudden_severe_headache": ["जिंदगी का सबसे तेज सिरदर्द", "अचानक बहुत तेज सिरदर्द", "achanak bahut tez sir dard"],
        "gi_bleeding": ["खून की उल्टी", "उल्टी में खून", "काला मल", "मल में खून", "khoon ki ulti", "kaala mal"],
        "severe_abdominal_symptoms": ["बहुत तेज पेट दर्द", "पेट में बहुत तेज दर्द", "bahut tez pet dard"],
        "persistent_high_fever": ["बहुत तेज बुखार", "लगातार तेज बुखार", "bahut tez bukhar", "lagatar tez bukhar"],
    }.get(_rule["id"], [])
    _rule["keywords"].extend(_extra)


def _is_negated(text: str, start: int) -> bool:
    """Simple local negation guard for phrases such as 'no chest pain' or 'not breathless'."""
    prefix = text[max(0, start - 60):start]
    return bool(re.search(r"\b(no|not|without|denies|deny|never|neither)\b|नहीं|नही|कोई नहीं|नाही|নেই|না|இல்லை|కాదు|లేదు|નથી|ಇಲ್ಲ", prefix))


def _keyword_hit(text: str, keyword: str):
    for match in re.finditer(re.escape(keyword), text):
        if not _is_negated(text, match.start()):
            return True
    return False


def detect_red_flags(question_id: str, answer: str, structured: dict):
    """Evaluate the current answer plus already collected answers.

    Returns unique, explainable alerts. Severity is deliberately limited to
    none / watch / urgent / emergency and does not represent a diagnosis.
    """
    all_text = " ".join(str(v) for v in structured.values() if v not in (None, ""))
    all_text += " " + answer
    all_text = re.sub(r"\s+", " ", all_text.lower()).strip()

    alerts = []
    for rule in RED_FLAG_RULES:
        if question_id not in rule["questions"] and not (rule["questions"] & set(structured.keys())):
            continue
        for keyword in rule["keywords"]:
            if _keyword_hit(all_text, keyword.lower()):
                alerts.append({
                    "id": rule["id"],
                    "level": rule["level"],
                    "label": rule["label"],
                    "message": rule["message"],
                })
                break

    # Cross-answer patterns: a single answer may be mild, but combinations can
    # justify a higher-priority review alert.
    chest = " ".join(str(structured.get(k, "")) for k in ("chief_complaint", "chest_radiation", "chest_breathlessness", "chest_exertion"))
    chest = (chest + " " + answer).lower()
    has_chest = any(_keyword_hit(chest, x) for x in ("chest pain", "chest discomfort", "chest pressure", "chest tightness"))
    has_breath = any(_keyword_hit(chest, x) for x in ("shortness of breath", "breathless", "difficulty breathing", "breathlessness"))
    has_sweat = any(_keyword_hit(chest, x) for x in ("sweating", "cold sweat"))
    has_radiation = any(_keyword_hit(chest, x) for x in ("left arm", "right arm", "both arms", "jaw", "back", "neck"))
    if has_chest and (has_breath and (has_sweat or has_radiation)):
        alerts.append({
            "id": "chest_multi_symptom_pattern",
            "level": "emergency",
            "label": "Chest symptoms with multiple concerning features",
            "message": "The combination of chest symptoms and multiple concerning features should be reviewed urgently by clinical/triage staff."
        })

    priority = {"none": 0, "watch": 1, "urgent": 2, "emergency": 3}
    alerts = {a["id"]: a for a in alerts}
    ordered = sorted(alerts.values(), key=lambda x: priority[x["level"]], reverse=True)
    level = ordered[0]["level"] if ordered else "none"
    return level, ordered


def _adaptive_known_slots(answered: dict) -> set:
    known = set()
    for key, value in answered.items():
        if key in {
            "chief_complaint","onset","location","severity","character",
            "chest_radiation","chest_exertion","chest_breathlessness",
            "headache_visual","headache_nausea","headache_trigger",
            "abdominal_food","abdominal_bowel","abdominal_urinary",
            "fever_pattern","fever_infection","fever_exposure",
            "respiratory_cough","respiratory_activity","respiratory_wheeze",
            "general_change","general_impact","associated","past_history",
            "medications","allergies","family_history","personal_history","review_systems"
        } and value not in (None, "", [], {}):
            known.add(key)
    evidence = answered.get("clinical_evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                if item.get("duration_mentions"): known.add("onset")
                if item.get("severity") is not None: known.add("severity")
    return known

QUESTION_PRIORITY = {
    "chief_complaint": 100, "onset": 90, "location": 88, "severity": 86, "character": 84,
    "chest_breathlessness": 99, "chest_radiation": 96, "chest_exertion": 94,
    "headache_visual": 98, "headache_trigger": 84, "headache_nausea": 82,
    "abdominal_bowel": 88, "abdominal_urinary": 80, "abdominal_food": 78,
    "fever_infection": 92, "fever_pattern": 90, "fever_exposure": 70,
    "respiratory_activity": 92, "respiratory_cough": 88, "respiratory_wheeze": 84,
    "general_change": 76, "general_impact": 68, "allergies": 65, "associated": 62,
    "medications": 60, "past_history": 58, "family_history": 42, "personal_history": 38, "review_systems": 30,
}

def adaptive_next_question(pathway_name: str, answered: dict):
    flow = build_question_flow(pathway_name)
    known = _adaptive_known_slots(answered)
    candidates = []
    focused_ids = {q["id"] for q in PATHWAYS.get(pathway_name, {}).get("questions", [])}
    for index, question in enumerate(flow):
        qid = question["id"]
        if qid in answered or qid in known:
            continue
        score = QUESTION_PRIORITY.get(qid, 20)
        if qid in focused_ids:
            score += 12
        if qid == "chief_complaint" and "chief_complaint" not in known:
            score += 1000
        candidates.append((score, -index, question))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not candidates:
        return None, flow, {"adaptive": True, "known_slots": sorted(known), "reason": "sufficient_information_for_current_flow"}
    score, _, question = candidates[0]
    return question, flow, {
        "adaptive": True,
        "known_slots": sorted(known),
        "selected_score": score,
        "candidate_count": len(candidates),
        "reason": "highest_priority_unanswered_information",
    }

def next_question_for(pathway_name: str, answered: dict):
    question, flow, _meta = adaptive_next_question(pathway_name, answered)
    return question, flow


def _json_loads(value, default):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def get_or_create_interview_state(db, patient_id: int, session_id: str, language: str = "en-IN"):
    state = db.query(InterviewState).filter(InterviewState.session_id == session_id).first()
    if state:
        if state.patient_id != patient_id:
            raise HTTPException(status_code=403, detail="Interview session does not belong to this patient.")
        if language in SUPPORTED_LANGUAGES and state.language != language and state.status == "active":
            state.language = language
        return state
    patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    state = InterviewState(
        patient_id=patient_id, session_id=session_id,
        language=language if language in SUPPORTED_LANGUAGES else "en-IN",
        structured_data=json.dumps({}, ensure_ascii=False),
        answered_question_ids=json.dumps([]), conversation=json.dumps([]),
        red_flags=json.dumps([]),
    )
    db.add(state)
    db.flush()
    return state


def interview_state_response(state: InterviewState):
    return {
        "session_id": state.session_id,
        "patient_id": state.patient_id,
        "language": state.language,
        "pathway": state.pathway,
        "current_question_id": state.current_question_id,
        "status": state.status,
        "structured": _json_loads(state.structured_data, {}),
        "answered_question_ids": _json_loads(state.answered_question_ids, []),
        "conversation": _json_loads(state.conversation, []),
        "risk_level": state.risk_level or "none",
        "red_flags": _json_loads(state.red_flags, []),
        "version": state.version,
        "repair_count": state.repair_count or 0,
        "voice_failure_count": state.voice_failure_count or 0,
        "last_input_mode": state.last_input_mode or "text",
        "last_repair_action": state.last_repair_action,
        "created_at": state.created_at.isoformat() if state.created_at else None,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
    }



# ---------- Phase 5C: Real AI / NLP layer ----------
# A small locally trained text classifier is bundled for the hackathon demo.
# It is deliberately used for complaint/pathway understanding, not diagnosis.
NLP_TRAIN_TEXTS = [
    "chest pain pressure tightness discomfort heart pain", "pain in chest while walking", "chest pain sweating",
    "headache migraine head pain dizziness", "severe headache with nausea", "pain behind eyes",
    "stomach pain abdominal pain bloating gas", "abdominal discomfort after food", "loose motions stomach cramps",
    "fever temperature chills body ache", "high fever and chills", "feeling feverish",
    "cough cold sore throat breathing difficulty", "shortness of breath wheezing cough", "phlegm congestion",
    "back pain joint pain muscle ache", "knee pain shoulder pain", "body pain after activity",
    "tired fatigue weakness low energy", "poor sleep insomnia stress", "loss of appetite weakness",
    "skin itching rash redness", "acne skin irritation", "itchy patches",
    "urine burning frequent urination", "pain while passing urine", "urinary discomfort",
    "anxiety worry nervousness", "feeling stressed unable to relax", "low mood emotional distress",
]
NLP_TRAIN_LABELS = [
    "chest","chest","chest","headache","headache","headache","abdominal","abdominal","abdominal",
    "fever","fever","fever","respiratory","respiratory","respiratory","pain","pain","pain",
    "general","general","general","skin","skin","skin","urinary","urinary","urinary","mental_wellbeing","mental_wellbeing","mental_wellbeing"
]
NLP_MODEL = None
if SKLEARN_AVAILABLE:
    try:
        NLP_MODEL = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        NLP_MODEL.fit(NLP_TRAIN_TEXTS, NLP_TRAIN_LABELS)
    except Exception:
        NLP_MODEL = None

NLP_SYMPTOMS = {
    "chest pain": ["chest pain", "chest discomfort", "chest pressure", "chest tightness"],
    "headache": ["headache", "head pain", "migraine"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light headed"],
    "nausea": ["nausea", "nauseous", "vomiting", "vomit"],
    "abdominal pain": ["stomach pain", "abdominal pain", "belly pain", "abdominal discomfort"],
    "bloating": ["bloating", "bloated", "gas"],
    "fever": ["fever", "feverish", "high temperature", "temperature"],
    "cough": ["cough", "coughing"],
    "shortness of breath": ["shortness of breath", "breathless", "difficulty breathing", "breathing difficulty"],
    "wheeze": ["wheeze", "wheezing"],
    "sore throat": ["sore throat", "throat pain"],
    "fatigue": ["fatigue", "tired", "tiredness", "low energy", "weakness"],
    "sleep disturbance": ["poor sleep", "insomnia", "cannot sleep", "can't sleep", "sleep disturbance"],
    "anxiety/stress": ["anxiety", "anxious", "stress", "stressed", "worry", "nervous"],
    "skin rash/itching": ["rash", "itching", "itchy", "skin irritation"],
    "urinary symptoms": ["burning urination", "burning while urinating", "frequent urination", "urine burning"],
    "back/joint pain": ["back pain", "joint pain", "knee pain", "shoulder pain", "muscle ache"],
    "diarrhea": ["diarrhea", "loose motions", "loose stools"],
    "appetite change": ["loss of appetite", "poor appetite", "increased appetite"],
}
NLP_NEGATIONS = ["no ", "not ", "without ", "never ", "don't ", "do not ", "didn't ", "did not ", "nahi ", "nahin ", "नहीं "]

def _normalize_nlp_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

def _negated(text: str, start: int) -> bool:
    window = text[max(0, start-35):start]
    return any(n in window for n in NLP_NEGATIONS)

def analyze_nlp_text(text: str, language: str = "en-IN"):
    raw = text or ""
    normalized = _normalize_nlp_text(raw)
    symptoms=[]
    for canonical, variants in NLP_SYMPTOMS.items():
        for term in variants:
            pos=normalized.find(term)
            if pos >= 0:
                if not _negated(normalized, pos):
                    symptoms.append({"term": canonical, "mention": term, "negated": False})
                else:
                    symptoms.append({"term": canonical, "mention": term, "negated": True})
                break
    # temporal expressions are extracted as evidence, not interpreted as diagnosis.
    durations=[]
    duration_patterns=[
        r"\b(?:for|since)\s+((?:\d+|one|two|three|four|five|six|seven|a|an))\s*(day|days|week|weeks|month|months|year|years)\b",
        r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*(?:ago|se pehle)\b",
        r"\b(since yesterday|since last night|since morning|for a long time)\b",
    ]
    for pat in duration_patterns:
        durations.extend(m.group(0) for m in re.finditer(pat, normalized, re.I))
    severity=None
    sev_map={"mild":["mild","slight","little"],"moderate":["moderate","medium"],"severe":["severe","very painful","worst","extreme","unbearable"]}
    for level, terms in sev_map.items():
        if any(t in normalized for t in terms): severity=level; break
    if NLP_MODEL:
        try:
            probs=NLP_MODEL.predict_proba([normalized])[0]
            classes=NLP_MODEL.classes_
            idx=probs.argmax()
            model_intent=str(classes[idx]); confidence=round(float(probs[idx]),3)
        except Exception:
            model_intent,confidence="general",0.0
    else:
        model_intent,confidence="general",0.0
    # Resolve an explicit symptom cluster into the nearest history pathway, while
    # preserving the raw ML prediction so the demo can show both signals.
    symptom_to_pathway={
        "chest pain":"chest","headache":"headache","dizziness":"headache","nausea":"abdominal",
        "abdominal pain":"abdominal","bloating":"abdominal","fever":"fever","cough":"respiratory",
        "shortness of breath":"respiratory","wheeze":"respiratory","sore throat":"respiratory",
        "fatigue":"general","sleep disturbance":"general","anxiety/stress":"mental_wellbeing",
        "skin rash/itching":"skin","urinary symptoms":"urinary","back/joint pain":"pain",
        "diarrhea":"abdominal","appetite change":"general"
    }
    positive=[x["term"] for x in symptoms if not x["negated"]]
    resolved=intent = symptom_to_pathway.get(positive[0], model_intent) if positive else model_intent
    return {
        "engine": "TF-IDF + Logistic Regression + clinical NLP rules",
        "model_available": bool(NLP_MODEL),
        "intent": intent,
        "model_intent": model_intent,
        "confidence": confidence,
        "symptoms": symptoms,
        "positive_symptoms": [x["term"] for x in symptoms if not x["negated"]],
        "negated_symptoms": [x["term"] for x in symptoms if x["negated"]],
        "duration_mentions": durations,
        "severity": severity,
        "language": language,
        "text_normalized": normalized,
        "disclaimer": "AI/NLP extraction supports clinical history organization; it is not a diagnosis or treatment recommendation."
    }

@app.post("/api/nlp/analyze")
def nlp_analyze(payload: dict):
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for NLP analysis.")
    return analyze_nlp_text(text, str(payload.get("language", "en-IN")))


# -----------------------------------------------------------------------------
# Phase AI-1B: conservative clinical language understanding
# -----------------------------------------------------------------------------
@app.post("/api/ai/clinical-understand")
def clinical_understand(payload: ClinicalNLURequest):
    if payload.language not in SUPPORTED_LANGUAGES:
        payload.language = "en-IN"
    return extract_clinical_entities(payload.text, payload.language)


# -----------------------------------------------------------------------------
# Phase 5F: Validation
# A small, transparent, versioned benchmark for the prototype NLP layer.
# These are synthetic test cases for engineering/demo validation, not clinical
# ground truth and not a substitute for prospective clinical validation.
# -----------------------------------------------------------------------------
VALIDATION_DATASET_VERSION = "5F.1"
VALIDATION_CASES = [
    {"id":"V01","text":"I have a headache and feel dizzy.","intent":"headache","symptoms":["headache","dizziness"]},
    {"id":"V02","text":"My chest hurts and I have chest pain.","intent":"chest","symptoms":["chest pain"]},
    {"id":"V03","text":"I have been coughing for three days.","intent":"respiratory","symptoms":["cough"]},
    {"id":"V04","text":"I feel short of breath and wheezy.","intent":"respiratory","symptoms":["shortness of breath","wheeze"]},
    {"id":"V05","text":"I have nausea and abdominal pain.","intent":"abdominal","symptoms":["nausea","abdominal pain"]},
    {"id":"V06","text":"I have bloating after meals.","intent":"abdominal","symptoms":["bloating"]},
    {"id":"V07","text":"I have had a fever since yesterday.","intent":"fever","symptoms":["fever"]},
    {"id":"V08","text":"I feel tired and exhausted all day.","intent":"general","symptoms":["fatigue"]},
    {"id":"V09","text":"My sleep has been disturbed for two weeks.","intent":"general","symptoms":["sleep disturbance"]},
    {"id":"V10","text":"I am feeling anxious and stressed lately.","intent":"mental_wellbeing","symptoms":["anxiety/stress"]},
    {"id":"V11","text":"I have an itchy skin rash.","intent":"skin","symptoms":["skin rash/itching"]},
    {"id":"V12","text":"I have burning and urinary symptoms.","intent":"urinary","symptoms":["urinary symptoms"]},
    {"id":"V13","text":"My back and joints hurt.","intent":"pain","symptoms":["back/joint pain"]},
    {"id":"V14","text":"I have diarrhea and nausea.","intent":"abdominal","symptoms":["diarrhea","nausea"]},
    {"id":"V15","text":"I do not have chest pain or shortness of breath.","intent":"general","symptoms":["chest pain","shortness of breath"],"negated":["chest pain","shortness of breath"]},
    {"id":"V16","text":"No headache today, but I feel fatigued.","intent":"general","symptoms":["headache","fatigue"],"negated":["headache"]},
    {"id":"V17","text":"I have a severe headache.","intent":"headache","symptoms":["headache"],"severity":"severe"},
    {"id":"V18","text":"Mild cough for one week.","intent":"respiratory","symptoms":["cough"],"severity":"mild"},
    {"id":"V19","text":"Moderate abdominal pain for five days.","intent":"abdominal","symptoms":["abdominal pain"],"severity":"moderate"},
    {"id":"V20","text":"My appetite has changed recently.","intent":"general","symptoms":["appetite change"]},
]

def _f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def run_validation_suite():
    rows=[]
    tp=fp=fn=0
    intent_correct=0
    neg_correct=0
    neg_total=0
    severity_correct=0
    severity_total=0
    duration_total=0
    duration_found=0
    for case in VALIDATION_CASES:
        pred=analyze_nlp_text(case["text"], "en-IN")
        predicted=set(pred.get("positive_symptoms",[]))
        expected=set(case.get("symptoms",[])) - set(case.get("negated",[]))
        tp_i=len(predicted & expected); fp_i=len(predicted-expected); fn_i=len(expected-predicted)
        tp+=tp_i; fp+=fp_i; fn+=fn_i
        intent_ok=pred.get("intent")==case["intent"]
        intent_correct += int(intent_ok)
        expected_neg=set(case.get("negated",[])); predicted_neg=set(pred.get("negated_symptoms",[]))
        neg_correct += len(expected_neg & predicted_neg)
        neg_total += len(expected_neg)
        if "severity" in case:
            severity_total+=1; severity_correct+=int(pred.get("severity")==case["severity"])
        if "for " in case["text"].lower() or "since " in case["text"].lower():
            duration_total+=1; duration_found+=int(bool(pred.get("duration_mentions")))
        rows.append({"id":case["id"],"expected_intent":case["intent"],"predicted_intent":pred.get("intent"),"intent_correct":intent_ok,"expected_symptoms":sorted(expected),"predicted_symptoms":sorted(predicted),"expected_negated":sorted(expected_neg),"predicted_negated":sorted(predicted_neg),"confidence":round(float(pred.get("confidence",0)),4),"severity_expected":case.get("severity"),"severity_predicted":pred.get("severity"),"duration_detected":bool(pred.get("duration_mentions")),"pass": bool(intent_ok and predicted==expected and expected_neg.issubset(predicted_neg) and ("severity" not in case or pred.get("severity")==case["severity"]))})
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=_f1(precision,recall)
    return {"version":VALIDATION_DATASET_VERSION,"dataset_size":len(VALIDATION_CASES),"metrics":{"symptom_precision":round(precision,4),"symptom_recall":round(recall,4),"symptom_f1":round(f1,4),"intent_accuracy":round(intent_correct/len(VALIDATION_CASES),4),"negation_recall":round(neg_correct/neg_total,4) if neg_total else None,"severity_accuracy":round(severity_correct/severity_total,4) if severity_total else None,"duration_detection_rate":round(duration_found/duration_total,4) if duration_total else None},"counts":{"tp":tp,"fp":fp,"fn":fn,"intent_correct":intent_correct,"negation_correct":neg_correct,"negation_total":neg_total},"cases_passed":sum(1 for r in rows if r["pass"]),"cases_failed":sum(1 for r in rows if not r["pass"]),"cases":rows,"limitations":["Synthetic, hand-labelled engineering benchmark.","Not a clinical validation study and not diagnostic evidence.","Metrics measure this prototype's current rule/model behavior on this fixed dataset; new data is required for unbiased evaluation."]}

@app.get("/api/validation/summary")
def validation_summary():
    return run_validation_suite()

@app.post("/api/ai/safety-evaluate")
def ai_safety_evaluate(payload: dict):
    """AI-2: evaluate patient-reported information for prototype triage flags."""
    answer = str(payload.get("answer") or payload.get("text") or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer or text is required")
    structured = payload.get("structured") or {}
    if not isinstance(structured, dict):
        raise HTTPException(status_code=400, detail="structured must be an object")
    return evaluate_safety(answer, structured)


@app.post("/api/validation/run")
def validation_run():
    return run_validation_suite()


@app.post("/api/ai/next-question")
def ai_next_question(payload: dict):
    answered = payload.get("structured") or payload.get("answered") or {}
    if not isinstance(answered, dict):
        raise HTTPException(status_code=400, detail="structured/answered must be an object.")
    pathway = str(payload.get("pathway") or classify_pathway(str(answered.get("chief_complaint", ""))))
    if pathway not in PATHWAYS:
        pathway = "general"
    language = str(payload.get("language", "en-IN"))
    if language not in SUPPORTED_LANGUAGES:
        language = "en-IN"
    question, flow, meta = adaptive_next_question(pathway, answered)
    return {
        "adaptive": True, "pathway": pathway,
        "pathway_label": PATHWAYS[pathway]["label"],
        "question": localize_question(question, language) if question else None,
        "completed": question is None, "meta": meta,
        "total_questions": len(flow),
        "answered_question_ids": [q["id"] for q in flow if q["id"] in answered],
        "disclaimer": "Question selection organizes clinical history; it is not diagnosis or treatment advice.",
    }


@app.post("/api/ai/orchestrate-turn")
def ai_orchestrate_turn(payload: dict):
    """Run one complete AI-1F turn: repair -> clinical understanding -> state -> safety -> next question."""
    patient_id = payload.get("patient_id")
    session_id = str(payload.get("session_id", "")).strip()
    if not patient_id or not session_id:
        raise HTTPException(status_code=400, detail="patient_id and session_id are required.")
    language = str(payload.get("language", "en-IN"))
    if language not in SUPPORTED_LANGUAGES:
        language = "en-IN"
    text = str(payload.get("text", ""))
    event = str(payload.get("event", "answer"))
    attempt = int(payload.get("attempt", 0) or 0)
    question_id = str(payload.get("question_id", ""))

    db = SessionLocal()
    try:
        state = get_or_create_interview_state(db, int(patient_id), session_id, language)
        if state.status == "completed":
            raise HTTPException(status_code=409, detail="This interview has already been completed.")

        repair = analyze_repair(text, event=event, attempt=attempt, current_language=state.language)
        state.last_input_mode = str(payload.get("input_mode", "voice" if event != "answer" else "text"))

        # Repair actions do not mutate clinical meaning.
        if repair.get("action") != "accept_answer":
            if repair.get("action") == "switch_language" and repair.get("requested_language") in SUPPORTED_LANGUAGES:
                state.language = repair["requested_language"]
            state.repair_count = (state.repair_count or 0) + 1
            if repair.get("action") in {"voice_retry", "touch_fallback"}:
                state.voice_failure_count = (state.voice_failure_count or 0) + 1
            state.last_repair_action = repair.get("action")
            state.updated_at = datetime.utcnow()
            db.commit()
            next_q = None
            if state.current_question_id:
                flow = build_question_flow(state.pathway or "general")
                q = next((q for q in flow if q.get("id") == state.current_question_id), None)
                if q:
                    next_q = localize_question(q, state.language)
            response = build_turn_response(
                state=interview_state_response(state), repair=repair,
                next_question=next_q, risk_level=state.risk_level,
                red_flags=_json_loads(state.red_flags, []),
                localized_repair_message=localized_response(repair, state.language),
            )
            return response

        if not text.strip():
            raise HTTPException(status_code=400, detail="Answer text is required when accepting an answer.")

        answered = _json_loads(state.structured_data, {})
        extracted = extract_structured(question_id, text)
        for key, value in extracted.items():
            if key not in ("raw", "nlu") and value not in (None, ""):
                answered[key] = value
        semantic = extract_clinical_entities(text, state.language)
        ledger = answered.get("clinical_evidence", [])
        if not isinstance(ledger, list): ledger = []
        ledger.append({"question_id": question_id, "text": text.strip(),
                       "positive_symptoms": semantic.get("positive_symptoms", []),
                       "negated_symptoms": semantic.get("negated_symptoms", []),
                       "duration_mentions": semantic.get("duration_mentions", []),
                       "severity": semantic.get("severity"), "intent": semantic.get("intent", "general")})
        answered["clinical_evidence"] = ledger[-50:]

        if not state.pathway and answered.get("chief_complaint"):
            state.pathway = classify_pathway(answered["chief_complaint"])
        pathway = state.pathway or "general"
        risk_level, red_flags = detect_red_flags(question_id, text, answered)
        priority = {"none": 0, "watch": 1, "urgent": 2, "emergency": 3}
        if priority.get(state.risk_level or "none", 0) > priority.get(risk_level, 0):
            risk_level = state.risk_level
        old_flags = _json_loads(state.red_flags, [])
        merged = {x.get("id"): x for x in old_flags if x.get("id")}
        for flag in red_flags: merged[flag.get("id")] = flag
        red_flags = sorted(merged.values(), key=lambda x: priority.get(x.get("level", "none"), 0), reverse=True)
        state.risk_level = risk_level
        state.red_flags = json.dumps(red_flags, ensure_ascii=False)

        answered_ids = _json_loads(state.answered_question_ids, [])
        if question_id and question_id not in answered_ids: answered_ids.append(question_id)
        conversation = _json_loads(state.conversation, [])
        conversation.append({"question_id": question_id, "answer": text.strip(),
                             "extracted": {k:v for k,v in extracted.items() if k != "raw"}})
        next_q, flow = next_question_for(pathway, answered)
        state.current_question_id = next_q["id"] if next_q else None
        state.status = "completed" if next_q is None else "active"
        state.structured_data = json.dumps(answered, ensure_ascii=False)
        state.answered_question_ids = json.dumps(answered_ids, ensure_ascii=False)
        state.conversation = json.dumps(conversation, ensure_ascii=False)
        state.last_repair_action = None
        state.updated_at = datetime.utcnow()
        db.commit()

        localized_q = localize_question(next_q, state.language) if next_q else None
        return build_turn_response(
            state=interview_state_response(state),
            next_question=localized_q,
            risk_level=risk_level,
            red_flags=red_flags,
        )
    finally:
        db.close()


@app.get("/api/interview/questions")
def interview_questions(language: str = "en-IN"):
    if language not in SUPPORTED_LANGUAGES:
        language = "en-IN"
    first = localize_question(COMMON_QUESTIONS[0], language)
    intro = "I will first understand your main concern. Based on your answer, I will ask relevant follow-up questions and then collect the standard clinical history. This prototype supports text, voice and guided answers and does not diagnose disease."
    if language == "hi-IN":
        intro = "मैं पहले आपकी मुख्य समस्या समझूंगा। आपके जवाब के आधार पर मैं संबंधित सवाल पूछूंगा और फिर सामान्य क्लिनिकल इतिहास लूंगा। यह प्रोटोटाइप टेक्स्ट, आवाज़ और टैप करके जवाब देने की सुविधा देता है और बीमारी का निदान नहीं करता।"
    return {"intro": intro, "question": first, "pathway": None, "adaptive": True, "language": language, "supported_languages": SUPPORTED_LANGUAGES, "message": "The next questions will adapt to your main complaint."}



@app.get("/api/interview/state/{session_id}")
def interview_state(session_id: str, patient_id: int):
    """Resume an active AI-1A encounter from server-owned state."""
    db = SessionLocal()
    try:
        state = db.query(InterviewState).filter(InterviewState.session_id == session_id, InterviewState.patient_id == patient_id).first()
        if not state:
            raise HTTPException(status_code=404, detail="Interview session not found.")
        return interview_state_response(state)
    finally:
        db.close()


@app.post("/api/interview/answer")
def interview_answer(payload: InterviewAnswer):
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Please provide an answer.")
    if payload.language not in SUPPORTED_LANGUAGES:
        payload.language = "en-IN"

    db = SessionLocal()
    try:
        state = get_or_create_interview_state(db, payload.patient_id, payload.session_id, payload.language)
        if state.status == "completed":
            raise HTTPException(status_code=409, detail="This interview has already been completed.")

        # Server-owned memory is the source of truth. The browser's `answers`
        # payload is accepted only as a backward-compatible fallback for older
        # sessions; it is never allowed to overwrite stored fields silently.
        answered = _json_loads(state.structured_data, {})
        if not answered and payload.answers:
            answered.update(payload.answers)

        extracted = extract_structured(payload.question_id, payload.answer)
        for key, value in extracted.items():
            if key not in ("raw", "nlu") and value not in (None, ""):
                answered[key] = value

        # Keep a compact encounter-level evidence ledger for AI-1B. This is
        # additive and never replaces the patient's original wording.
        semantic = extract_clinical_entities(payload.answer, payload.language)
        evidence_ledger = answered.get("clinical_evidence", [])
        if not isinstance(evidence_ledger, list):
            evidence_ledger = []
        evidence_ledger.append({
            "question_id": payload.question_id,
            "text": payload.answer.strip(),
            "positive_symptoms": semantic.get("positive_symptoms", []),
            "negated_symptoms": semantic.get("negated_symptoms", []),
            "duration_mentions": semantic.get("duration_mentions", []),
            "severity": semantic.get("severity"),
            "intent": semantic.get("intent", "general"),
        })
        answered["clinical_evidence"] = evidence_ledger[-50:]

        # The complaint determines the pathway once it is known. We do not
        # change an established pathway on every subsequent answer.
        if not state.pathway:
            complaint = answered.get("chief_complaint", "")
            if complaint:
                state.pathway = classify_pathway(complaint)
        pathway_name = state.pathway or classify_pathway(answered.get("chief_complaint", ""))
        state.pathway = pathway_name

        nlp = analyze_nlp_text(payload.answer, payload.language)
        safety = evaluate_safety(payload.answer, answered)
        risk_level = safety["risk_level"]
        red_flags = safety["alerts"]

        # Preserve the highest risk already seen in this encounter.
        priority = {"none": 0, "watch": 1, "urgent": 2, "emergency": 3}
        if priority.get(state.risk_level or "none", 0) > priority.get(risk_level, 0):
            risk_level = state.risk_level
        else:
            state.risk_level = risk_level
        old_flags = _json_loads(state.red_flags, [])
        merged_flags = {x.get("id"): x for x in old_flags if x.get("id")}
        for flag in red_flags:
            merged_flags[flag.get("id")] = flag
        red_flags = sorted(merged_flags.values(), key=lambda x: priority.get(x.get("level", "none"), 0), reverse=True)
        state.red_flags = json.dumps(red_flags, ensure_ascii=False)

        answered_ids = _json_loads(state.answered_question_ids, [])
        if payload.question_id not in answered_ids:
            answered_ids.append(payload.question_id)

        conversation = _json_loads(state.conversation, [])
        conversation.append({
            "question_id": payload.question_id,
            "question": next((q.get("text") for q in build_question_flow(pathway_name) if q.get("id") == payload.question_id), ""),
            "answer": payload.answer.strip(),
            "extracted": {k: v for k, v in extracted.items() if k != "raw"},
        })

        next_question, flow = next_question_for(pathway_name, answered)
        completed = next_question is None
        state.status = "completed" if completed else "active"
        state.current_question_id = next_question["id"] if next_question else None
        state.structured_data = json.dumps(answered, ensure_ascii=False)
        state.answered_question_ids = json.dumps(answered_ids, ensure_ascii=False)
        state.conversation = json.dumps(conversation, ensure_ascii=False)
        state.updated_at = datetime.utcnow()
        db.commit()

        completed_count = len([q for q in flow if q["id"] in answered])
        return {
            "message": "Answer received",
            "question_id": payload.question_id,
            "extracted": extracted,
            "nlp": nlp,
            "pathway": pathway_name,
            "pathway_label": PATHWAYS[pathway_name]["label"],
            "next_question": localize_question(next_question, payload.language) if next_question else None,
            "completed": completed,
            "language": payload.language,
            "progress": round((completed_count / len(flow)) * 100),
            "question_number": completed_count + (0 if completed else 1),
            "total_questions": len(flow),
            "risk_level": risk_level,
            "red_flags": red_flags,
            "safety": safety,
            "state": interview_state_response(state),
            "safety_message": (
                "Urgent clinical review is recommended based on the responses so far."
                if risk_level in ("urgent", "emergency") else
                "No immediate red flag was identified by the prototype rules so far."
            ),
        }
    finally:
        db.close()


def _get_whisper_model():
    global _whisper_model
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("Voice transcription is not installed. Install faster-whisper first.")
    if _whisper_model is None:
        device = WHISPER_DEVICE
        compute = WHISPER_COMPUTE_TYPE
        if device == "auto":
            # faster-whisper accepts cpu/cuda; CPU int8 is the safest default.
            device = "cpu"
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute)
    return _whisper_model


def _normalize_voice_language(language: str) -> str:
    value = (language or "en-IN").strip()
    aliases = {"en": "en-IN", "hi": "hi-IN", "english": "en-IN", "hindi": "hi-IN"}
    return aliases.get(value.lower(), value)


def _voice_language_code(language: str) -> str:
    return _normalize_voice_language(language).split("-")[0].lower()


def _transcribe_audio(path: str, language: str):
    model = _get_whisper_model()
    lang = _voice_language_code(language)
    segments, info = model.transcribe(
        path,
        language=lang if lang in {"en", "hi"} else None,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    text_value = " ".join(segment.text.strip() for segment in segments).strip()
    detected = getattr(info, "language", None) or lang
    return text_value, detected


ACCESSIBILITY_INPUT_MODES = {"touch", "voice", "hybrid"}
ACCESSIBILITY_FONT_SCALES = {"0.9", "1.0", "1.15", "1.3", "1.5", "1.75", "2.0"}
ACCESSIBILITY_AUDIO_SPEEDS = {"0.75", "1.0", "1.25", "1.5"}

def _accessibility_defaults(patient_id: int):
    return {
        "patient_id": patient_id,
        "language": "en-IN",
        "input_mode": "touch",
        "font_scale": "1.0",
        "high_contrast": False,
        "reduced_motion": False,
        "captions": True,
        "audio_enabled": True,
        "audio_speed": "1.0",
        "assisted_mode": False,
    }

def _serialize_accessibility(row):
    return {
        "patient_id": row.patient_id, "language": row.language, "input_mode": row.input_mode,
        "font_scale": row.font_scale, "high_contrast": bool(row.high_contrast),
        "reduced_motion": bool(row.reduced_motion), "captions": bool(row.captions),
        "audio_enabled": bool(row.audio_enabled), "audio_speed": row.audio_speed,
        "assisted_mode": bool(row.assisted_mode), "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }

def _can_access_patient_preferences(actor, patient_id: int):
    if actor["role"] == "patient" and actor["id"] != patient_id:
        raise HTTPException(status_code=403, detail="You can only manage your own accessibility preferences.")
    if actor["role"] not in {"patient", "doctor", "triage", "admin"}:
        raise HTTPException(status_code=403, detail="Not authorized to access accessibility preferences.")


@app.get("/api/accessibility/capabilities")
def accessibility_capabilities():
    return {
        "phase": "AI-5H",
        "supported_languages": SUPPORTED_LANGUAGES,
        "input_modes": sorted(ACCESSIBILITY_INPUT_MODES),
        "font_scales": sorted(ACCESSIBILITY_FONT_SCALES, key=float),
        "audio_speeds": sorted(ACCESSIBILITY_AUDIO_SPEEDS, key=float),
        "features": {
            "voice_input": True,
            "touch_fallback": True,
            "captions": True,
            "high_contrast": True,
            "reduced_motion": True,
            "assisted_mode": True,
            "server_tts": EDGE_TTS_AVAILABLE,
            "server_stt": FASTER_WHISPER_AVAILABLE,
        },
        "tts_languages": [lang for lang in SUPPORTED_LANGUAGES if lang in TTS_VOICE_MAP],
        "notes": [
            "Voice transcription availability depends on faster-whisper and its configured model.",
            "Server text-to-speech currently exposes configured voices only; touch/text fallback remains available for every supported UI language.",
        ],
    }


@app.get("/api/patients/{patient_id}/accessibility")
def get_accessibility_preferences(patient_id: int, request: Request):
    actor = authenticate_request(request)
    _can_access_patient_preferences(actor, patient_id)
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        row = db.query(AccessibilityPreference).filter(AccessibilityPreference.patient_id == patient_id).first()
        return {"preferences": _serialize_accessibility(row) if row else _accessibility_defaults(patient_id)}
    finally:
        db.close()


@app.put("/api/patients/{patient_id}/accessibility")
def update_accessibility_preferences(patient_id: int, payload: AccessibilityPreferenceRequest, request: Request):
    actor = authenticate_request(request)
    _can_access_patient_preferences(actor, patient_id)
    language = _normalize_voice_language(payload.language)
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"Unsupported language. Choose one of: {', '.join(SUPPORTED_LANGUAGES)}")
    if payload.input_mode not in ACCESSIBILITY_INPUT_MODES:
        raise HTTPException(status_code=422, detail="input_mode must be touch, voice, or hybrid.")
    if payload.font_scale not in ACCESSIBILITY_FONT_SCALES:
        raise HTTPException(status_code=422, detail="font_scale is not supported.")
    if payload.audio_speed not in ACCESSIBILITY_AUDIO_SPEEDS:
        raise HTTPException(status_code=422, detail="audio_speed is not supported.")
    db = SessionLocal()
    try:
        patient = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
        row = db.query(AccessibilityPreference).filter(AccessibilityPreference.patient_id == patient_id).first()
        if not row:
            row = AccessibilityPreference(patient_id=patient_id)
            db.add(row)
        row.language = language
        row.input_mode = payload.input_mode
        row.font_scale = payload.font_scale
        row.high_contrast = int(payload.high_contrast)
        row.reduced_motion = int(payload.reduced_motion)
        row.captions = int(payload.captions)
        row.audio_enabled = int(payload.audio_enabled)
        row.audio_speed = payload.audio_speed
        row.assisted_mode = int(payload.assisted_mode)
        row.updated_at = datetime.utcnow()
        audit(db, actor["id"], actor["role"], "accessibility_preferences_updated", f"patient:{patient_id}")
        db.commit(); db.refresh(row)
        return {"message": "Accessibility preferences saved", "preferences": _serialize_accessibility(row)}
    finally:
        db.close()


@app.get("/api/voice/status")
def voice_status():
    return {
        "phase": "AI-5H",
        "transcription": {
            "available": FASTER_WHISPER_AVAILABLE,
            "provider": "faster-whisper" if FASTER_WHISPER_AVAILABLE else None,
            "model": WHISPER_MODEL_SIZE if FASTER_WHISPER_AVAILABLE else None,
        },
        "tts": {
            "available": EDGE_TTS_AVAILABLE,
            "provider": "edge-tts" if EDGE_TTS_AVAILABLE else None,
        },
        "languages": list(SUPPORTED_LANGUAGES.keys()),
        "tts_languages": [lang for lang in SUPPORTED_LANGUAGES if lang in TTS_VOICE_MAP],
        "raw_audio_persistence": False,
        "diagnostic": False,
    }


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    patient_id: int = Form(...),
    session_id: str = Form(...),
    language: str = Form("en-IN"),
    request: Request = None,
):
    actor = authenticate_request(request) if request is not None else None
    if actor and actor["role"] == "patient" and actor["id"] != patient_id:
        raise HTTPException(status_code=403, detail="You can only submit voice input for your own patient account.")
    language = _normalize_voice_language(language)
    if not FASTER_WHISPER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice transcription is not configured. Install faster-whisper and download/use the configured model.")
    suffix = Path(audio.filename or "audio.webm").suffix.lower() or ".webm"
    allowed = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac"}
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported audio format. Use webm, wav, mp3, m4a, ogg, or flac.")
    temp_path = None
    db = SessionLocal()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=VOICE_TMP_DIR) as tmp:
            temp_path = tmp.name
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        size = os.path.getsize(temp_path)
        if size == 0 or size > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio must be between 1 byte and 25 MB.")
        transcript, detected = _transcribe_audio(temp_path, language)
        if not transcript:
            repair = analyze_repair("", event="no_speech", attempt=0, current_language=language)
            return {
                "transcript": "",
                "detected_language": detected,
                "language": language,
                "nlu": None,
                "voice_turn_id": None,
                "repair": {**repair, "response_text": localized_response(repair, language)},
                "fallback": {"available": True, "mode": "touch"},
            }
        turn = VoiceTurn(patient_id=patient_id, session_id=session_id, language=language, direction="input", transcript=transcript, status="completed", provider="faster-whisper")
        db.add(turn)
        db.commit()
        # Feed the same conservative clinical understanding layer used by text input.
        nlu = extract_clinical_entities(transcript, language=language)
        repair = analyze_repair(transcript, event="answer", attempt=0, current_language=language)
        return {
            "transcript": transcript,
            "detected_language": detected,
            "language": language,
            "nlu": nlu,
            "voice_turn_id": turn.id,
            "repair": {**repair, "response_text": localized_response(repair, language)},
            "fallback": {"available": True, "mode": "touch"},
        }
    finally:
        db.close()
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/api/voice/speak")
async def voice_speak(text: str = Form(...), language: str = Form("en-IN")):
    language = _normalize_voice_language(language)
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Text-to-speech is not configured. Install edge-tts first.")
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Text is required.")
    if len(clean) > 1000:
        raise HTTPException(status_code=413, detail="TTS text is too long for one kiosk prompt.")
    voice = TTS_VOICE_MAP.get(language, TTS_VOICE_MAP["en-IN"])
    output_path = VOICE_TMP_DIR / f"tts_{uuid.uuid4().hex}.mp3"
    try:
        communicate = edge_tts.Communicate(clean, voice)
        await communicate.save(str(output_path))
        return FileResponse(str(output_path), media_type="audio/mpeg", filename="medikiosk_prompt.mp3")
    except Exception as exc:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {exc}")


@app.post("/api/ai/conversation-repair")
def conversation_repair(payload: ConversationRepairRequest):
    """AI-1E: classify conversational repair needs without changing clinical meaning."""
    language = payload.language if payload.language in SUPPORTED_LANGUAGES else "en-IN"
    db = SessionLocal()
    try:
        state = get_or_create_interview_state(db, payload.patient_id, payload.session_id, language)
        repair = analyze_repair(payload.text, event=payload.event, attempt=payload.attempt, current_language=state.language)
        if repair.get("requested_language") in SUPPORTED_LANGUAGES:
            state.language = repair["requested_language"]
            language = state.language
        action = repair.get("action")
        if action in {"repeat_question", "simplify_question", "request_correction", "voice_retry", "touch_fallback", "switch_language"}:
            state.repair_count = (state.repair_count or 0) + 1
        if action in {"voice_retry", "touch_fallback"}:
            state.voice_failure_count = (state.voice_failure_count or 0) + 1
        state.last_input_mode = payload.input_mode
        state.last_repair_action = action
        state.updated_at = datetime.utcnow()
        db.commit()
        repair["response_text"] = localized_response(repair, language)
        repair["language"] = language
        repair["state"] = interview_state_response(state)
        return repair
    finally:
        db.close()


def generate_physician_summary(structured: dict, risk_level: str, red_flags: list, nlp: Optional[dict] = None):
    """Generate a concise, non-diagnostic physician handoff from structured history."""
    s = structured or {}
    complaint = str(s.get("chief_complaint") or "Not clearly stated")
    parts = []

    def val(key):
        v = s.get(key)
        return str(v).strip() if v not in (None, "") else ""

    symptom_items = []
    for key in ("associated_symptoms", "chest_breathlessness", "headache_nausea", "abdominal_bowel", "fever_infection", "respiratory_cough", "respiratory_wheeze", "review_of_systems"):
        v = val(key)
        if v:
            symptom_items.append(v)
    if nlp:
        positives = nlp.get("positive_symptoms") or []
        negatives = nlp.get("negated_symptoms") or []
    else:
        positives, negatives = [], []

    history = []
    for key, label in (("onset","Onset"),("location","Location"),("severity","Severity"),("character","Character"),("general_change","Aggravating/relieving factors"),("general_impact","Daily impact")):
        v = val(key)
        if v: history.append(f"{label}: {v}")

    background=[]
    for key,label in (("past_history","Past history"),("medications","Medications"),("allergies","Allergies"),("family_history","Family history"),("personal_history","Personal history")):
        v=val(key)
        if v: background.append(f"{label}: {v}")

    missing=[]
    for key,label in (("onset","onset"),("severity","severity"),("medications","current medications"),("allergies","allergies")):
        if not val(key): missing.append(label)

    assessment=[]
    assessment.append(f"Primary complaint: {complaint}.")
    if history: assessment.append("History: " + "; ".join(history) + ".")
    if symptom_items: assessment.append("Associated information: " + "; ".join(symptom_items) + ".")
    if positives: assessment.append("NLP-detected symptoms: " + ", ".join(dict.fromkeys(positives)) + ".")
    if negatives: assessment.append("Reported negatives: " + ", ".join(dict.fromkeys(negatives)) + ".")
    if background: assessment.append("Background: " + "; ".join(background) + ".")
    if risk_level != "none": assessment.append(f"Safety status: {risk_level.upper()} review flag from prototype safety rules.")
    else: assessment.append("Safety status: no immediate red flag identified by prototype rules in the collected history.")
    if red_flags: assessment.append("Alerts: " + "; ".join(x.get("label", "Alert") for x in red_flags) + ".")

    focus=[]
    if risk_level in ("urgent","emergency"): focus.append("Review safety alerts before routine history interpretation.")
    if missing: focus.append("Clarify missing history: " + ", ".join(missing) + ".")
    focus.append("Verify the patient-reported history against examination, records and clinician assessment.")

    return {
        "headline": f"Physician handoff — {complaint[:100]}",
        "clinical_summary": " ".join(assessment),
        "key_findings": list(dict.fromkeys(positives + symptom_items)),
        "reported_negatives": list(dict.fromkeys(negatives)),
        "history_points": history,
        "background": background,
        "safety": {"level": risk_level, "alerts": red_flags},
        "physician_focus": focus,
        "data_gaps": missing,
        "engine": "Phase 5D structured clinical synthesis + Phase 5C NLP evidence",
        "disclaimer": "This is an AI-assisted handoff of patient-reported information. It is not a diagnosis, treatment recommendation, or substitute for physician judgment."
    }


# ---------- Phase 5E: Explainability / Sources ----------
# Explainability is deliberately provenance-first: every displayed AI finding is
# tied to either patient text/structured answers, a transparent local NLP rule,
# the local ML classifier, or an authoritative reference page. It never presents
# a source as proof of a diagnosis.
EXPLAINABILITY_SOURCES = [
    {
        "id": "who-icd11",
        "title": "WHO ICD-11",
        "publisher": "World Health Organization",
        "url": "https://www.who.int/standards/classifications/classification-of-diseases",
        "use": "Reference for standardized health terminology and clinical documentation context.",
    },
    {
        "id": "ministry-ayush",
        "title": "Ministry of Ayush",
        "publisher": "Government of India",
        "url": "https://ayush.gov.in/",
        "use": "Official context for AYUSH systems, policy, education and research resources.",
    },
    {
        "id": "ayush-research-portal",
        "title": "AYUSH Research Portal",
        "publisher": "Ministry of Ayush, Government of India",
        "url": "https://arp.ayush.gov.in/",
        "use": "Evidence-based AYUSH research discovery; used as a reference library, not as an automatic treatment recommender.",
    },
]

def _sentence_for_term(text: str, term: str):
    if not text or not term:
        return ""
    for sentence in re.split(r"(?<=[.!?])\\s+|\\n+", text.strip()):
        if term.lower() in sentence.lower():
            return sentence.strip()
    return text.strip()[:240]

def generate_explainability(structured: dict, nlp: dict, summary: dict, risk_level: str, red_flags: list):
    nlp = nlp or {}
    structured = structured or {}
    summary = summary or {}
    raw_text = str(nlp.get("text_normalized") or structured.get("chief_complaint") or "")
    evidence = []
    for item in nlp.get("symptoms") or []:
        mention = item.get("mention") or item.get("term")
        evidence.append({
            "type": "nlp_extraction",
            "finding": item.get("term"),
            "status": "negated" if item.get("negated") else "present",
            "evidence": _sentence_for_term(raw_text, mention),
            "method": "Clinical phrase matching + local negation window",
            "confidence": "high" if mention and mention.lower() in raw_text.lower() else "moderate",
        })

    intent = nlp.get("intent") or "general"
    model_intent = nlp.get("model_intent") or intent
    confidence = float(nlp.get("confidence") or 0)
    evidence.append({
        "type": "intent_classification",
        "finding": intent,
        "status": "selected pathway",
        "evidence": raw_text[:300] or "No free-text evidence available",
        "method": "Local TF-IDF + Logistic Regression; symptom pathway resolution",
        "confidence": f"{round(confidence*100)}% model probability" if confidence else "rule-resolved",
        "details": {"model_intent": model_intent, "resolved_intent": intent},
    })

    for duration in nlp.get("duration_mentions") or []:
        evidence.append({
            "type": "temporal_extraction",
            "finding": duration,
            "status": "reported",
            "evidence": _sentence_for_term(raw_text, duration),
            "method": "Transparent temporal-expression pattern",
            "confidence": "high",
        })
    if nlp.get("severity"):
        evidence.append({
            "type": "severity_extraction",
            "finding": nlp.get("severity"),
            "status": "reported",
            "evidence": raw_text[:300],
            "method": "Transparent severity phrase matching",
            "confidence": "moderate",
        })

    for alert in red_flags or []:
        evidence.append({
            "type": "safety_rule",
            "finding": alert.get("label", "Safety alert"),
            "status": alert.get("level", risk_level),
            "evidence": alert.get("message", "Triggered by the prototype safety rules."),
            "method": "Deterministic red-flag rule engine",
            "confidence": "rule match",
        })

    summary_trace = []
    for finding in summary.get("key_findings") or []:
        matching = [e for e in evidence if e.get("finding") == finding]
        summary_trace.append({
            "summary_item": finding,
            "supported_by": matching[0]["type"] if matching else "structured_history",
            "evidence": matching[0]["evidence"] if matching else str(structured.get("chief_complaint") or "Structured patient history"),
        })

    return {
        "version": "5E.1",
        "headline": "Why the AI produced this handoff",
        "evidence": evidence,
        "summary_trace": summary_trace,
        "sources": EXPLAINABILITY_SOURCES,
        "model": {
            "engine": nlp.get("engine", "Phase 5C NLP"),
            "model_intent": model_intent,
            "resolved_intent": intent,
            "confidence": confidence,
            "limitations": [
                "The classifier is a small local prototype trained on a limited synthetic demonstration set.",
                "Phrase extraction can miss synonyms, context or sarcasm and is not a diagnostic model.",
                "External sources provide reference context; they are not used to automatically diagnose or prescribe treatment.",
            ],
        },
        "safety": {
            "level": risk_level or "none",
            "rule_count": len(red_flags or []),
        },
        "disclaimer": "Explainability shows provenance and evidence for the prototype's outputs. It does not establish a diagnosis, treatment plan, or clinical truth; the physician remains responsible for interpretation.",
    }

@app.get("/api/doctor/consultations/{consultation_id}/explainability")
def get_explainability(consultation_id: int):
    db = SessionLocal()
    try:
        c = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not c:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        structured = json.loads(c.structured_data or "{}")
        nlp = json.loads(c.nlp_data or "{}") if c.nlp_data else {}
        summary = json.loads(c.ai_summary) if c.ai_summary else generate_physician_summary(structured, c.risk_level or "none", json.loads(c.red_flags or "[]"), nlp)
        return {"consultation_id": c.id, "explainability": generate_explainability(structured, nlp, summary, c.risk_level or "none", json.loads(c.red_flags or "[]"))}
    finally:
        db.close()


@app.post("/api/doctor/consultations/{consultation_id}/ai-summary")
def create_ai_summary(consultation_id: int):
    db = SessionLocal()
    try:
        c = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not c:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        structured = json.loads(c.structured_data or "{}")
        nlp = json.loads(c.nlp_data or "{}") if c.nlp_data else {}
        summary = generate_physician_summary(structured, c.risk_level or "none", json.loads(c.red_flags or "[]"), nlp)
        c.ai_summary = json.dumps(summary, ensure_ascii=False)
        c.ai_summary_generated_at = datetime.utcnow()
        db.commit()
        return {"consultation_id": c.id, "ai_summary": summary, "generated_at": c.ai_summary_generated_at.isoformat()}
    finally:
        db.close()


@app.get("/api/doctor/consultations/{consultation_id}/ai-summary")
def get_ai_summary(consultation_id: int):
    db = SessionLocal()
    try:
        c = db.query(Consultation).filter(Consultation.id == consultation_id).first()
        if not c:
            raise HTTPException(status_code=404, detail="Consultation not found.")
        if not c.ai_summary:
            structured = json.loads(c.structured_data or "{}")
            nlp = json.loads(c.nlp_data or "{}") if c.nlp_data else {}
            summary = generate_physician_summary(structured, c.risk_level or "none", json.loads(c.red_flags or "[]"), nlp)
            c.ai_summary = json.dumps(summary, ensure_ascii=False)
            c.ai_summary_generated_at = datetime.utcnow()
            db.commit()
        return {"consultation_id": c.id, "ai_summary": json.loads(c.ai_summary), "generated_at": c.ai_summary_generated_at.isoformat() if c.ai_summary_generated_at else None}
    finally:
        db.close()


@app.post("/api/interview/complete")
def interview_complete(payload: InterviewComplete):
    db = SessionLocal()
    try:
        patient = db.query(User).filter(
            User.id == payload.patient_id, User.role == "patient"
        ).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")

        structured = payload.structured
        complaint = structured.get("chief_complaint", "Clinical history")
        summary_parts = []

        labels = [
            ("chief_complaint", "Main complaint"), ("onset", "Onset"), ("location", "Location"),
            ("severity", "Severity"), ("character", "Character"), ("associated_symptoms", "Associated symptoms"),
            ("chest_radiation", "Chest radiation"), ("chest_exertion", "Chest/exertion pattern"),
            ("chest_breathlessness", "Chest associated symptoms"), ("headache_visual", "Headache neurological/visual symptoms"),
            ("headache_nausea", "Headache associated symptoms"), ("headache_trigger", "Headache triggers"),
            ("abdominal_food", "Abdominal food/relief pattern"), ("abdominal_bowel", "Bowel symptoms"),
            ("abdominal_urinary", "Urinary symptoms"), ("fever_pattern", "Fever pattern"),
            ("fever_infection", "Infection symptoms"), ("fever_exposure", "Exposure history"),
            ("respiratory_cough", "Cough details"), ("respiratory_activity", "Breathing pattern"),
            ("respiratory_wheeze", "Wheeze/lung history"), ("general_change", "Aggravating/relieving factors"),
            ("general_impact", "Daily impact"), ("past_history", "Past history"), ("medications", "Medications"),
            ("allergies", "Allergies"), ("family_history", "Family history"),
            ("personal_history", "Personal history"), ("review_of_systems", "Review of systems"),
        ]
        for key, label in labels:
            if key in structured and structured[key] not in ("", None):
                summary_parts.append(f"{label}: {structured[key]}")

        summary = " | ".join(summary_parts) if summary_parts else "Interview completed."
        risk_level, red_flags = detect_red_flags("completion", "", structured)
        if red_flags:
            summary += " | Safety alerts: " + "; ".join(flag["label"] for flag in red_flags)
        combined_text = " ".join(str(v) for v in structured.values() if v not in (None, ""))
        completion_nlp = analyze_nlp_text(combined_text, "en-IN") if combined_text else {}
        physician_summary = generate_physician_summary(structured, risk_level, red_flags, completion_nlp)
        record = Consultation(
            patient_id=patient.id,
            title=payload.title or str(complaint)[:80],
            summary=summary,
            status="AI History — Phase 5D Physician Summary Ready",
            risk_level=risk_level,
            red_flags=json.dumps(red_flags),
            structured_data=json.dumps(structured, ensure_ascii=False),
            nlp_data=json.dumps(completion_nlp, ensure_ascii=False),
            ai_summary=json.dumps(physician_summary, ensure_ascii=False),
            ai_summary_generated_at=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "message": "Clinical history saved",
            "consultation_id": record.id,
            "title": record.title,
            "summary": summary,
            "structured": structured,
            "risk_level": risk_level,
            "red_flags": red_flags,
            "ai_summary": physician_summary
        }
    finally:
        db.close()


# ---------- AI-5F: Hospital Administration ----------
ADMIN_ROLES = {"admin"}
VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

def require_admin(request: Request):
    actor = authenticate_request(request)
    if actor["role"] not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Hospital administration is restricted to admin users.")
    return actor


def validate_time_range(start_time: str, end_time: str):
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", start_time) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", end_time):
        raise HTTPException(status_code=422, detail="Times must use HH:MM 24-hour format.")
    if start_time >= end_time:
        raise HTTPException(status_code=422, detail="Start time must be earlier than end time.")


def serialize_department(d):
    return {"id": d.id, "name": d.name, "specialty": d.specialty, "active": bool(d.active), "created_at": d.created_at.isoformat() if d.created_at else None}


@app.get("/api/admin/departments")
def admin_list_departments(request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        rows = db.query(Department).order_by(Department.name.asc()).all()
        return {"departments": [serialize_department(d) for d in rows]}
    finally:
        db.close()


@app.post("/api/admin/departments")
def admin_create_department(payload: AdminDepartmentRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        name = payload.name.strip()
        if db.query(Department).filter(Department.name == name).first():
            raise HTTPException(status_code=409, detail="Department already exists.")
        d = Department(name=name, specialty=payload.specialty.strip(), active=int(payload.active))
        db.add(d); db.flush()
        audit(db, actor["id"], actor["role"], "admin_department_create", f"department:{d.id}")
        db.commit(); db.refresh(d)
        return {"message": "Department created", "department": serialize_department(d)}
    finally:
        db.close()


@app.put("/api/admin/departments/{department_id}")
def admin_update_department(department_id: int, payload: AdminDepartmentRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        d = db.query(Department).filter(Department.id == department_id).first()
        if not d: raise HTTPException(status_code=404, detail="Department not found.")
        name = payload.name.strip()
        duplicate = db.query(Department).filter(Department.name == name, Department.id != d.id).first()
        if duplicate: raise HTTPException(status_code=409, detail="Another department already uses this name.")
        old_name = d.name
        d.name, d.specialty, d.active = name, payload.specialty.strip(), int(payload.active)
        # Keep dependent configuration references consistent when an admin renames a department.
        db.query(OPDConfiguration).filter(OPDConfiguration.department == old_name).update({"department": name}, synchronize_session=False)
        db.query(DoctorProfile).filter(DoctorProfile.department == old_name).update({"department": name}, synchronize_session=False)
        db.query(RoutingRule).filter(RoutingRule.department == old_name).update({"department": name}, synchronize_session=False)
        hospital = db.query(HospitalConfiguration).first()
        if hospital and hospital.default_department == old_name:
            hospital.default_department = name
        audit(db, actor["id"], actor["role"], "admin_department_update", f"department:{d.id}:{old_name}->{d.name}")
        db.commit(); db.refresh(d)
        return {"message": "Department updated", "department": serialize_department(d)}
    finally:
        db.close()


@app.get("/api/admin/doctors")
def admin_list_doctors(request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        rows = db.query(User, DoctorProfile).join(DoctorProfile, DoctorProfile.user_id == User.id).filter(User.role == "doctor").order_by(User.name.asc()).all()
        return {"doctors": [{"id": u.id, "name": u.name, "email": u.email, "specialty": p.specialty, "department": p.department, "registration_number": p.registration_number or "", "active": bool(p.active)} for u,p in rows]}
    finally:
        db.close()


@app.post("/api/admin/doctors")
def admin_create_doctor(payload: AdminDoctorRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        if payload.user_id:
            user = db.query(User).filter(User.id == payload.user_id).first()
            if not user: raise HTTPException(status_code=404, detail="User not found.")
            if user.role != "doctor": raise HTTPException(status_code=422, detail="Selected user is not a doctor.")
            if db.query(DoctorProfile).filter(DoctorProfile.user_id == user.id).first(): raise HTTPException(status_code=409, detail="Doctor profile already exists.")
        else:
            if not payload.name or not payload.email or not payload.password:
                raise HTTPException(status_code=422, detail="name, email and password are required for a new doctor.")
            email = payload.email.strip().lower()
            if db.query(User).filter(User.email == email).first(): raise HTTPException(status_code=409, detail="Email is already registered.")
            user = User(name=payload.name.strip(), email=email, password_hash=hash_password(payload.password), role="doctor")
            db.add(user); db.flush()
        profile = DoctorProfile(user_id=user.id, specialty=payload.specialty.strip(), department=payload.department.strip(), registration_number=payload.registration_number.strip(), active=int(payload.active))
        db.add(profile); db.flush()
        audit(db, actor["id"], actor["role"], "admin_doctor_create", f"doctor:{user.id}")
        db.commit()
        return {"message": "Doctor created", "doctor": {"id": user.id, "name": user.name, "email": user.email, "specialty": profile.specialty, "department": profile.department, "registration_number": profile.registration_number or "", "active": bool(profile.active)}}
    finally:
        db.close()


@app.put("/api/admin/doctors/{doctor_id}")
def admin_update_doctor(doctor_id: int, payload: AdminDoctorRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == doctor_id, User.role == "doctor").first()
        if not user: raise HTTPException(status_code=404, detail="Doctor not found.")
        profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == doctor_id).first()
        if not profile:
            profile = DoctorProfile(user_id=doctor_id); db.add(profile)
        if payload.name: user.name = payload.name.strip()
        if payload.email:
            email = payload.email.strip().lower()
            dup = db.query(User).filter(User.email == email, User.id != doctor_id).first()
            if dup: raise HTTPException(status_code=409, detail="Email is already registered.")
            user.email = email
        if payload.password: user.password_hash = hash_password(payload.password)
        profile.specialty, profile.department = payload.specialty.strip(), payload.department.strip()
        profile.registration_number, profile.active = payload.registration_number.strip(), int(payload.active)
        audit(db, actor["id"], actor["role"], "admin_doctor_update", f"doctor:{doctor_id}")
        db.commit()
        return {"message": "Doctor updated", "doctor_id": doctor_id, "active": bool(profile.active)}
    finally:
        db.close()


@app.get("/api/admin/opd-config")
def admin_list_opd(request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        rows = db.query(OPDConfiguration).order_by(OPDConfiguration.department.asc()).all()
        return {"opd_configurations": [{"id": r.id, "department": r.department, "working_days": r.working_days.split(",") if r.working_days else [], "start_time": r.start_time, "end_time": r.end_time, "active": bool(r.active)} for r in rows]}
    finally: db.close()


@app.post("/api/admin/opd-config")
def admin_upsert_opd(payload: AdminOPDRequest, request: Request):
    actor = require_admin(request)
    if any(day not in VALID_DAYS for day in payload.working_days): raise HTTPException(status_code=422, detail="working_days contains an invalid day.")
    validate_time_range(payload.start_time, payload.end_time)
    db = SessionLocal()
    try:
        row = db.query(OPDConfiguration).filter(OPDConfiguration.department == payload.department.strip()).first()
        if not row: row = OPDConfiguration(department=payload.department.strip()); db.add(row)
        row.working_days = ",".join(dict.fromkeys(payload.working_days)); row.start_time = payload.start_time; row.end_time = payload.end_time; row.active = int(payload.active); row.updated_at = datetime.utcnow()
        audit(db, actor["id"], actor["role"], "admin_opd_config_update", f"department:{row.department}")
        db.commit(); db.refresh(row)
        return {"message": "OPD configuration saved", "id": row.id, "department": row.department, "working_days": row.working_days.split(","), "start_time": row.start_time, "end_time": row.end_time, "active": bool(row.active)}
    finally: db.close()


@app.get("/api/admin/availability")
def admin_list_availability(request: Request, doctor_id: Optional[int] = None):
    require_admin(request)
    db = SessionLocal()
    try:
        q = db.query(DoctorAvailability)
        if doctor_id: q = q.filter(DoctorAvailability.doctor_id == doctor_id)
        rows = q.order_by(DoctorAvailability.doctor_id, DoctorAvailability.day_of_week, DoctorAvailability.start_time).all()
        return {"availability": [{"id": r.id, "doctor_id": r.doctor_id, "day_of_week": r.day_of_week, "start_time": r.start_time, "end_time": r.end_time, "active": bool(r.active)} for r in rows]}
    finally: db.close()


@app.post("/api/admin/availability")
def admin_create_availability(payload: AdminAvailabilityRequest, request: Request):
    actor = require_admin(request)
    if payload.day_of_week not in VALID_DAYS: raise HTTPException(status_code=422, detail="Invalid day_of_week.")
    validate_time_range(payload.start_time, payload.end_time)
    db = SessionLocal()
    try:
        doctor = db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first()
        if not doctor: raise HTTPException(status_code=404, detail="Doctor not found.")
        if db.query(DoctorAvailability).filter(DoctorAvailability.doctor_id == payload.doctor_id, DoctorAvailability.day_of_week == payload.day_of_week, DoctorAvailability.start_time == payload.start_time, DoctorAvailability.end_time == payload.end_time).first(): raise HTTPException(status_code=409, detail="Availability window already exists.")
        row = DoctorAvailability(doctor_id=payload.doctor_id, day_of_week=payload.day_of_week, start_time=payload.start_time, end_time=payload.end_time, active=int(payload.active))
        db.add(row); db.flush(); audit(db, actor["id"], actor["role"], "admin_availability_create", f"availability:{row.id}"); db.commit()
        return {"message": "Doctor availability created", "availability_id": row.id}
    finally: db.close()


@app.put("/api/admin/availability/{availability_id}")
def admin_update_availability(availability_id: int, payload: AdminAvailabilityRequest, request: Request):
    actor = require_admin(request)
    if payload.day_of_week not in VALID_DAYS: raise HTTPException(status_code=422, detail="Invalid day_of_week.")
    validate_time_range(payload.start_time, payload.end_time)
    db = SessionLocal()
    try:
        row = db.query(DoctorAvailability).filter(DoctorAvailability.id == availability_id).first()
        if not row: raise HTTPException(status_code=404, detail="Availability not found.")
        doctor = db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first()
        if not doctor: raise HTTPException(status_code=404, detail="Doctor not found.")
        duplicate = db.query(DoctorAvailability).filter(DoctorAvailability.doctor_id == payload.doctor_id, DoctorAvailability.day_of_week == payload.day_of_week, DoctorAvailability.start_time == payload.start_time, DoctorAvailability.end_time == payload.end_time, DoctorAvailability.id != row.id).first()
        if duplicate: raise HTTPException(status_code=409, detail="Availability window already exists.")
        row.doctor_id, row.day_of_week, row.start_time, row.end_time, row.active = payload.doctor_id, payload.day_of_week, payload.start_time, payload.end_time, int(payload.active)
        audit(db, actor["id"], actor["role"], "admin_availability_update", f"availability:{row.id}"); db.commit()
        return {"message": "Doctor availability updated", "availability_id": row.id}
    finally: db.close()


@app.get("/api/admin/routing")
def admin_list_routing(request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        rows = db.query(RoutingRule).order_by(RoutingRule.department, RoutingRule.priority, RoutingRule.id).all()
        return {"routing_rules": [{"id": r.id, "department": r.department, "specialty": r.specialty, "doctor_id": r.doctor_id, "priority": r.priority, "active": bool(r.active)} for r in rows]}
    finally: db.close()


@app.post("/api/admin/routing")
def admin_create_routing(payload: AdminRoutingRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        if payload.doctor_id and not db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first(): raise HTTPException(status_code=404, detail="Doctor not found.")
        row = RoutingRule(department=payload.department.strip(), specialty=payload.specialty.strip(), doctor_id=payload.doctor_id, priority=payload.priority, active=int(payload.active))
        db.add(row); db.flush(); audit(db, actor["id"], actor["role"], "admin_routing_create", f"routing:{row.id}"); db.commit()
        return {"message": "Routing rule created", "routing_rule_id": row.id}
    finally: db.close()


@app.put("/api/admin/routing/{routing_id}")
def admin_update_routing(routing_id: int, payload: AdminRoutingRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        row = db.query(RoutingRule).filter(RoutingRule.id == routing_id).first()
        if not row: raise HTTPException(status_code=404, detail="Routing rule not found.")
        if payload.doctor_id and not db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first(): raise HTTPException(status_code=404, detail="Doctor not found.")
        row.department, row.specialty, row.doctor_id, row.priority, row.active = payload.department.strip(), payload.specialty.strip(), payload.doctor_id, payload.priority, int(payload.active)
        audit(db, actor["id"], actor["role"], "admin_routing_update", f"routing:{row.id}"); db.commit()
        return {"message": "Routing rule updated", "routing_rule_id": row.id}
    finally: db.close()


@app.get("/api/admin/hospital")
def admin_get_hospital(request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        row = db.query(HospitalConfiguration).first()
        if not row: row = HospitalConfiguration(); db.add(row); db.commit(); db.refresh(row)
        return {"hospital": {"id": row.id, "hospital_name": row.hospital_name, "facility_code": row.facility_code, "timezone": row.timezone, "default_department": row.default_department, "active": bool(row.active)}}
    finally: db.close()


@app.put("/api/admin/hospital")
def admin_update_hospital(payload: AdminHospitalRequest, request: Request):
    actor = require_admin(request)
    db = SessionLocal()
    try:
        row = db.query(HospitalConfiguration).first()
        if not row: row = HospitalConfiguration(); db.add(row)
        row.hospital_name, row.facility_code, row.timezone, row.default_department, row.active = payload.hospital_name.strip(), payload.facility_code.strip(), payload.timezone.strip(), payload.default_department.strip(), int(payload.active)
        row.updated_at = datetime.utcnow()
        audit(db, actor["id"], actor["role"], "admin_hospital_update", "hospital_configuration")
        db.commit(); db.refresh(row)
        return {"message": "Hospital configuration saved", "hospital": {"id": row.id, "hospital_name": row.hospital_name, "facility_code": row.facility_code, "timezone": row.timezone, "default_department": row.default_department, "active": bool(row.active)}}
    finally: db.close()


@app.get("/api/admin/analytics")
def admin_analytics(request: Request, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Aggregate operational analytics for hospital administrators.

    Metrics are read-only and intentionally aggregate; this endpoint does not
    expose individual patient records or make clinical decisions.
    """
    require_admin(request)
    db = SessionLocal()
    try:
        try:
            return build_analytics_dashboard(
                db,
                Encounter=Encounter,
                Consultation=Consultation,
                MedicalDocument=MedicalDocument,
                AuditEvent=AuditEvent,
                User=User,
                Department=Department,
                start_date=start_date,
                end_date=end_date,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    finally:
        db.close()


@app.get("/api/doctor/consultations")
def doctor_consultations():
    db = SessionLocal()
    try:
        items = db.query(Consultation).order_by(Consultation.created_at.desc()).all()
        result = []
        for item in items:
            patient = db.query(User).filter(User.id == item.patient_id).first()
            document_count = db.query(MedicalDocument).filter(MedicalDocument.patient_id == item.patient_id).count()
            result.append({
                "id": item.id,
                "patient_id": item.patient_id,
                "patient_name": patient.name if patient else "Unknown",
                "title": item.title,
                "summary": item.summary,
                "status": item.status,
                "risk_level": item.risk_level or "none",
                "red_flags": json.loads(item.red_flags) if item.red_flags else [],
                "doctor_review": item.doctor_review or "Pending",
                "doctor_notes": item.doctor_notes or "",
                "ai_summary": json.loads(item.ai_summary) if item.ai_summary else None,
                "ai_summary_generated_at": item.ai_summary_generated_at.isoformat() if item.ai_summary_generated_at else None,
                "document_count": document_count,
                "created_at": item.created_at.isoformat()
            })
        return {"consultations": result}
    finally:
        db.close()
