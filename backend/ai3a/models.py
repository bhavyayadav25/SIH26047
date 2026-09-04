from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY_FOR_OCR = "READY_FOR_OCR"
    FAILED = "FAILED"

@dataclass
class DocumentRecord:
    document_id: str
    encounter_id: str
    patient_id: str
    original_filename: str
    stored_filename: str
    mime_type: str
    extension: str
    file_size: int
    status: DocumentStatus
    document_type: str = "Unknown"
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        return asdict(self)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex}"
