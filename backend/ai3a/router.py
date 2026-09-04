from __future__ import annotations
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional

from .service import DocumentIntakeService

router = APIRouter(prefix="/api/documents", tags=["AI-3A Documents"])

def build_router(service: DocumentIntakeService) -> APIRouter:
    # Namespaced to avoid colliding with the legacy /api/documents routes in the existing backend.
    router = APIRouter(prefix="/api/documents/intake", tags=["AI-3A Documents"])

    @router.post("/upload")
    async def upload_document(
        patient_id: str = Form(...),
        encounter_id: str = Form(...),
        document: UploadFile = File(...),
    ):
        try:
            content = await document.read()
            record = service.create(
                patient_id=patient_id,
                encounter_id=encounter_id,
                filename=document.filename or "document",
                content_type=document.content_type or "application/octet-stream",
                content=content,
            )
            return {
                "success": True,
                "document": record.to_dict(),
                "next_stage": "OCR",
                "message": "Document accepted and ready for OCR.",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/encounter/{encounter_id}")
    async def list_documents(encounter_id: str):
        records = service.list_for_encounter(encounter_id)
        return {"documents": [r.to_dict() for r in records]}

    @router.get("/{document_id}")
    async def get_document(document_id: str):
        record = service.get(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"document": record.to_dict()}

    @router.delete("/{document_id}")
    async def delete_document(document_id: str):
        if not service.delete(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"success": True, "document_id": document_id}

    return router
