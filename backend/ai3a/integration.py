"""
Minimal integration helper.

In your existing FastAPI main.py:

    from ai3a.integration import register_ai3a
    register_ai3a(app)

If you already have a shared storage/config system, pass it to register_ai3a.
"""

from fastapi import FastAPI
from .router import build_router
from .service import DocumentIntakeService

def register_ai3a(app: FastAPI, storage_dir: str = "./data/documents"):
    service = DocumentIntakeService(storage_dir=storage_dir)
    app.include_router(build_router(service))
    app.state.ai3a_document_service = service
    return service
