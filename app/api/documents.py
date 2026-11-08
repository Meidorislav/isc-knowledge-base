import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.ingestion.parsing import UnsupportedFormatError
from app.ingestion.service import DocumentNotFoundError, DuplicateDocumentError, submit_document
from app.models import Document, DocumentStatus

router = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[Session, Depends(get_session)]


class DocumentOut(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    title: str
    category: str | None
    owner: str
    status: DocumentStatus
    current_version_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


# Upload handlers are sync `def` on purpose: submit_document uses a sync DB session, and FastAPI
# runs sync handlers in a threadpool. As `async def` they would block the event loop.
@router.post("", response_model=DocumentOut, status_code=202)
def upload_document(
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    owner: Annotated[str, Form()],
    title: Annotated[str | None, Form()] = None,
    category: Annotated[str | None, Form()] = None,
) -> Document:
    """Upload a new document. Indexing is asynchronous: poll GET /documents/{id} for status."""
    raw = _read_limited(file)
    return _submit(session, file=file, raw=raw, owner=owner, title=title, category=category)


@router.post("/{document_id}/versions", response_model=DocumentOut, status_code=202)
def upload_version(
    session: SessionDep,
    document_id: uuid.UUID,
    file: Annotated[UploadFile, File()],
    owner: Annotated[str, Form()],
) -> Document:
    raw = _read_limited(file)
    return _submit(session, file=file, raw=raw, owner=owner, document_id=document_id)


@router.get("", response_model=list[DocumentOut])
def list_documents(session: SessionDep) -> list[Document]:
    return list(session.scalars(select(Document).order_by(Document.updated_at.desc())))


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(session: SessionDep, document_id: uuid.UUID) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Документ не найден")
    return document


def _submit(session: Session, *, file: UploadFile, raw: bytes, **kwargs) -> Document:
    try:
        return submit_document(session, filename=file.filename or "upload.txt", raw=raw, **kwargs)
    except UnsupportedFormatError as e:
        raise HTTPException(415, str(e)) from e
    except DuplicateDocumentError as e:
        raise HTTPException(409, {"message": str(e), "document_id": str(e.document_id)}) from e
    except DocumentNotFoundError as e:
        raise HTTPException(404, "Документ не найден") from e


def _read_limited(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    raw = file.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(413, f"Файл больше {get_settings().max_upload_mb} МБ")
    if not raw:
        raise HTTPException(422, "Пустой файл")
    return raw
