"""Single entry point for adding documents. Both the upload API and the bulk-import CLI
go through submit_document(), so they behave identically."""

import hashlib
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ingestion import queue
from app.ingestion.parsing import detect_format
from app.models import Document, DocumentStatus, DocumentVersion


class DuplicateDocumentError(ValueError):
    def __init__(self, document_id: uuid.UUID):
        super().__init__(f"Такой файл уже загружен (документ {document_id})")
        self.document_id = document_id


class DocumentNotFoundError(LookupError):
    pass


def submit_document(
    session: Session,
    *,
    filename: str,
    raw: bytes,
    owner: str,
    title: str | None = None,
    category: str | None = None,
    document_id: uuid.UUID | None = None,
) -> Document:
    """Store the file and queue it for ingestion. With document_id, adds a new version."""
    file_format = detect_format(filename)
    content_hash = hashlib.sha256(raw).hexdigest()

    existing = session.scalar(
        select(DocumentVersion.document_id).where(DocumentVersion.content_hash == content_hash)
    )
    if existing is not None:
        raise DuplicateDocumentError(existing)

    if document_id is None:
        document = Document(title=title or _title_from_filename(filename), owner=owner, category=category)
        session.add(document)
        session.flush()
        version_no = 1
    else:
        document = session.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        version_no = 1 + (
            session.scalar(
                select(func.max(DocumentVersion.version)).where(DocumentVersion.document_id == document_id)
            )
            or 0
        )

    version = DocumentVersion(
        document_id=document.id,
        version=version_no,
        filename=filename,
        file_format=file_format,
        content_hash=content_hash,
        raw=raw,
    )
    session.add(version)
    session.flush()

    document.status = DocumentStatus.UPLOADED
    queue.enqueue(session, version.id)
    session.commit()
    return document


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("_", " ").strip() or filename
