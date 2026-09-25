"""Worker-side processing of one document version: parse -> chunk -> embed -> swap chunks."""

import logging
import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion.chunking import CHUNKER_VERSION, chunk_document
from app.ingestion.embedding import embed
from app.ingestion.parsing import parse
from app.models import Chunk, Document, DocumentStatus, DocumentVersion

log = logging.getLogger(__name__)


def process_version(session: Session, version_id: uuid.UUID) -> None:
    version = session.get(DocumentVersion, version_id)
    if version is None:
        raise LookupError(f"version {version_id} not found")
    document = session.get(Document, version.document_id)

    if _is_stale(session, document, version):
        log.info("skip stale version %s of document %s", version.version, document.id)
        return

    document.status = DocumentStatus.PROCESSING
    session.commit()

    version.text = parse(version.raw, version.file_format)
    drafts = chunk_document(version.text, document.title)
    if not drafts:
        raise ValueError("В документе не найдено текста")

    vectors = embed([d.search_text for d in drafts])
    model = get_settings().embed_model

    # One transaction: search never sees a half-updated document.
    session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    session.add_all(
        Chunk(
            document_id=document.id,
            version_id=version.id,
            ord=i,
            heading_path=d.heading_path,
            text=d.text,
            search_text=d.search_text,
            token_count=d.token_count,
            structure=d.structure,
            embedding=vec,
            embedding_model=model,
            chunker_version=CHUNKER_VERSION,
        )
        for i, (d, vec) in enumerate(zip(drafts, vectors, strict=True))
    )
    document.current_version_id = version.id
    document.status = DocumentStatus.INDEXED
    session.commit()
    log.info("indexed document %s v%s: %d chunks", document.id, version.version, len(drafts))


def _is_stale(session: Session, document: Document, version: DocumentVersion) -> bool:
    """A newer version is already indexed (e.g. jobs finished out of order)."""
    if document.current_version_id is None:
        return False
    current = session.get(DocumentVersion, document.current_version_id)
    return current is not None and current.version > version.version
