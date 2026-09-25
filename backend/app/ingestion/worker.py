"""Ingestion worker: `python -m app.ingestion.worker`."""

import logging
import time

import httpx
from sqlalchemy.exc import OperationalError

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion import queue
from app.ingestion.pipeline import process_version
from app.models import Document, DocumentStatus, DocumentVersion

log = logging.getLogger(__name__)

# Transient failures worth retrying; anything else (bad file, empty text) fails immediately.
RETRYABLE = (httpx.HTTPError, OperationalError)


def run_once() -> bool:
    """Process one job. Returns False if the queue was empty."""
    settings = get_settings()
    with SessionLocal() as session:
        job = queue.claim_next(session)
        if job is None:
            return False
        try:
            process_version(session, job.version_id)
            queue.mark_done(session, job.id)
            session.commit()
        except Exception as exc:
            session.rollback()
            retry = isinstance(exc, RETRYABLE) and job.attempts < settings.job_max_attempts
            log.exception("job %s failed (attempt %s, retry=%s)", job.id, job.attempts, retry)
            queue.mark_failed(session, job.id, f"{type(exc).__name__}: {exc}", retry=retry)
            if not retry:
                _mark_document_failed(session, job.version_id)
            session.commit()
    return True


def _mark_document_failed(session, version_id) -> None:
    version = session.get(DocumentVersion, version_id)
    if version is not None:
        session.get(Document, version.document_id).status = DocumentStatus.FAILED


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    poll = get_settings().worker_poll_seconds
    log.info("ingestion worker started")
    while True:
        if not run_once():
            time.sleep(poll)


if __name__ == "__main__":
    main()
