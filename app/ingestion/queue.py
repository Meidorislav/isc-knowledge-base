"""Job queue on top of Postgres: no separate broker needed for a single-process app."""

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import IngestionJob


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    version_id: uuid.UUID
    attempts: int


def enqueue(session: Session, version_id: uuid.UUID) -> IngestionJob:
    job = IngestionJob(version_id=version_id)
    session.add(job)
    return job


def claim_next(session: Session) -> ClaimedJob | None:
    """Atomically take one queued job. SKIP LOCKED lets several workers run safely."""
    row = session.execute(
        text("""
            UPDATE ingestion_jobs
               SET status = 'running', started_at = now(), attempts = attempts + 1
             WHERE id = (
                   SELECT id FROM ingestion_jobs
                    WHERE status = 'queued'
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1)
            RETURNING id, version_id, attempts
        """)
    ).one_or_none()
    session.commit()
    return ClaimedJob(*row) if row else None


def mark_done(session: Session, job_id: int) -> None:
    session.execute(
        text("UPDATE ingestion_jobs SET status = 'done', finished_at = now() WHERE id = :id"),
        {"id": job_id},
    )


def mark_failed(session: Session, job_id: int, error: str, *, retry: bool) -> None:
    session.execute(
        text("""
            UPDATE ingestion_jobs
               SET status = CAST(:status AS job_status), last_error = :error, finished_at = now()
             WHERE id = :id
        """),
        {"id": job_id, "error": error, "status": "queued" if retry else "failed"},
    )
