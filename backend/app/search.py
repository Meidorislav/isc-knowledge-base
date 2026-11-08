from dataclasses import dataclass
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion.embedding import embed
from app.models import EMBED_DIM


@dataclass(frozen=True)
class Bm25Hit:
    chunk_id: int
    document_id: UUID
    heading_path: list[str]
    text: str
    score: float


@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    document_id: UUID
    heading_path: list[str]
    text: str
    distance: float


def search_bm25(session: Session, query: str, limit: int) -> list[Bm25Hit]:
    query = query.strip()
    if not query or limit <= 0:
        return []

    statement = text("""
        SELECT id AS chunk_id, document_id, heading_path, text, pdb.score(id) AS score
        FROM chunks
        WHERE search_text ||| :query
        ORDER BY score DESC
        LIMIT :limit
    """)
    rows = session.execute(statement, {"query": query, "limit": limit}).mappings()
    return [Bm25Hit(**row) for row in rows]


def search_vector(session: Session, query: str, limit: int) -> list[VectorHit]:
    query = query.strip()
    if not query or limit <= 0:
        return []

    [query_vector] = embed([query])
    statement = text("""
        SELECT id AS chunk_id, document_id, heading_path, text,
               embedding <=> CAST(:query_vector AS vector) AS distance
        FROM chunks
        WHERE embedding_model = :model
        ORDER BY distance ASC
        LIMIT :limit
    """).bindparams(bindparam("query_vector", type_=Vector(EMBED_DIM)))
    params = {"query_vector": query_vector, "model": get_settings().embed_model, "limit": limit}
    rows = session.execute(statement, params).mappings()
    return [VectorHit(**row) for row in rows]
