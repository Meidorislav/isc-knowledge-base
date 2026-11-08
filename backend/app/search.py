from dataclasses import dataclass
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion.embedding import embed
from app.models import EMBED_DIM

RRF_K = 60
RRF_CANDIDATE_LIMIT = 30


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


@dataclass(frozen=True)
class RrfHit:
    chunk_id: int
    document_id: UUID
    heading_path: list[str]
    text: str
    rrf_score: float


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


def fuse_rrf(bm25_hits: list[Bm25Hit], vector_hits: list[VectorHit], limit: int) -> list[RrfHit]:
    if limit <= 0:
        return []

    hits_by_id: dict[int, Bm25Hit | VectorHit] = {}
    scores: dict[int, float] = {}
    for ranking in (bm25_hits, vector_hits):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk_id
            hits_by_id.setdefault(chunk_id, hit)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)

    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    results: list[RrfHit] = []
    for chunk_id in ranked_ids[:limit]:
        hit = hits_by_id[chunk_id]
        results.append(
            RrfHit(
                chunk_id=chunk_id,
                document_id=hit.document_id,
                heading_path=hit.heading_path,
                text=hit.text,
                rrf_score=scores[chunk_id],
            )
        )
    return results


def search_hybrid(session: Session, query: str, limit: int) -> list[RrfHit]:
    query = query.strip()
    if not query or limit <= 0:
        return []

    candidate_limit = max(RRF_CANDIDATE_LIMIT, limit)
    bm25_hits = search_bm25(session, query, candidate_limit)
    vector_hits = search_vector(session, query, candidate_limit)
    return fuse_rrf(bm25_hits, vector_hits, limit)
