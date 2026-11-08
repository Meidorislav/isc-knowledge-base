from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class Bm25Hit:
    chunk_id: int
    document_id: UUID
    heading_path: list[str]
    text: str
    score: float


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
