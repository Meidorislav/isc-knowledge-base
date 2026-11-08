from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_session
from app.search import Bm25Hit, VectorHit, search_bm25, search_vector

router = APIRouter(prefix="/search", tags=["search"])
SearchSession = Annotated[Session, Depends(get_session)]
SearchLimit = Annotated[int, Query(ge=1, le=100)]


@router.get("/bm25")
def search_chunks_bm25(
    session: SearchSession,
    q: str,
    limit: SearchLimit = 10,
) -> list[Bm25Hit]:
    return search_bm25(session, q, limit)


@router.get("/vector")
def search_chunks_vector(
    session: SearchSession,
    q: str,
    limit: SearchLimit = 10,
) -> list[VectorHit]:
    return search_vector(session, q, limit)
