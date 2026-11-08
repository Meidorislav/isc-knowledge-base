from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_session
from app.search import Bm25Hit, search_bm25

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
def search_chunks(
    session: Annotated[Session, Depends(get_session)],
    q: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[Bm25Hit]:
    return search_bm25(session, q, limit)
