from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy.orm import Session

from app.db import get_session
from app.ingestion.embedding import EmbeddingError
from app.rag import InvalidModelAnswer, RagAnswer, answer_question

router = APIRouter(prefix="/ask", tags=["rag"])


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


@router.post("")
def ask(request: AskRequest, session: Annotated[Session, Depends(get_session)]) -> RagAnswer:
    try:
        return answer_question(session, request.question)
    except InvalidModelAnswer as exc:
        raise HTTPException(502, detail="Модель вернула некорректный ответ") from exc
    except (httpx.HTTPError, EmbeddingError) as exc:
        raise HTTPException(502, detail="Ошибка при обращении к Ollama") from exc
