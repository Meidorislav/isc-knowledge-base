"""FastAPI interface for semantic search."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import psycopg
from pydantic import BaseModel, Field, StringConstraints

from isc_knowledge_base.database import connect_postgres
from isc_knowledge_base.encoder import load_checkpoint
from isc_knowledge_base.semantic_search import SearchHit, search_fragments


NO_STORE = {"Cache-Control": "no-store"}
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchRequest(BaseModel):
    project_id: NonEmptyText
    query: NonEmptyText
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    load_dotenv(Path.cwd() / ".env")
    checkpoint = os.getenv("ENCODER_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError("Set ENCODER_CHECKPOINT_PATH to a trained encoder checkpoint")
    app.state.checkpoint = load_checkpoint(Path(checkpoint))
    yield


app = FastAPI(title="ISC Knowledge Base", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    # The default validation errors include the raw request input.
    details = [
        {key: entry[key] for key in ("loc", "msg", "type")}
        for entry in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details}, headers=NO_STORE)


@app.post("/search")
def search(payload: SearchRequest, request: Request, response: Response) -> list[SearchHit]:
    model_id, encoder = request.app.state.checkpoint
    try:
        with connect_postgres() as connection:
            hits = search_fragments(
                connection,
                project_id=payload.project_id,
                model_id=model_id,
                encoder=encoder,
                query=payload.query,
                limit=payload.limit,
            )
    except psycopg.OperationalError as error:
        raise HTTPException(
            status_code=503, detail="Search database unavailable", headers=NO_STORE,
        ) from error
    except psycopg.Error as error:
        raise HTTPException(
            status_code=500, detail="Search failed", headers=NO_STORE,
        ) from error

    response.headers.update(NO_STORE)
    return hits
