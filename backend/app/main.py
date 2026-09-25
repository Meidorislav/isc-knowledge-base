from fastapi import FastAPI

from app.api import documents

app = FastAPI(title="ISC Knowledge Base", version="0.1.0")
app.include_router(documents.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
