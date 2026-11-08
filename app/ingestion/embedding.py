import httpx

from app.config import get_settings


class EmbeddingError(RuntimeError):
    pass


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with Ollama in batches. bge-m3 needs no query/document prefixes."""
    settings = get_settings()
    vectors: list[list[float]] = []
    with httpx.Client(base_url=settings.ollama_url, timeout=120) as client:
        for start in range(0, len(texts), settings.embed_batch_size):
            batch = texts[start : start + settings.embed_batch_size]
            response = client.post("/api/embed", json={"model": settings.embed_model, "input": batch})
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])

    for v in vectors:
        if len(v) != settings.embed_dim:
            raise EmbeddingError(
                f"{settings.embed_model} returned dim {len(v)}, expected {settings.embed_dim}"
            )
    return vectors
