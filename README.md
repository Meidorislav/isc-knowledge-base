# isc-knowledge-base

**[EN](README.md) | [RU](README.ru.md)**

> Hybrid RAG search over a corporate knowledge base: local LLM, anti-hallucination loop, and gap analytics.

Diploma project (final qualifying work): "Development of an intelligent search and answer generation platform based on Retrieval-Augmented Generation with an anti-hallucination loop for the enterprise LLC 'ISC'".

## What the system does

- **Hybrid RAG search** — an employee asks a question in plain language; the system searches documents using a hybrid approach (BM25 + vector search), reranks the candidates, and uses a local LLM to produce a short answer with a source reference. Strictly grounded in company documents — if there's no answer, it honestly says "I don't know."
- **Gap analytics** — logs questions with no answer found, so those responsible for the knowledge base know what to add.
- **Authorization** — integrates with the company's existing auth service (SSO); the exact scheme is still being clarified, a mock user is used during development.

Everything runs **locally**, within the company's infrastructure — employee questions and the content of internal regulations never leave to external APIs.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python / FastAPI (single process: search, RAG, analytics) |
| Storage | PostgreSQL + pgvector |
| Full-text search | Postgres FTS |
| LLM | Ollama (Qwen2.5 / Llama 3.1, quantized) |
| Embeddings | bge-m3 / nomic-embed-text (via Ollama) |
| Reranker | bge-reranker (cross-encoder) |
| Frontend | TypeScript + React |
| CI/CD | GitHub Actions |

## License

MIT
