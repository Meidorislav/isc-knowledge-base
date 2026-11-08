"""initial schema: documents, versions, chunks, ingestion jobs

Revision ID: 0001
Revises:
Create Date: 2026-09-25
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")

    op.execute("CREATE TYPE document_status AS ENUM ('uploaded', 'processing', 'indexed', 'failed')")
    op.execute("CREATE TYPE job_status AS ENUM ('queued', 'running', 'done', 'failed')")

    op.execute("""
        CREATE TABLE documents (
            id                 uuid PRIMARY KEY,
            title              text NOT NULL,
            category           text,
            owner              text NOT NULL,
            visibility         varchar(32) NOT NULL DEFAULT 'all',
            status             document_status NOT NULL DEFAULT 'uploaded',
            current_version_id uuid,
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE document_versions (
            id           uuid PRIMARY KEY,
            document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            version      integer NOT NULL,
            filename     text NOT NULL,
            file_format  varchar(16) NOT NULL,
            content_hash varchar(64) NOT NULL,
            raw          bytea NOT NULL,
            text         text,
            created_at   timestamptz NOT NULL DEFAULT now(),
            UNIQUE (document_id, version)
        )
    """)
    op.execute("CREATE INDEX ix_document_versions_content_hash ON document_versions (content_hash)")
    op.execute("""
        ALTER TABLE documents
            ADD CONSTRAINT fk_documents_current_version
            FOREIGN KEY (current_version_id) REFERENCES document_versions(id) ON DELETE SET NULL
    """)

    op.execute("""
        CREATE TABLE chunks (
            id              bigserial PRIMARY KEY,
            document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            version_id      uuid NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
            ord             integer NOT NULL,
            heading_path    text[] NOT NULL,
            text            text NOT NULL,
            search_text     text NOT NULL,
            token_count     integer NOT NULL,
            structure       varchar(16) NOT NULL,
            embedding       vector(1024) NOT NULL,
            embedding_model text NOT NULL,
            chunker_version varchar(16) NOT NULL,
            metadata        jsonb NOT NULL DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX ix_chunks_document_id ON chunks (document_id)")
    op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
    # Real BM25 (ParadeDB pg_search) with a Russian stemmer and stopwords.
    op.execute("""
        CREATE INDEX ix_chunks_bm25 ON chunks
        USING bm25 (
            id,
            (search_text::pdb.simple('stemmer=russian', 'stopwords_language=russian'))
        )
        WITH (key_field = 'id')
    """)

    op.execute("""
        CREATE TABLE ingestion_jobs (
            id          bigserial PRIMARY KEY,
            version_id  uuid NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
            status      job_status NOT NULL DEFAULT 'queued',
            attempts    integer NOT NULL DEFAULT 0,
            last_error  text,
            created_at  timestamptz NOT NULL DEFAULT now(),
            started_at  timestamptz,
            finished_at timestamptz
        )
    """)
    # Lets the worker find queued jobs quickly without scanning finished ones.
    op.execute("CREATE INDEX ix_ingestion_jobs_queued ON ingestion_jobs (id) WHERE status = 'queued'")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ingestion_jobs")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS fk_documents_current_version")
    op.execute("DROP TABLE IF EXISTS document_versions")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TYPE IF EXISTS job_status")
    op.execute("DROP TYPE IF EXISTS document_status")
