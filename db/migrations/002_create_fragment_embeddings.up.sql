BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embedding_models (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    dimension integer NOT NULL CHECK (dimension BETWEEN 1 AND 16000),
    UNIQUE (id, dimension)
);

CREATE TABLE fragment_embeddings (
    fragment_id text NOT NULL REFERENCES document_fragments (id) ON DELETE CASCADE,
    model_id text NOT NULL,
    embedding vector NOT NULL,
    dimension integer GENERATED ALWAYS AS (vector_dims(embedding)) STORED,
    PRIMARY KEY (model_id, fragment_id),
    FOREIGN KEY (model_id, dimension) REFERENCES embedding_models (id, dimension)
        ON DELETE CASCADE
);

CREATE INDEX fragment_embeddings_fragment_id_idx ON fragment_embeddings (fragment_id);

COMMIT;
