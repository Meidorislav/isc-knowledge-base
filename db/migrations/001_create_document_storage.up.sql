BEGIN;

CREATE TABLE source_projects (
    id text PRIMARY KEY CHECK (id ~ '[^[:space:]]'),
    last_imported_commit_sha text
        CHECK (last_imported_commit_sha ~ '^([0-9a-f]{40}|[0-9a-f]{64})$')
);

CREATE TABLE source_documents (
    project_id text NOT NULL REFERENCES source_projects (id) ON DELETE CASCADE,
    path text NOT NULL CHECK (path ~ '[^[:space:]]'),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (project_id, path)
);

CREATE TABLE document_fragments (
    id text PRIMARY KEY CHECK (id ~ '^[0-9a-f]{64}$'),
    project_id text NOT NULL,
    path text NOT NULL,
    heading_path text[] NOT NULL,
    start_line integer NOT NULL CHECK (start_line >= 1),
    end_line integer NOT NULL CHECK (end_line >= start_line),
    text text NOT NULL CHECK (text ~ '[^[:space:]]'),
    FOREIGN KEY (project_id, path) REFERENCES source_documents (project_id, path)
        ON DELETE CASCADE
);

CREATE INDEX document_fragments_project_path_idx
    ON document_fragments (project_id, path);

COMMIT;
