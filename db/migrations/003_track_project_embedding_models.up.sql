BEGIN;

CREATE TABLE source_project_embedding_models (
    project_id text NOT NULL REFERENCES source_projects (id) ON DELETE CASCADE,
    model_id text NOT NULL REFERENCES embedding_models (id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, model_id)
);

CREATE INDEX source_project_embedding_models_model_id_idx
    ON source_project_embedding_models (model_id);

INSERT INTO source_project_embedding_models (project_id, model_id)
SELECT DISTINCT fragment.project_id, embedding.model_id
FROM fragment_embeddings AS embedding
JOIN document_fragments AS fragment ON fragment.id = embedding.fragment_id;

COMMIT;
