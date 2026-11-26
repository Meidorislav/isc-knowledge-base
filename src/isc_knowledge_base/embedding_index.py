"""Generate and store embeddings for a project's imported fragments."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import psycopg

from isc_knowledge_base.database import connect_postgres
from isc_knowledge_base.encoder import WordMeanEncoder, load_checkpoint


@dataclass(frozen=True, slots=True)
class EmbeddingIndexResult:
    stored: int
    skipped: int


def index_project_embeddings(
    connection: psycopg.Connection, project_id: str, model_id: str, encoder: WordMeanEncoder,
) -> EmbeddingIndexResult:
    with connection.transaction(), connection.cursor() as cursor:
        # Markdown imports lock the same row before replacing the project's fragments.
        cursor.execute("SELECT id FROM source_projects WHERE id = %s FOR UPDATE", (project_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"No imported project with ID {project_id!r}")

        cursor.execute(
            """
            INSERT INTO embedding_models (id, dimension) VALUES (%s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (model_id, encoder.dimension),
        )
        cursor.execute(
            """
            INSERT INTO source_project_embedding_models (project_id, model_id)
            VALUES (%s, %s)
            ON CONFLICT (project_id, model_id) DO NOTHING
            """,
            (project_id, model_id),
        )
        cursor.execute(
            """
            SELECT fragment.id, fragment.heading_path, fragment.text
            FROM document_fragments AS fragment
            WHERE fragment.project_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM fragment_embeddings AS existing
                  WHERE existing.model_id = %s AND existing.fragment_id = fragment.id
              )
            """,
            (project_id, model_id),
        )
        pending = cursor.fetchall()
        records = []
        for fragment_id, heading_path, text in pending:
            vector = encoder.encode("\n".join([*heading_path, text]))
            if vector is not None:
                records.append((fragment_id, model_id, vector.tolist()))

        cursor.executemany(
            """
            INSERT INTO fragment_embeddings (fragment_id, model_id, embedding)
            VALUES (%s, %s, %s::double precision[]::vector)
            """,
            records,
        )

    return EmbeddingIndexResult(stored=len(records), skipped=len(pending) - len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to the trained encoder checkpoint")
    parser.add_argument("--project-id", required=True, help="identifier of an imported project")
    args = parser.parse_args()
    try:
        model_id, encoder = load_checkpoint(args.checkpoint)
        with connect_postgres() as connection:
            result = index_project_embeddings(connection, args.project_id.strip(), model_id, encoder)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except psycopg.Error as error:
        parser.error(error.diag.message_primary or str(error))

    print(f"Model {model_id} ({encoder.dimension} dimensions).")
    print(f"Embeddings: {result.stored} stored, {result.skipped} skipped (no known tokens).")


if __name__ == "__main__":
    main()
