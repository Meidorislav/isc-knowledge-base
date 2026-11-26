"""Synchronize a local Markdown tree with PostgreSQL."""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import psycopg

from isc_knowledge_base.database import connect_postgres
from isc_knowledge_base.embedding_index import index_project_embeddings
from isc_knowledge_base.encoder import WordMeanEncoder, load_checkpoint
from isc_knowledge_base.markdown_import import MarkdownSource


@dataclass(frozen=True, slots=True)
class ImportResult:
    added: int
    updated: int
    unchanged: int
    deleted: int


def sync_project(
    connection: psycopg.Connection,
    source: MarkdownSource,
    *,
    allow_empty: bool = False,
    encoders: Mapping[str, WordMeanEncoder] | None = None,
) -> ImportResult:
    added = updated = unchanged = 0
    encoders = encoders or {}
    with connection.transaction(), connection.cursor() as cursor:
        # The project upsert locks this project row until the import commits.
        cursor.execute(
            """
            INSERT INTO source_projects (id, last_imported_commit_sha)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE
            SET last_imported_commit_sha = EXCLUDED.last_imported_commit_sha
            """,
            (source.project_id, source.commit_sha),
        )
        cursor.execute(
            "SELECT path, content_sha256 FROM source_documents WHERE project_id = %s",
            (source.project_id,),
        )
        previous_hashes = dict(cursor.fetchall())

        found_document = False
        for document in source.documents():
            found_document = True
            content_hash = sha256(document.content).hexdigest()
            previous_hash = previous_hashes.pop(document.path, None)
            if previous_hash == content_hash:
                unchanged += 1
                continue

            cursor.execute(
                """
                INSERT INTO source_documents (project_id, path, content_sha256)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id, path) DO UPDATE
                SET content_sha256 = EXCLUDED.content_sha256
                """,
                (source.project_id, document.path, content_hash),
            )
            if previous_hash is None:
                added += 1
            else:
                cursor.execute(
                    "DELETE FROM document_fragments WHERE project_id = %s AND path = %s",
                    (source.project_id, document.path),
                )
                updated += 1

            cursor.executemany(
                """
                INSERT INTO document_fragments
                    (id, project_id, path, heading_path, start_line, end_line, text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    (
                        fragment.id,
                        source.project_id,
                        document.path,
                        list(fragment.source.heading_path),
                        fragment.source.start_line,
                        fragment.source.end_line,
                        fragment.text,
                    )
                    for fragment in source.fragments(document)
                ),
            )

        if not found_document and not allow_empty:
            raise ValueError("No Markdown files found. Pass --allow-empty to import an empty snapshot.")

        if previous_hashes:
            cursor.execute(
                "DELETE FROM source_documents WHERE project_id = %s AND path = ANY(%s)",
                (source.project_id, list(previous_hashes)),
            )

        if added or updated:
            cursor.execute(
                "SELECT model_id FROM source_project_embedding_models WHERE project_id = %s",
                (source.project_id,),
            )
            missing_models = {model_id for (model_id,) in cursor} - set(encoders)
            if missing_models:
                raise ValueError(
                    "Supply --checkpoint for the project's indexed models: "
                    + ", ".join(sorted(missing_models))
                )

        for model_id in sorted(encoders):
            index_project_embeddings(connection, source.project_id, model_id, encoders[model_id])

    return ImportResult(added, updated, unchanged, len(previous_hashes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root of the local Markdown tree")
    parser.add_argument("--project-id", required=True, help="stable identifier of the source project")
    parser.add_argument("--commit-sha", help="full commit SHA of the source snapshot, if known")
    parser.add_argument("--allow-empty", action="store_true", help="allow clearing a project with an empty Markdown tree")
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=[],
        help="index with this encoder in the same transaction; repeat for each indexed model",
    )
    args = parser.parse_args()
    try:
        source = MarkdownSource(args.root, project_id=args.project_id, commit_sha=args.commit_sha)
        encoders = dict(load_checkpoint(path) for path in args.checkpoint)
        with connect_postgres() as connection:
            result = sync_project(connection, source, allow_empty=args.allow_empty, encoders=encoders)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except psycopg.Error as error:
        parser.error(error.diag.message_primary or str(error))

    print(
        f"Markdown files: {result.added} added, {result.updated} updated, "
        f"{result.unchanged} unchanged, {result.deleted} deleted."
    )


if __name__ == "__main__":
    main()
