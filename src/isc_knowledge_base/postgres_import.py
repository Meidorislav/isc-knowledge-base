"""Synchronize a local Markdown tree with PostgreSQL."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import psycopg

from isc_knowledge_base.database import connect_postgres
from isc_knowledge_base.markdown_import import MarkdownSource


@dataclass(frozen=True, slots=True)
class ImportResult:
    added: int
    updated: int
    unchanged: int
    deleted: int


def sync_project(connection: psycopg.Connection, source: MarkdownSource) -> ImportResult:
    added = updated = unchanged = 0
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

        for document in source.documents():
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

        if previous_hashes:
            cursor.execute(
                "DELETE FROM source_documents WHERE project_id = %s AND path = ANY(%s)",
                (source.project_id, list(previous_hashes)),
            )

    return ImportResult(added, updated, unchanged, len(previous_hashes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root of the local Markdown tree")
    parser.add_argument("--project-id", required=True, help="stable identifier of the source project")
    parser.add_argument("--commit-sha", help="full commit SHA of the source snapshot, if known")
    args = parser.parse_args()
    try:
        source = MarkdownSource(args.root, project_id=args.project_id, commit_sha=args.commit_sha)
        with connect_postgres() as connection:
            result = sync_project(connection, source)
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
