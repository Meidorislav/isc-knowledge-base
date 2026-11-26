"""Search indexed Markdown fragments using PostgreSQL and pgvector."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import psycopg

from isc_knowledge_base.database import connect_postgres
from isc_knowledge_base.encoder import WordMeanEncoder, load_checkpoint
from isc_knowledge_base.markdown_import import FragmentSource


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    text: str
    source: FragmentSource


def search_fragments(
    connection: psycopg.Connection,
    *,
    project_id: str,
    model_id: str,
    encoder: WordMeanEncoder,
    query: str,
    limit: int = 5,
) -> list[SearchHit]:
    project_id = project_id.strip()
    if not project_id:
        raise ValueError("Project ID must not be empty")
    if not query.strip():
        raise ValueError("Query must not be empty")
    if limit < 1:
        raise ValueError("Limit must be positive")

    query_vector = encoder.encode(query)
    if query_vector is None:
        return []

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT fragment.id, fragment.text, fragment.path, fragment.heading_path,
                   fragment.start_line, fragment.end_line, project.last_imported_commit_sha,
                   embedding.embedding <#> %s::double precision[]::vector AS distance
            FROM fragment_embeddings AS embedding
            JOIN document_fragments AS fragment ON fragment.id = embedding.fragment_id
            JOIN source_projects AS project ON project.id = fragment.project_id
            WHERE fragment.project_id = %s AND embedding.model_id = %s
            ORDER BY distance, fragment.path, fragment.start_line, fragment.id
            LIMIT %s
            """,
            (query_vector.tolist(), project_id, model_id, limit),
        )
        return [
            SearchHit(
                id=fragment_id,
                score=-distance,
                text=text,
                source=FragmentSource(
                    project_id=project_id,
                    path=path,
                    commit_sha=commit_sha,
                    heading_path=tuple(heading_path),
                    start_line=start_line,
                    end_line=end_line,
                ),
            )
            for fragment_id, text, path, heading_path, start_line, end_line, commit_sha, distance in cursor
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to the trained encoder checkpoint")
    parser.add_argument("--project-id", required=True, help="identifier of an indexed project")
    parser.add_argument("--query", required=True, help="text to search for")
    parser.add_argument("--limit", type=int, default=5, help="maximum number of hits (default: 5)")
    args = parser.parse_args()
    try:
        model_id, encoder = load_checkpoint(args.checkpoint)
        with connect_postgres() as connection:
            hits = search_fragments(
                connection,
                project_id=args.project_id,
                model_id=model_id,
                encoder=encoder,
                query=args.query,
                limit=args.limit,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except psycopg.Error as error:
        parser.error(error.diag.message_primary or str(error))

    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
