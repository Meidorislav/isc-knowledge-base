import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

import numpy as np
import psycopg
from psycopg import sql

from isc_knowledge_base.embedding_index import index_project_embeddings
from isc_knowledge_base.encoder import WordMeanEncoder
from isc_knowledge_base.markdown_import import MarkdownSource
from isc_knowledge_base.postgres_import import sync_project
from isc_knowledge_base.semantic_search import search_fragments


POSTGRES_TEST_DSN = os.getenv("POSTGRES_TEST_DSN")
MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"


class FailingEncoder(WordMeanEncoder):
    def encode(self, text: str) -> None:
        raise ValueError("Test encoder failure")


@unittest.skipUnless(POSTGRES_TEST_DSN, "set POSTGRES_TEST_DSN to run PostgreSQL integration tests")
class PostgresImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = psycopg.connect(POSTGRES_TEST_DSN, autocommit=True)
        self.addCleanup(self.connection.close)
        schema = sql.Identifier(f"test_import_{uuid4().hex}")
        self.connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        self.addCleanup(
            self.connection.execute, sql.SQL("DROP SCHEMA {} CASCADE").format(schema),
        )
        self.connection.execute(sql.SQL("SET search_path TO {}, public").format(schema))
        for migration in sorted(MIGRATIONS.glob("*.up.sql")):
            self.connection.execute(migration.read_text(encoding="utf-8"))
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_empty_import_requires_explicit_permission(self) -> None:
        document = self.root / "travel.md"
        document.write_text("# Travel\n\nKeep receipts.\n", encoding="utf-8")
        first = MarkdownSource(self.root, project_id="demo", commit_sha="a" * 40)
        self.assertEqual(sync_project(self.connection, first).added, 1)
        document.unlink()

        empty = MarkdownSource(self.root, project_id="demo", commit_sha="b" * 40)
        with self.assertRaisesRegex(ValueError, "--allow-empty"):
            sync_project(self.connection, empty)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM document_fragments").fetchone(), (1,))
        self.assertEqual(
            self.connection.execute("SELECT last_imported_commit_sha FROM source_projects").fetchone(),
            ("a" * 40,),
        )

        self.assertEqual(sync_project(self.connection, empty, allow_empty=True).deleted, 1)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM document_fragments").fetchone(), (0,))

    def test_texts_vectors_and_commit_are_published_together(self) -> None:
        document = self.root / "travel.md"
        document.write_text("# Travel\n\nFirst.\n", encoding="utf-8")
        words = ["travel", "first", "next"]
        weights = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        encoder = WordMeanEncoder(words, weights)
        first_model = "a" * 64
        second_model = "b" * 64
        original_sha = "a" * 40
        next_sha = "b" * 40
        first = MarkdownSource(self.root, project_id="demo", commit_sha=original_sha)
        sync_project(self.connection, first, encoders={first_model: encoder})
        index_project_embeddings(self.connection, "demo", second_model, encoder)

        document.write_text("# Travel\n\nNext.\n", encoding="utf-8")
        updated = MarkdownSource(self.root, project_id="demo", commit_sha=next_sha)

        def snapshot():
            return self.connection.execute(
                """
                SELECT p.last_imported_commit_sha, f.text, e.model_id, e.embedding::text
                FROM source_projects p
                JOIN document_fragments f ON f.project_id = p.id
                JOIN fragment_embeddings e ON e.fragment_id = f.id
                ORDER BY e.model_id
                """
            ).fetchall()

        original = [(original_sha, "First.", model, "[1,0]") for model in (first_model, second_model)]
        with self.assertRaisesRegex(ValueError, "--checkpoint"):
            sync_project(self.connection, updated, encoders={first_model: encoder})
        self.assertEqual(snapshot(), original)

        with self.assertRaisesRegex(ValueError, "Test encoder failure"):
            sync_project(
                self.connection, updated,
                encoders={first_model: encoder, second_model: FailingEncoder(words, weights)},
            )
        self.assertEqual(snapshot(), original)

        self.assertEqual(
            sync_project(self.connection, updated, encoders={first_model: encoder, second_model: encoder}).updated,
            1,
        )
        self.assertEqual(snapshot(), [(next_sha, "Next.", model, "[0,1]") for model in (first_model, second_model)])

    def test_search_ranks_and_filters_by_project_and_model(self) -> None:
        corpus = self.root / "corpus"
        corpus.mkdir()
        for name, text in (("b.md", "alpha"), ("a.md", "alpha"), ("c.md", "beta")):
            (corpus / name).write_text(f"# Expenses\n\n{text}\n", encoding="utf-8")
        other = self.root / "other"
        other.mkdir()
        (other / "high.md").write_text("# Other\n\npriority\n", encoding="utf-8")

        words = ["query", "alpha", "beta", "priority"]
        weights = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [9.0, 0.0]])
        encoder = WordMeanEncoder(words, weights)
        wide_encoder = WordMeanEncoder(words, np.column_stack((weights, np.ones(len(words)))))
        model_id = "c" * 64
        commit_sha = "d" * 40
        source = MarkdownSource(corpus, project_id="demo", commit_sha=commit_sha)
        sync_project(self.connection, source, encoders={model_id: encoder, "d" * 64: wide_encoder})
        sync_project(
            self.connection, MarkdownSource(other, project_id="other"), encoders={model_id: encoder},
        )

        search_options = {"project_id": "demo", "model_id": model_id, "encoder": encoder}
        hits = search_fragments(self.connection, **search_options, query="query")
        self.assertEqual([(hit.source.path, hit.score) for hit in hits], [("a.md", 2.0), ("b.md", 2.0), ("c.md", -1.0)])
        self.assertTrue(hits[0].id)
        self.assertEqual(hits[0].text, "alpha")
        self.assertEqual(hits[0].source.project_id, "demo")
        self.assertEqual(hits[0].source.heading_path, ("Expenses",))
        self.assertEqual(hits[0].source.commit_sha, commit_sha)
        self.assertEqual((hits[0].source.start_line, hits[0].source.end_line), (1, 3))
        self.assertEqual(search_fragments(self.connection, **search_options, query="query", limit=2), hits[:2])
        self.assertEqual(search_fragments(self.connection, **search_options, query="unknown"), [])
