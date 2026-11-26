import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql

from isc_knowledge_base.markdown_import import MarkdownSource
from isc_knowledge_base.postgres_import import sync_project


POSTGRES_TEST_DSN = os.getenv("POSTGRES_TEST_DSN")
MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"


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
