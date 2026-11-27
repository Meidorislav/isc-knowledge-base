import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import psycopg

from isc_knowledge_base.api import app
from isc_knowledge_base.encoder import WordMeanEncoder
from isc_knowledge_base.markdown_import import FragmentSource
from isc_knowledge_base.semantic_search import SearchHit


class SearchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = Path(self.enterContext(TemporaryDirectory()))
        checkpoint = directory / "encoder.npz"
        WordMeanEncoder(["hotel"], np.ones((1, 2))).save(checkpoint)
        self.enterContext(patch.dict(os.environ, {"ENCODER_CHECKPOINT_PATH": str(checkpoint)}))
        self.client = self.enterContext(TestClient(app))

    def test_returns_ranked_hits_with_sources_and_no_store(self) -> None:
        hit = SearchHit(
            id="fragment",
            score=2.0,
            text="Keep the invoice.",
            source=FragmentSource("demo", "travel.md", None, ("Hotel",), 2, 4),
        )
        with (
            patch("isc_knowledge_base.api.connect_postgres") as database,
            patch("isc_knowledge_base.api.search_fragments", return_value=[hit]) as search,
        ):
            response = self.client.post("/search", json={"project_id": "demo", "query": " hotel "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.json()[0]["source"], {
            "project_id": "demo", "path": "travel.md", "commit_sha": None,
            "heading_path": ["Hotel"], "start_line": 2, "end_line": 4,
        })
        search.assert_called_once()
        self.assertIs(search.call_args.args[0], database.return_value.__enter__.return_value)
        self.assertEqual(search.call_args.kwargs["query"], "hotel")
        self.assertEqual(search.call_args.kwargs["limit"], 5)

    def test_validation_errors_exclude_input_values(self) -> None:
        with patch("isc_knowledge_base.api.connect_postgres") as database:
            response = self.client.post(
                "/search",
                json={"project_id": "demo", "query": {"text": "private test input"}, "limit": 101},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual({tuple(error["loc"]) for error in response.json()["detail"]}, {
            ("body", "query"), ("body", "limit"),
        })
        self.assertNotIn("private test input", response.text)
        database.assert_not_called()

    def test_database_failure_has_a_generic_response(self) -> None:
        with patch("isc_knowledge_base.api.connect_postgres", side_effect=psycopg.OperationalError("test details")):
            response = self.client.post("/search", json={"project_id": "demo", "query": "hotel"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.json(), {"detail": "Search database unavailable"})
