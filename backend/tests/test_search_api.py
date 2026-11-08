from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api import search
from app.main import app
from app.search import Bm25Hit

client = TestClient(app)


@pytest.fixture
def bm25(monkeypatch):
    session = object()
    execute_search = Mock(return_value=[])
    monkeypatch.setitem(app.dependency_overrides, search.get_session, lambda: session)
    monkeypatch.setattr(search, "search_bm25", execute_search)
    return execute_search, session


@pytest.mark.parametrize(("params", "expected_limit"), [({}, 10), ({"limit": 5}, 5)])
def test_search_returns_bm25_hits(bm25, params, expected_limit):
    execute_search, session = bm25
    document_id = UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276")
    execute_search.return_value = [
        Bm25Hit(1, document_id, ["Подача заявки", "Сроки"], "Ответ приходит за три дня.", 2.218)
    ]
    query = "срок рассмотрения заявки"

    response = client.get("/search", params={"q": query, **params})

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": 1,
            "document_id": str(document_id),
            "heading_path": ["Подача заявки", "Сроки"],
            "text": "Ответ приходит за три дня.",
            "score": 2.218,
        }
    ]
    execute_search.assert_called_once_with(session, query, expected_limit)


def test_search_returns_empty_list_without_matches(bm25):
    response = client.get("/search", params={"q": "неизвестный термин"})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("params", [{}, {"q": "заявка", "limit": 0}, {"q": "заявка", "limit": 101}])
def test_search_rejects_invalid_parameters(bm25, params):
    execute_search, _ = bm25

    response = client.get("/search", params=params)

    assert response.status_code == 422
    execute_search.assert_not_called()
