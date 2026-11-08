from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api import search
from app.main import app
from app.search import Bm25Hit, RrfHit, VectorHit

client = TestClient(app)


@pytest.fixture(params=["hybrid", "bm25", "vector"])
def search_route(monkeypatch, request):
    mode = request.param
    path = "/search" if mode == "hybrid" else f"/search/{mode}"
    session = object()
    execute_search = Mock(return_value=[])
    monkeypatch.setitem(app.dependency_overrides, search.get_session, lambda: session)
    monkeypatch.setattr(search, f"search_{mode}", execute_search)
    return mode, path, execute_search, session


@pytest.mark.parametrize(("params", "expected_limit"), [({}, 10), ({"limit": 5}, 5)])
def test_search_returns_hits(search_route, params, expected_limit):
    mode, path, execute_search, session = search_route
    document_id = UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276")
    hit_type, metric, value = {
        "hybrid": (RrfHit, "rrf_score", 0.0328),
        "bm25": (Bm25Hit, "score", 2.218),
        "vector": (VectorHit, "distance", 0.402),
    }[mode]
    execute_search.return_value = [
        hit_type(1, document_id, ["Подача заявки", "Сроки"], "Ответ приходит за три дня.", value)
    ]
    query = "срок рассмотрения заявки"

    response = client.get(path, params={"q": query, **params})

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": 1,
            "document_id": str(document_id),
            "heading_path": ["Подача заявки", "Сроки"],
            "text": "Ответ приходит за три дня.",
            metric: value,
        }
    ]
    execute_search.assert_called_once_with(session, query, expected_limit)


def test_search_returns_empty_list_without_matches(search_route):
    _, path, _, _ = search_route
    response = client.get(path, params={"q": "неизвестный термин"})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("params", [{}, {"q": "заявка", "limit": 0}, {"q": "заявка", "limit": 101}])
def test_search_rejects_invalid_parameters(search_route, params):
    _, path, execute_search, _ = search_route

    response = client.get(path, params=params)

    assert response.status_code == 422
    execute_search.assert_not_called()
