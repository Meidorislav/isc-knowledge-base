from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from app import rag
from app.db import get_session
from app.main import app
from app.rag import ModelAnswer
from app.search import RrfHit

client = TestClient(app)
DOCUMENT_ID = UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276")


@pytest.fixture
def ask_services(monkeypatch):
    session = object()
    hit = RrfHit(42, DOCUMENT_ID, ["Подача заявки", "Сроки"], "Ответ приходит за три дня.", 0.0328)
    search = Mock(return_value=[hit])
    generate = Mock(return_value=ModelAnswer(answer="В течение трёх дней.", citations=[1]))
    monkeypatch.setitem(app.dependency_overrides, get_session, lambda: session)
    monkeypatch.setattr(rag, "search_hybrid", search)
    monkeypatch.setattr(rag, "generate_answer", generate)
    return session, search, generate


def test_ask_returns_answer_with_resolved_source(ask_services):
    session, search, generate = ask_services

    response = client.post("/ask", json={"question": " Когда ответят на заявку? "})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "В течение трёх дней.",
        "sources": [
            {
                "number": 1,
                "document_id": str(DOCUMENT_ID),
                "chunk_id": 42,
                "heading_path": ["Подача заявки", "Сроки"],
                "text": "Ответ приходит за три дня.",
            }
        ],
    }
    search.assert_called_once_with(session, "Когда ответят на заявку?", limit=3)
    generate.assert_called_once()


def test_ask_returns_refusal(ask_services):
    _, _, generate = ask_services
    generate.return_value = ModelAnswer(answer=None, citations=[])

    response = client.post("/ask", json={"question": "Как приготовить борщ?"})

    assert response.status_code == 200
    assert response.json() == {"answer": None, "sources": []}


def test_ask_rejects_out_of_range_citation(ask_services):
    _, _, generate = ask_services
    generate.return_value = ModelAnswer(answer="Три дня.", citations=[2])

    response = client.post("/ask", json={"question": "Когда ответят?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Модель вернула некорректный ответ"}


def test_ask_reports_ollama_connection_failure(ask_services):
    _, _, generate = ask_services
    generate.side_effect = httpx.ConnectError("Connection refused")

    response = client.post("/ask", json={"question": "Когда ответят?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Ошибка при обращении к Ollama"}


@pytest.mark.parametrize("payload", [{}, {"question": " \t "}, {"question": "а" * 2001}])
def test_ask_rejects_invalid_question_before_search(ask_services, payload):
    _, search, _ = ask_services

    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    search.assert_not_called()
