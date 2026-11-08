import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from app import rag
from app.rag import InvalidModelAnswer, ModelAnswer, build_messages, format_context, resolve_answer
from app.search import RrfHit


@pytest.fixture
def hits():
    return [
        RrfHit(
            chunk_id=42,
            document_id=UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276"),
            heading_path=["Подача заявки", "Сроки рассмотрения"],
            text="Ответ приходит за три дня.",
            rrf_score=0.0328,
        ),
        RrfHit(
            chunk_id=7,
            document_id=UUID("362a0193-0fa0-489f-814d-fd3a57d99ee2"),
            heading_path=["Архив заявки"],
            text="Копии хранят в архиве.",
            rrf_score=0.0164,
        ),
    ]


def test_format_context_numbers_sources_in_search_order(hits):
    assert format_context(hits) == (
        "[1] Подача заявки > Сроки рассмотрения\n"
        "Ответ приходит за три дня.\n\n"
        "[2] Архив заявки\n"
        "Копии хранят в архиве."
    )


def test_format_context_returns_empty_string_without_sources():
    assert format_context([]) == ""


def test_build_messages_separates_rules_from_question_and_context(hits):
    system, user = build_messages("Когда ответят на заявку?", hits[:1])

    assert system["role"] == "system"
    assert "только по переданным источникам" in system["content"]
    assert '{"answer": null, "citations": []}' in system["content"]
    assert user == {
        "role": "user",
        "content": (
            "Вопрос\nКогда ответят на заявку?\n\n"
            "Источники\n[1] Подача заявки > Сроки рассмотрения\n"
            "Ответ приходит за три дня."
        ),
    }


@pytest.mark.parametrize(("answer", "citations"), [("В течение трёх дней.", [1]), (None, [])])
def test_generate_answer_requests_structured_chat(monkeypatch, answer, citations):
    messages = build_messages("Когда ответят на заявку?", [])
    response_content = json.dumps({"answer": answer, "citations": citations})

    def respond(request):
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3:1.7b"
        assert payload["messages"] == messages
        assert payload["format"] == ModelAnswer.model_json_schema()
        assert payload["stream"] is False
        return httpx.Response(200, json={"message": {"content": response_content}})

    client_type = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(rag.httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs))
    settings = SimpleNamespace(ollama_url="http://ollama.test", llm_model="qwen3:1.7b")
    monkeypatch.setattr(rag, "get_settings", lambda: settings)

    assert rag.generate_answer(messages) == ModelAnswer(answer=answer, citations=citations)


def test_model_answer_rejects_string_citation_numbers():
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate_json('{"answer": "Три дня.", "citations": ["1"]}')


def test_resolve_answer_looks_up_sources_by_number_and_keeps_citation_order(hits):
    result = resolve_answer(ModelAnswer(answer="Три дня.", citations=[2, 1, 2]), hits)

    assert result.answer == "Три дня."
    assert [source.number for source in result.sources] == [2, 1]
    assert [source.chunk_id for source in result.sources] == [7, 42]
    assert result.sources[0].document_id == hits[1].document_id
    assert result.sources[0].text == hits[1].text


@pytest.mark.parametrize(
    ("answer", "citations"),
    [("Три дня.", [0]), ("Три дня.", [-1]), ("Три дня.", [1, 3]), ("Три дня.", []), (" ", [1]), (None, [1])],
)
def test_resolve_answer_rejects_invalid_references_or_answer(hits, answer, citations):
    with pytest.raises(InvalidModelAnswer):
        resolve_answer(ModelAnswer(answer=answer, citations=citations), hits)


def test_resolve_answer_accepts_refusal(hits):
    result = resolve_answer(ModelAnswer(answer=None, citations=[]), hits)

    assert result.model_dump() == {"answer": None, "sources": []}


def test_answer_question_skips_generation_without_hits(monkeypatch):
    monkeypatch.setattr(rag, "search_hybrid", Mock(return_value=[]))
    generate = Mock()
    monkeypatch.setattr(rag, "generate_answer", generate)

    result = rag.answer_question(Mock(), "Как приготовить борщ?")

    assert result.model_dump() == {"answer": None, "sources": []}
    generate.assert_not_called()
