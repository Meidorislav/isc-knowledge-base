from uuid import UUID

from app.rag import build_messages, format_context
from app.search import RrfHit


def test_format_context_numbers_sources_in_search_order():
    hits = [
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

    assert format_context(hits) == (
        "[1] Подача заявки > Сроки рассмотрения\n"
        "Ответ приходит за три дня.\n\n"
        "[2] Архив заявки\n"
        "Копии хранят в архиве."
    )


def test_format_context_returns_empty_string_without_sources():
    assert format_context([]) == ""


def test_build_messages_separates_rules_from_question_and_context():
    hit = RrfHit(
        chunk_id=42,
        document_id=UUID("197a31c6-2fe8-45bf-a3f8-0fe074c43276"),
        heading_path=["Подача заявки", "Сроки рассмотрения"],
        text="Ответ приходит за три дня.",
        rrf_score=0.0328,
    )

    system, user = build_messages("Когда ответят на заявку?", [hit])

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
