from unittest.mock import Mock

import pytest

from app.ingestion import chunking


@pytest.mark.parametrize(
    ("source", "body", "path", "structure"),
    [
        (
            "# Сроки рассмотрения\n\nОтвет приходит за три дня.",
            "Ответ приходит за три дня.",
            ["Подача заявки", "Сроки рассмотрения"],
            "markdown",
        ),
        (
            "Ответ приходит за три дня.",
            "Ответ приходит за три дня.",
            ["Подача заявки"],
            "flat",
        ),
        ("#срочно", "#срочно", ["Подача заявки"], "flat"),
    ],
)
def test_chunk_document_returns_one_chunk(monkeypatch, source, body, path, structure):
    tokenizer = Mock()
    tokenizer.encode.return_value.ids = [10, 20, 30]
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)

    [draft] = chunking.chunk_document(source, "Подача заявки")

    assert draft.text == body
    assert draft.heading_path == path
    assert draft.structure == structure
    assert draft.token_count == 3
    tokenizer.encode.assert_called_once_with(draft.search_text)


def test_chunk_document_with_only_heading_returns_no_chunks(monkeypatch):
    get_tokenizer = Mock()
    monkeypatch.setattr(chunking, "_get_tokenizer", get_tokenizer)

    assert chunking.chunk_document("# Сроки рассмотрения", "Подача заявки") == []
    get_tokenizer.assert_not_called()


def test_chunk_document_tracks_markdown_siblings(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.return_value.ids = [10, 20, 30]
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)
    text = (
        "# Рассмотрение\n## Сроки\nОтвет приходит за три дня.\n\n## Уведомление\nРешение отправляют на почту."
    )

    first, second = chunking.chunk_document(text, "Подача заявки")

    assert first.heading_path == ["Подача заявки", "Рассмотрение", "Сроки"]
    assert first.text == "Ответ приходит за три дня."
    assert second.heading_path == ["Подача заявки", "Рассмотрение", "Уведомление"]
    assert second.text == "Решение отправляют на почту."
    assert first.structure == second.structure == "markdown"


def test_chunk_document_preserves_intro_and_starts_new_branch(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.return_value.ids = [10, 20, 30]
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)
    text = "Перед подачей проверьте данные.\n# Рассмотрение\n### Сроки\nТри дня.\n# Архив\nОдин год."

    intro, deadline, archive = chunking.chunk_document(text, "Подача заявки")

    assert intro.heading_path == ["Подача заявки"]
    assert intro.text == "Перед подачей проверьте данные."
    assert deadline.heading_path == ["Подача заявки", "Рассмотрение", "Сроки"]
    assert archive.heading_path == ["Подача заявки", "Архив"]
