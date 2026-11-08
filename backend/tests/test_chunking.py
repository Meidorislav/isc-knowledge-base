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


@pytest.fixture
def word_count_tokenizer(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text: Mock(ids=text.split())
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)


def test_chunk_document_splits_long_section_by_paragraphs(word_count_tokenizer):
    paragraphs = [
        " ".join([word] * count)
        for word, count in [("форма", 240), ("заявка", 239), ("ответ", 310), ("архив", 175)]
    ]

    first, second = chunking.chunk_document("\n\n".join(paragraphs), "Инструкция")

    assert first.text == "\n\n".join(paragraphs[:2])
    assert second.text == "\n\n".join(paragraphs[2:])
    assert first.heading_path == second.heading_path == ["Инструкция"]
    assert first.token_count == 480
    assert second.token_count == 486


def test_chunk_document_keeps_paragraph_at_max_tokens(word_count_tokenizer):
    text = " ".join(["ответ"] * 799) + "\n\nПримечание."

    first, second = chunking.chunk_document(text, "Инструкция")

    assert first.token_count == 800
    assert second.text == "Примечание."


def test_chunk_document_splits_oversized_paragraph_by_lines(word_count_tokenizer):
    lines = [" ".join([word] * count) for word, count in [("форма", 240), ("заявка", 230), ("ответ", 350)]]
    paragraphs = ["Сначала вступление.", "\n".join(lines), "Наконец примечание."]

    drafts = chunking.chunk_document("\n\n".join(paragraphs), "Инструкция")

    assert [draft.text for draft in drafts] == [paragraphs[0], "\n".join(lines[:2]), lines[2], paragraphs[2]]
    assert [draft.token_count for draft in drafts] == [3, 471, 351, 3]


@pytest.mark.parametrize("prefix", ["", "Первая строка.\n"])
def test_chunk_document_reports_line_over_max_tokens(word_count_tokenizer, prefix):
    text = prefix + " ".join(["ответ"] * 800)

    with pytest.raises(NotImplementedError, match="длинной строки"):
        chunking.chunk_document(text, "Инструкция")
