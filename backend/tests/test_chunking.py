from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

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
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)


@pytest.fixture
def word_count_without_overlap(monkeypatch, word_count_tokenizer):
    monkeypatch.setattr(chunking, "OVERLAP_TOKENS", 0)


def test_chunk_document_splits_long_section_by_paragraphs(word_count_without_overlap):
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


def test_chunk_document_keeps_paragraph_at_max_tokens(word_count_without_overlap):
    text = " ".join(["ответ"] * 799) + "\n\nПримечание."

    first, second = chunking.chunk_document(text, "Инструкция")

    assert first.token_count == 800
    assert second.text == "Примечание."


def test_chunk_document_splits_oversized_paragraph_by_lines(word_count_without_overlap):
    lines = [" ".join([word] * count) for word, count in [("форма", 240), ("заявка", 230), ("ответ", 350)]]
    paragraphs = ["Сначала вступление.", "\n".join(lines), "Наконец примечание."]

    drafts = chunking.chunk_document("\n\n".join(paragraphs), "Инструкция")

    assert [draft.text for draft in drafts] == [paragraphs[0], "\n".join(lines[:2]), lines[2], paragraphs[2]]
    assert [draft.token_count for draft in drafts] == [3, 471, 351, 3]


@pytest.mark.parametrize("prefix", ["", "Первая строка.\n"])
def test_chunk_document_splits_oversized_line_by_sentences(word_count_without_overlap, prefix):
    sentences = [" ".join(["день"] * count).capitalize() + "." for count in (240, 230, 350)]
    first_body = "  ".join(sentences[:2])
    text = prefix + first_body + " " + sentences[2]

    drafts = chunking.chunk_document(text, "Инструкция")

    expected = [prefix.rstrip()] if prefix else []
    assert [draft.text for draft in drafts] == expected + [first_body, sentences[2]]
    assert [draft.token_count for draft in drafts[-2:]] == [471, 351]


def test_chunk_document_keeps_abbreviation_inside_sentence(word_count_without_overlap):
    first_sentence = " ".join(["день"] * 480).capitalize() + "."
    second_sentence = "Справки и т. д. приложите к заявке " + " ".join(["заранее"] * 40) + "."
    third_sentence = " ".join(["ответ"] * 310).capitalize() + "."
    text = " ".join([first_sentence, second_sentence, third_sentence])

    first, second = chunking.chunk_document(text, "Инструкция")

    assert first.text == first_sentence
    assert second.text == f"{second_sentence} {third_sentence}"


@pytest.mark.parametrize("prefix", ["", "Первая строка.\n"])
def test_chunk_document_splits_oversized_sentence_at_word_boundaries(word_count_without_overlap, prefix):
    first_body = "День  " + " ".join(["день"] * 498)
    second_body = " ".join(["день"] * 400) + "."
    drafts = chunking.chunk_document(prefix + first_body + "\t" + second_body, "Инструкция")

    expected_intro = [prefix.rstrip()] if prefix else []
    assert [draft.text for draft in drafts] == expected_intro + [first_body, second_body]
    assert [draft.token_count for draft in drafts[-2:]] == [500, 401]


@pytest.fixture
def character_count_tokenizer(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: Mock(
        ids=range(len(text)), offsets=[(i, i + 1) for i in range(len(text))]
    )
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: tokenizer)


def test_chunk_document_splits_oversized_word_without_losing_characters(character_count_tokenizer):
    word = "абв" * 300
    first, second = chunking.chunk_document(word, "Тест")

    assert second.text[:60] == first.text[-60:]
    assert first.text + second.text[60:] == word
    assert first.token_count == 500
    assert second.token_count == 472


def test_chunk_document_reports_path_without_room_for_text(character_count_tokenizer):
    title = "абв" * 266

    with pytest.raises(ValueError, match="один символ"):
        chunking.chunk_document("День.", title)


def test_chunk_document_overlaps_within_section_only(word_count_tokenizer):
    first_body = " ".join(["форма"] * 480)
    second_body = " ".join(["ответ"] * 400)
    text = f"# Подача\n{first_body}\n\n{second_body}\n\n# Архив\nОдин год."

    first, second, archive = chunking.chunk_document(text, "Инструкция")

    assert first.text == first_body
    assert second.text == " ".join(["форма"] * 60) + "\n\n" + second_body
    assert second.token_count == 463
    assert archive.text == "Один год."


@pytest.mark.parametrize(("body_tokens", "overlap_tokens"), [(794, 5), (799, 0)])
def test_chunk_document_reduces_overlap_to_fit_limit(word_count_tokenizer, body_tokens, overlap_tokens):
    first_body = " ".join(["форма"] * 120)
    second_body = " ".join(["ответ"] * body_tokens)

    first, second = chunking.chunk_document(f"{first_body}\n\n{second_body}", "Инструкция")

    expected_overlap = " ".join(["форма"] * overlap_tokens) + "\n\n" if overlap_tokens else ""
    assert first.text == first_body
    assert second.text == expected_overlap + second_body
    assert second.token_count == 800
