"""Train a mean pooling text encoder on one sample policy example."""

import re

import numpy as np
from numpy.typing import NDArray


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_word_indices(text: str, word_ids: dict[str, int]) -> list[int]:
    words = tokenize(text)
    if not words:
        raise ValueError("Text must contain at least one word")

    return [word_ids[word] for word in words]


def encode(indices: list[int], embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    return embeddings[indices].mean(axis=0)


def ranking_loss(correct_score: float, incorrect_score: float, margin: float) -> float:
    return max(0.0, margin + incorrect_score - correct_score)


def train_step(
    embeddings: NDArray[np.float64],
    question_indices: list[int],
    correct_indices: list[int],
    incorrect_indices: list[int],
    *,
    margin: float,
    learning_rate: float,
) -> None:
    question = encode(question_indices, embeddings)
    correct_section = encode(correct_indices, embeddings)
    incorrect_section = encode(incorrect_indices, embeddings)
    correct_score = float(question @ correct_section)
    incorrect_score = float(question @ incorrect_section)
    if ranking_loss(correct_score, incorrect_score, margin) == 0.0:
        return

    question_gradient = incorrect_section - correct_section
    correct_gradient = -question
    incorrect_gradient = question

    # Mean pooling divides the gradient among the tokens in each text.
    # add.at accumulates all contributions, including repeated word indices.
    gradients = np.zeros_like(embeddings)
    np.add.at(gradients, question_indices, question_gradient / len(question_indices))
    np.add.at(gradients, correct_indices, correct_gradient / len(correct_indices))
    np.add.at(gradients, incorrect_indices, incorrect_gradient / len(incorrect_indices))

    embeddings -= learning_rate * gradients


def main() -> None:
    # These hand-picked word vectors reproduce the scores from the first example.
    word_vectors = {
        "заявка": [0.7, 0.2],
        "на": [0.3, 0.1],
        "отпуск": [0.2, 0.6],
        "как": [0.6, 0.2],
        "оформить": [0.7, 0.4],
        "деление": [0.9, 0.5],
        "отпуска": [0.8, 0.5],
        "части": [0.8, 0.5],
    }
    word_ids = {word: index for index, word in enumerate(word_vectors)}
    embeddings = np.array(list(word_vectors.values()), dtype=np.float64)

    question_indices = get_word_indices("Заявка на отпуск?", word_ids)
    correct_indices = get_word_indices("Как оформить отпуск", word_ids)
    incorrect_indices = get_word_indices("Деление отпуска на части", word_ids)
    margin = 0.1

    def print_scores(stage: str) -> None:
        question = encode(question_indices, embeddings)
        correct_section = encode(correct_indices, embeddings)
        incorrect_section = encode(incorrect_indices, embeddings)
        correct_score = float(question @ correct_section)
        incorrect_score = float(question @ incorrect_section)
        loss = ranking_loss(correct_score, incorrect_score, margin)
        print(stage)
        print(f"Балл § 2 (подходящий пункт): {correct_score:.4f}")
        print(f"Балл § 4 (похожий пункт): {incorrect_score:.4f}")
        print(f"Потеря: {loss:.4f}")

    print_scores("Шаг 0 (до обучения)")
    for step in range(1, 21):
        train_step(
            embeddings,
            question_indices,
            correct_indices,
            incorrect_indices,
            margin=margin,
            learning_rate=0.1,
        )
        if step in (1, 5, 10, 15, 20):
            print()
            print_scores(f"Шаг {step}")


if __name__ == "__main__":
    main()
