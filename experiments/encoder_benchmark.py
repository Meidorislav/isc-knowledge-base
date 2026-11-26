"""Compare a word embedding encoder before and after training on sample policies."""

import argparse
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import numpy as np
from numpy.typing import NDArray

from encoder_scoring import encode, tokenize, train_step


@dataclass(frozen=True)
class Question:
    text: str
    section_index: int


def load_bench(bench_dir: Path) -> tuple[list[str], list[Question], int]:
    sections = []
    section_indices = {}
    for path in sorted((bench_dir / "corpus").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        headings = list(re.finditer(r"^## (\d+)\. ([^\n]+)\n", text, flags=re.MULTILINE))
        boundaries = [heading.start() for heading in headings] + [len(text)]
        for heading, end in zip(headings, boundaries[1:], strict=True):
            section_indices[path.stem, heading[1]] = len(sections)
            sections.append(f"{heading[2]}\n{text[heading.end():end].strip()}")

    records = json.loads((bench_dir / "questions.json").read_text(encoding="utf-8"))
    questions = [
        Question(record["q"], section_indices[record["doc"], record["sec"]])
        for record in records
        if record["doc"] is not None
    ]
    return sections, questions, len(records) - len(questions)


def ranks_for_questions(
    embeddings: NDArray[np.float64],
    section_tokens: list[list[int]],
    questions: list[tuple[list[int], int]],
) -> list[int | None]:
    section_vectors = np.stack([encode(tokens, embeddings) for tokens in section_tokens])
    ranks = []
    for tokens, correct_index in questions:
        if not tokens:
            ranks.append(None)
            continue
        scores = section_vectors @ encode(tokens, embeddings)
        order = np.argsort(-scores, kind="stable")
        ranks.append(order.tolist().index(correct_index) + 1)
    return ranks


def print_metrics(label: str, ranks: list[int | None]) -> None:
    top_five = sum(rank is not None and rank <= 5 for rank in ranks)
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    print(f"{label}: топ-5 {top_five}/{len(ranks)}, MRR {mrr:.3f}")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    load_dotenv(project_dir / ".env")
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Command-line options override environment variables, then .env, then built-in defaults.",
    )
    default_bench = project_dir.parent / "bench"
    parser.add_argument("bench_dir", type=Path, nargs="?", default=default_bench)
    parser.add_argument(
        "--embedding-dim", type=int, default=os.getenv("ENCODER_EMBEDDING_DIM", "32"),
        help="number of coordinates per token vector (ENCODER_EMBEDDING_DIM)",
    )
    parser.add_argument(
        "--epochs", type=int, default=os.getenv("ENCODER_EPOCHS", "20"),
        help="number of passes over the training examples (ENCODER_EPOCHS)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=os.getenv("ENCODER_LEARNING_RATE", "0.1"),
        help="step size for updating token vectors (ENCODER_LEARNING_RATE)",
    )
    parser.add_argument(
        "--margin", type=float, default=os.getenv("ENCODER_MARGIN", "0.1"),
        help="required score gap between the correct and incorrect sections (ENCODER_MARGIN)",
    )
    parser.add_argument(
        "--seed", type=int, default=os.getenv("ENCODER_SEED", "42"),
        help="random seed for splitting, initialization and shuffling (ENCODER_SEED)",
    )
    args = parser.parse_args()
    if args.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive")
    if args.epochs < 0:
        parser.error("--epochs must be non-negative")
    if not isfinite(args.learning_rate) or args.learning_rate < 0:
        parser.error("--learning-rate must be finite and non-negative")
    if not isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be finite and non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    sections, questions, without_answer = load_bench(args.bench_dir)

    rng = np.random.default_rng(args.seed)
    gold_indices = sorted({question.section_index for question in questions})
    if len(gold_indices) < 2:
        parser.error("The benchmark must contain questions about at least two sections")

    # Keep questions with the same annotated answer section together.
    test_indices = set(rng.permutation(gold_indices)[: max(1, len(gold_indices) // 4)])
    train = [question for question in questions if question.section_index not in test_indices]
    test = [question for question in questions if question.section_index in test_indices]

    vocabulary = {word for section in sections for word in tokenize(section)}
    vocabulary.update(word for question in train for word in tokenize(question.text))
    word_ids = {word: index for index, word in enumerate(sorted(vocabulary))}

    def to_indices(text: str) -> list[int]:
        return [word_ids[word] for word in tokenize(text) if word in word_ids]

    section_tokens = [to_indices(section) for section in sections]
    train_tokens = [(to_indices(question.text), question.section_index) for question in train]
    test_tokens = [(to_indices(question.text), question.section_index) for question in test]

    section_words = [set(tokens) for tokens in section_tokens]
    triplets = []
    for tokens, correct_index in train_tokens:
        query_words = set(tokens)
        negative_index = max(
            (index for index in range(len(sections)) if index != correct_index),
            key=lambda index: len(query_words & section_words[index]),
        )
        triplets.append((tokens, section_tokens[correct_index], section_tokens[negative_index]))

    embeddings = rng.normal(0.0, 0.1, size=(len(word_ids), args.embedding_dim))
    initial_train = ranks_for_questions(embeddings, section_tokens, train_tokens)
    initial_test = ranks_for_questions(embeddings, section_tokens, test_tokens)

    for _ in range(args.epochs):
        for example_index in rng.permutation(len(triplets)):
            query, correct, incorrect = triplets[example_index]
            train_step(
                embeddings,
                query,
                correct,
                incorrect,
                margin=args.margin,
                learning_rate=args.learning_rate,
            )

    final_train = ranks_for_questions(embeddings, section_tokens, train_tokens)
    final_test = ranks_for_questions(embeddings, section_tokens, test_tokens)
    print(f"Фрагментов: {len(sections)}. Вопросов для обучения: {len(train)}, для проверки: {len(test)}.")
    print(f"Вопросов вне базы, пропущенных в эксперименте: {without_answer}.")
    print_metrics("Обучающие вопросы до обучения", initial_train)
    print_metrics("Обучающие вопросы после обучения", final_train)
    print_metrics("Отложенные вопросы до обучения", initial_test)
    print_metrics("Отложенные вопросы после обучения", final_test)
    print("\nМесто правильного пункта на отложенных вопросах (до → после):")
    for question, before, after in zip(test, initial_test, final_test, strict=True):
        print(f"{before or '—':>2} → {after or '—':<2}  {question.text}")


if __name__ == "__main__":
    main()
