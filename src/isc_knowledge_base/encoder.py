"""Inference and checkpoint storage for the mean-pooled word encoder."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import BinaryIO
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def format_fragment_text(heading_path: Sequence[str], body: str) -> str:
    return "\n".join([*heading_path, body])


def token_indices(text: str, word_ids: Mapping[str, int]) -> list[int]:
    return [word_ids[word] for word in tokenize(text) if word in word_ids]


def mean_pool(indices: list[int], embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    return embeddings[indices].mean(axis=0)


class WordMeanEncoder:
    """Encode text with a fixed vocabulary and one learned vector per word."""

    def __init__(self, vocabulary: Sequence[str], embeddings: NDArray[np.float64]) -> None:
        if (
            embeddings.ndim != 2
            or not embeddings.size
            or embeddings.shape[0] != len(vocabulary)
            or embeddings.dtype != np.float64
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("Expected a finite float64 matrix with one vector per word")

        self._vocabulary = tuple(vocabulary)
        self._word_ids = {word: index for index, word in enumerate(self._vocabulary)}
        if len(self._word_ids) != len(self._vocabulary):
            raise ValueError("Vocabulary must contain unique words")
        self._embeddings = embeddings.copy()

    @property
    def dimension(self) -> int:
        return self._embeddings.shape[1]

    def encode(self, text: str) -> NDArray[np.float64] | None:
        """Return a vector, or None if the text has no words in the vocabulary."""
        indices = token_indices(text, self._word_ids)
        if not indices:
            return None
        return mean_pool(indices, self._embeddings)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary_dir:
            checkpoint = Path(temporary_dir) / "encoder.npz"
            np.savez(
                checkpoint,
                version=1,
                vocabulary=np.asarray(self._vocabulary, dtype=np.str_),
                embeddings=self._embeddings,
            )
            checkpoint.replace(path)

    @classmethod
    def load(cls, path: Path | BinaryIO) -> "WordMeanEncoder":
        checkpoint = np.load(path, allow_pickle=False)
        if not isinstance(checkpoint, np.lib.npyio.NpzFile):
            raise ValueError("Expected an encoder checkpoint in NPZ format")
        with checkpoint:
            if checkpoint["version"].item() != 1:
                raise ValueError("Unsupported encoder checkpoint version")
            vocabulary = checkpoint["vocabulary"]
            if vocabulary.ndim != 1 or vocabulary.dtype.kind != "U":
                raise ValueError("Expected a one-dimensional vocabulary of strings")
            return cls(vocabulary.tolist(), checkpoint["embeddings"])


def load_checkpoint(path: Path) -> tuple[str, WordMeanEncoder]:
    """Load an encoder and identify it by the hash of the same checkpoint bytes."""
    content = path.read_bytes()
    try:
        encoder = WordMeanEncoder.load(BytesIO(content))
    except (BadZipFile, EOFError, KeyError) as error:
        raise ValueError("Invalid encoder checkpoint") from error
    return sha256(content).hexdigest(), encoder
