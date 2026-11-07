"""Structure-aware chunking with a flat fallback. See docs/ingestion.md, section "Нарезка".

Algorithm (to be implemented):
  1. Detect headings: markdown `#` -> numbered `1.2 Title` -> heuristics (short line,
     no trailing period, surrounded by blank lines, caps). Build a section tree.
     No headings found -> the whole document is one section, heading_path = [title].
  2. Per section: fits the limit -> one chunk; too long -> recursive split
     (paragraphs -> lines -> sentences via `razdel`) with overlap; too short -> merge
     with the next sibling.
  3. search_text = " > ".join(heading_path) + "\n\n" + text.
"""

from dataclasses import dataclass
from functools import cache

from tokenizers import Tokenizer

CHUNKER_VERSION = "0.1"

TARGET_TOKENS = 500
MAX_TOKENS = 800
MIN_TOKENS = 100
OVERLAP_TOKENS = 60


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    heading_path: list[str]
    token_count: int
    structure: str  # markdown | numbered | heuristic | flat

    @property
    def search_text(self) -> str:
        return _format_search_text(self.heading_path, self.text)


def _format_search_text(heading_path: list[str], text: str) -> str:
    return " > ".join(heading_path) + "\n\n" + text


@cache
def _get_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(
        "BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
    )


def chunk_document(text: str, title: str) -> list[ChunkDraft]:
    if not text:
        return []

    heading_path = [title]
    search_text = _format_search_text(heading_path, text)
    token_count = len(_get_tokenizer().encode(search_text).ids)
    if token_count > MAX_TOKENS:
        raise NotImplementedError("Разбиение длинного текста пока не реализовано")

    draft = ChunkDraft(
        text=text,
        heading_path=heading_path,
        token_count=token_count,
        structure="flat",
    )
    return [draft]
