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
        return " > ".join(self.heading_path) + "\n\n" + self.text


def chunk_document(text: str, title: str) -> list[ChunkDraft]:
    raise NotImplementedError("chunker is the next step, see module docstring")
