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

import re
from dataclasses import dataclass
from functools import cache

from razdel import sentenize
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


def _count_tokens(heading_path: list[str], text: str) -> int:
    search_text = _format_search_text(heading_path, text)
    return len(_get_tokenizer().encode(search_text).ids)


def _sentence_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    start = 0
    for sentence in sentenize(text):
        # Start at the previous sentence's end to keep the original spacing.
        fragments.append(text[start : sentence.stop])
        start = sentence.stop
    return fragments


def _find_word_cut(text: str, start: int, heading_path: list[str]) -> tuple[int, int]:
    end = start + 1
    token_count = _count_tokens(heading_path, text[start:end])
    if token_count > MAX_TOKENS:
        raise ValueError("Путь заголовков и один символ текста превышают лимит чанка")
    if token_count > TARGET_TOKENS:
        return end, token_count

    # Find a nearby upper bound without repeatedly tokenizing the entire remainder.
    while end < len(text):
        upper = min(len(text), start + 2 * (end - start))
        probe_count = _count_tokens(heading_path, text[start:upper])
        if probe_count > TARGET_TOKENS:
            break
        end, token_count = upper, probe_count
    else:
        return end, token_count

    while end + 1 < upper:
        middle = (end + upper) // 2
        probe_count = _count_tokens(heading_path, text[start:middle])
        if probe_count <= TARGET_TOKENS:
            end, token_count = middle, probe_count
        else:
            upper = middle
    return end, token_count


def _split_long_word(text: str, heading_path: list[str]) -> list[tuple[str, int]]:
    parts: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end, token_count = _find_word_cut(text, start, heading_path)
        parts.append((text[start:end], token_count))
        start = end
    return parts


def _split_oversized_text(text: str, heading_path: list[str], separator: str | None) -> list[tuple[str, int]]:
    joiner = ""
    if separator is None:
        fragments = _sentence_fragments(text)
    elif separator == " ":
        fragments = re.findall(r"\s*\S+", text)
    else:
        fragments = text.split(separator)
        joiner = separator

    parts: list[tuple[str, int]] = []
    current = ""
    current_count = 0

    for fragment in fragments:
        standalone = fragment.lstrip() if not joiner else fragment
        candidate = current + joiner + fragment if current else standalone
        candidate_count = _count_tokens(heading_path, candidate)

        if current and candidate_count > TARGET_TOKENS:
            parts.append((current, current_count))
            candidate = standalone
            candidate_count = _count_tokens(heading_path, candidate)

        if candidate_count > MAX_TOKENS:
            if separator == " ":
                parts.extend(_split_long_word(candidate, heading_path))
            else:
                next_separator = None
                if separator is None:
                    next_separator = " "
                elif separator == "\n\n" and "\n" in candidate:
                    next_separator = "\n"
                parts.extend(_split_oversized_text(candidate, heading_path, next_separator))
            current = ""
            current_count = 0
            continue

        current = candidate
        current_count = candidate_count

    if current:
        parts.append((current, current_count))
    return parts


def _prepend_overlap(previous: str, following: str, heading_path: list[str]) -> tuple[str, int] | None:
    tokenizer = _get_tokenizer()
    encoding = tokenizer.encode(previous, add_special_tokens=False)
    starts = sorted({start for start, _ in encoding.offsets[-OVERLAP_TOKENS:] if start < len(previous)})
    word_starts = [
        start for start in starts if start == 0 or previous[start - 1].isspace() or previous[start].isspace()
    ]

    for start in word_starts or starts:
        tail = previous[start:].lstrip()
        if not tail or len(tokenizer.encode(tail, add_special_tokens=False).ids) > OVERLAP_TOKENS:
            continue

        candidate = tail + following
        token_count = _count_tokens(heading_path, candidate)
        if token_count <= MAX_TOKENS:
            return candidate, token_count
    return None


def _apply_overlap(text: str, heading_path: list[str], parts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    if not OVERLAP_TOKENS or len(parts) < 2:
        return parts

    result: list[tuple[str, int]] = []
    previous = ""
    previous_end = 0
    for part, token_count in parts:
        start = text.index(part, previous_end)
        end = start + len(part)
        overlapped = _prepend_overlap(previous, text[previous_end:end], heading_path) if previous else None
        result.append(overlapped if overlapped is not None else (part, token_count))
        previous, previous_end = part, end
    return result


def chunk_document(text: str, title: str) -> list[ChunkDraft]:
    if not text:
        return []

    headings: list[tuple[int, str]] = []
    body_lines: list[str] = []
    sections: list[tuple[list[str], list[str]]] = [([title], body_lines)]

    for line in text.split("\n"):
        marker, _, heading = line.partition(" ")
        heading = heading.strip()
        if marker not in ("#", "##", "###", "####", "#####", "######") or not heading:
            body_lines.append(line)
            continue

        level = len(marker)
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, heading))

        heading_path = [title] + [name for _, name in headings]
        body_lines = []
        sections.append((heading_path, body_lines))

    structure = "markdown" if headings else "flat"
    drafts: list[ChunkDraft] = []
    for heading_path, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue

        token_count = _count_tokens(heading_path, body)
        if token_count > MAX_TOKENS:
            parts = _split_oversized_text(body, heading_path, "\n\n")
            parts = _apply_overlap(body, heading_path, parts)
        else:
            parts = [(body, token_count)]

        for part, part_count in parts:
            drafts.append(
                ChunkDraft(
                    text=part,
                    heading_path=heading_path.copy(),
                    token_count=part_count,
                    structure=structure,
                )
            )

    return drafts
