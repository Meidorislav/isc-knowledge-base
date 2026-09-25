"""Raw file bytes -> normalized text (Markdown-ish, headings as `#` where known)."""

import io
import re
from pathlib import PurePath

SUPPORTED_FORMATS = {"txt", "md", "docx"}


class UnsupportedFormatError(ValueError):
    pass


def detect_format(filename: str) -> str:
    ext = PurePath(filename).suffix.lower().lstrip(".")
    if ext == "markdown":
        ext = "md"
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Формат '.{ext}' не поддерживается. Допустимо: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return ext


def parse(raw: bytes, file_format: str) -> str:
    match file_format:
        case "txt" | "md":
            text = _decode(raw)
        case "docx":
            text = _parse_docx(raw)
        case _:
            raise UnsupportedFormatError(file_format)
    return normalize(text)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


def _decode(raw: bytes) -> str:
    # Old Russian corporate files are often cp1251 rather than UTF-8.
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_HEADING_STYLE = re.compile(r"^(?:Heading|Заголовок)\s*(\d)$", re.IGNORECASE)


def _parse_docx(raw: bytes) -> str:
    """Paragraphs and tables in document order; Word heading styles become `#` headings."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(raw))
    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = item.style.name if item.style is not None else ""
            if style == "Title":
                blocks.append(f"# {text}")
            elif m := _HEADING_STYLE.match(style):
                blocks.append(f"{'#' * int(m.group(1))} {text}")
            else:
                blocks.append(text)
        elif isinstance(item, Table):
            blocks.append(_table_to_markdown(item))
    return "\n\n".join(blocks)


def _table_to_markdown(table) -> str:
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
