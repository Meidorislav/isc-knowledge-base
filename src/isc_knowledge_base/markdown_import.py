"""Import local Markdown files as fragments with source metadata."""

import argparse
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from markdown_it import MarkdownIt


@dataclass(frozen=True, slots=True)
class FragmentSource:
    project_id: str
    path: str
    commit_sha: str | None
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class MarkdownFragment:
    id: str
    source: FragmentSource
    text: str


def _raise_walk_error(error: OSError) -> None:
    raise error


def _markdown_files(root: Path) -> Iterator[Path]:
    for directory, folders, filenames in root.walk(on_error=_raise_walk_error):
        if ".git" in folders:
            folders.remove(".git")
        for filename in filenames:
            path = directory / filename
            if path.suffix.lower() == ".md" and not path.is_symlink() and path.is_file():
                yield path


def _sections(
    markdown: str, parser: MarkdownIt,
) -> Iterator[tuple[tuple[str, ...], int, int, str]]:
    tokens = parser.parse(markdown)
    stack: list[tuple[int, str]] = []
    boundaries: list[tuple[tuple[str, ...], int, int]] = [((), 0, 0)]
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0 or token.map is None:
            continue
        level = int(token.tag[1:])
        title = " ".join(tokens[index + 1].content.split())
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = tuple(title for _, title in stack)
        start, body_start = token.map
        boundaries.append((heading_path, start, body_start))

    lines = markdown.split("\n")
    ends = [start for _, start, _ in boundaries[1:]] + [len(lines)]
    for (heading_path, start, body_start), end in zip(boundaries, ends, strict=True):
        while body_start < end and not lines[body_start].strip():
            body_start += 1
        while end > body_start and not lines[end - 1].strip():
            end -= 1
        yield heading_path, start + 1, end, "\n".join(lines[body_start:end])


def import_markdown(
    root: Path, *, project_id: str, commit_sha: str | None = None,
) -> Iterator[MarkdownFragment]:
    """Read a Markdown tree. Source line numbers are one-based and inclusive."""
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a Markdown source directory: {root}")
    project_id = project_id.strip()
    if not project_id:
        raise ValueError("Project ID must not be empty")
    if commit_sha is not None:
        if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit_sha):
            raise ValueError("Commit SHA must be a full Git commit hash")
        commit_sha = commit_sha.lower()

    parser = MarkdownIt("commonmark")
    for path in sorted(_markdown_files(root)):
        relative_path = path.relative_to(root).as_posix()
        markdown = path.read_text(encoding="utf-8-sig")
        occurrences: Counter[tuple[str, ...]] = Counter()
        for heading_path, start_line, end_line, text in _sections(markdown, parser):
            occurrences[heading_path] += 1
            if not text:
                continue
            key = (project_id, relative_path, heading_path, occurrences[heading_path])
            identity = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            yield MarkdownFragment(
                id=sha256(identity.encode("utf-8")).hexdigest(),
                source=FragmentSource(
                    project_id=project_id,
                    path=relative_path,
                    commit_sha=commit_sha,
                    heading_path=heading_path,
                    start_line=start_line,
                    end_line=end_line,
                ),
                text=text,
            )


def _write_jsonl(output: Path, fragments: Iterator[MarkdownFragment]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(dir=output.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / "fragments.jsonl"
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            for fragment in fragments:
                file.write(json.dumps(asdict(fragment), ensure_ascii=False) + "\n")
                count += 1
        temporary_path.chmod(0o600)
        temporary_path.replace(output)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root of the local Markdown tree")
    parser.add_argument("--project-id", required=True, help="stable identifier of the source project")
    parser.add_argument("--commit-sha", help="full commit SHA of the source snapshot, if known")
    parser.add_argument("--output", required=True, type=Path, help="path for the imported JSONL file")
    args = parser.parse_args()
    if args.output.suffix.lower() == ".md":
        parser.error("--output must be a JSONL file path, not a Markdown file")

    try:
        fragments = import_markdown(args.root, project_id=args.project_id, commit_sha=args.commit_sha)
        count = _write_jsonl(args.output, fragments)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Imported {count} fragments.")


if __name__ == "__main__":
    main()
